import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict

MASTER_CONFIG = {
    'batch_size':16,          
    'context_window': 300,  # 最大文本/token长度
    'vocab_size':6400  ,   # 词表长度
    'd_model': 512,        # token维度
    'epochs': 1,        # 训练次数
    'log_interval': 10,      # 每10个 Iteration打印一次log
    'n_heads': 6,         # attention头数量
    'n_layers': 4,        # 隐藏层的数量
}


device = torch.device("cuda:0")

# LayerNorm 一种
class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super().__init__()
        # 注册训练层，命名为scale，并初始化为形状为layer_shape，所有值为1。
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))
    def forward(self, x):
        # 计算Frobenius范数（某个矩阵中所有元素的平方和再开方得到，该范数用来衡量矩阵的大小） RMS = 1/sqrt(N) * Frobenius
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw


# 代替 Transform的FNN(ReLU+线性层)，结合了Swish激活函数 和 门控线性单元（GLU） 
class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        # 门控的线性层
        self.linear_gate = nn.Linear(size, size) 
        # 门控结构主干线性层 
        self.linear = nn.Linear(size, size)
        # 初始化一个随机数作为beta系数  
        self.beta = torch.randn(1, requires_grad=True)  

        # nn.Parameter用于指定某一层参数为可学习的，即本来不能通过训练更改参数，现在变成了可以经过训练来更新的参数。
        self.beta = nn.Parameter(torch.ones(1))
        # 将随机数beta指定为一个名为beta的神经网络层
        self.register_parameter("beta", self.beta)

    def forward(self, x):
        # SwiGLU(x)=Swish(xW+b)⊗(xV+c)
        # Swish(x)=x⋅σ(βx)
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)  
        return out


# 旋转编码矩阵
def get_rotary_matrix(context_window, embedding_dim):
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    for position in range(context_window):
        for i in range(embedding_dim // 2):
            theta = 10000. ** (-2. * (i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2 * i, 2 * i] = np.cos(m_theta)
            R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
    return R


# 单头注意力
class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # 旋转编码矩阵
        self.R = get_rotary_matrix(config['context_window'], config['d_model']).to(device,dtype=torch.float16)


    def forward(self, x, return_attn_weights=False):
        # batch size, sequence length, dimension
        b, m, d = x.shape  

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        # 考虑到输入文本的长度，因此对位置编码矩阵在第一维度做截断，与文本长度一样。
        # 注意 q.shape: (bts,sl,dim) R.shape(bts,sl,dim) 主要对一个进行转置才能相乘，
        q_rotated = (torch.bmm(q.transpose(0, 1), self.R[:m])).transpose(0, 1)
        k_rotated = (torch.bmm(k.transpose(0, 1), self.R[:m])).transpose(0, 1)

        # attention核心公式 Attention(Q,K,V)
        activations = F.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=0.1, is_causal=True
        )

        # 如果return_attn_weights参数置为1，则需要对attention进行掩码
        if return_attn_weights:
            # 创建注意力掩码矩阵，其中torch.tril函数为：对于矩阵，取左下三角，剩下的都置0
            attn_mask = torch.tril(torch.ones((m, m)), diagonal=0)
            # 计算注意力机制的权重矩阵，并对最后一维度做归一化
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights

        return activations

# 多头注意力机制
class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])
        ])
        # 线性层输入形状：注意力机制的头数，乘以矩阵的维度 输出为：模型的embedding维度数
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])  
        self.dropout = nn.Dropout(0.1)  

    def forward(self, x):
        # (batch, sequence length, dimension)
        heads = [h(x) for h in self.heads]
        # 输入张量x经过多个头计算attention（同时，attention是已经覆盖了RoPE的），重新拼接成新的矩阵，重新放入变量x
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.attention = RoPEMaskedMultiheadAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):
        x = self.rms(x) 
        x = x + self.attention(x)
        x = self.rms(x) 
        x = x + self.feedforward(x)
        return x



class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )


    
    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_blocks(x)
        logits = self.ffn(x)
        # 推理
        if targets is None:
            return logits
        # 训练
        else:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss


if __name__ == "__main__":
    pass