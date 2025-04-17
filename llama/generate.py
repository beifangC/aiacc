from model import Llama
from model import MASTER_CONFIG
import torch
from tokenizer import MiniMindTokenizer
from torch.nn import functional as F


device = torch.device("cpu")
def generate(model, promote,config=MASTER_CONFIG, max_new_tokens=128):
    idx=list()
    tokenizer = MiniMindTokenizer("minimind_tokenizer")
    idx.append(tokenizer.encode(promote))
    generated_text=''

    idx= torch.tensor(idx, dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        # 输入截断
        logits = model(idx[:, -config['context_window']:])
    
        # 得到模型输出的结果
        last_time_step_logits = logits[:, -1, :]
   
        # 计算概率分布
        p = F.softmax(last_time_step_logits, dim=-1)
      
        # 随机采样
        # idx_next = torch.multinomial(p, num_samples=1)
        # 贪心采样
        idx_next = torch.argmax(p, dim=-1, keepdim=True)
    
        # 将新的idx通过张量拼接写入到解码序列中
        idx = torch.cat([idx, idx_next], dim=-1)

        new_token = tokenizer.decode(idx_next.tolist()[0])
        print(new_token, end='', flush=True)  # 立即输出，不换行
        generated_text += new_token

    # return [tokenizer.decode(x) for x in idx.tolist()]
    return generated_text



def generate_stream(
    model, 
    prompt, 
    config=MASTER_CONFIG, 
    max_new_tokens=128,
    return_tokens=False,  # 是否返回原始 token（默认返回文本）
):
    idx = []
    tokenizer = MiniMindTokenizer("minimind_tokenizer")
    idx.append(tokenizer.encode(prompt))
    
    idx = torch.tensor(idx, dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        # 输入截断
        logits = model(idx[:, -config['context_window']:])
        # 最后一个位置上的输出
        last_time_step_logits = logits[:, -1, :]
   
        p = F.softmax(last_time_step_logits, dim=-1)
      
        # torch.multinomial做的是随机采样
        idx_next = torch.multinomial(p, num_samples=1)
    
        idx = torch.cat([idx, idx_next], dim=-1)
        
        new_token = tokenizer.decode(idx_next.tolist()[0])
        
        if return_tokens:
            yield idx_next.item()  # 返回原始 token（整数）
        else:
            yield new_token  # 返回解码后的文本

if __name__=='__main__':
    model = Llama(MASTER_CONFIG).to(device)

    # 加载参数，注意设备类型，如果训练和推理的设备不同过需要显示指定
    model.load_state_dict(torch.load("model_weights.pth",map_location=torch.device('cpu')))
    model.eval()

    # res=generate(model,"注意力机制",max_new_tokens=100)
    # print(res)

    for token_text in generate_stream(model, "注意力机制", max_new_tokens=100):
        print(token_text, end='', flush=True) 
    print("\n---")


