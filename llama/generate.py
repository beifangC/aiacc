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
    

    idx= torch.tensor(idx, dtype=torch.long, device=device)
    print(idx.shape)
    for _ in range(max_new_tokens):
        # 输入截断
        logits = model(idx[:, -config['context_window']:])
    
        # 得到模型输出的结果
        last_time_step_logits = logits[:, -1, :]
   
        # 计算概率分布
        p = F.softmax(last_time_step_logits, dim=-1)
      
        # 根据概率分布计算下一个token，这里使用 torch.multinomial做的是随机采样
        idx_next = torch.multinomial(p, num_samples=1)
    
        # 将新的idx通过张量拼接写入到解码序列中
        idx = torch.cat([idx, idx_next], dim=-1)
    return [tokenizer.decode(x) for x in idx.tolist()]


if __name__=='__main__':
    model = Llama(MASTER_CONFIG).to(device)

    # 加载参数，注意设备类型，如果训练和推理的设备不通过需要显示指定
    model.load_state_dict(torch.load("model_weights.pth",map_location=torch.device('cpu')))
    model.eval()

    res=generate(model,"你好")
    print(res)