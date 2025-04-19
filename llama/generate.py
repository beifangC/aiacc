from model import Llama
from model import MASTER_CONFIG
import torch
from tokenizer import MiniMindTokenizer
from torch.nn import functional as F
from transformers import AutoTokenizer


device = torch.device("cuda:0")
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
      
        # 随机采样
        idx_next = torch.multinomial(p, num_samples=1)
        # 贪心采样
        # idx_next = torch.argmax(p, dim=-1, keepdim=True)

        # idx_next=top_k_sampling(last_time_step_logits, top_k=50, temperature=1.0)

        idx = torch.cat([idx, idx_next], dim=-1)
        
        new_token = tokenizer.decode(idx_next.tolist()[0])
        
        if return_tokens:
            yield idx_next.item()  # 返回原始 token（整数）
        else:
            yield new_token  # 返回解码后的文本



def top_k_sampling(logits, top_k=50, temperature=1.0):
    """
    logits: Tensor，形状为 (vocab_size,)，模型输出的未归一化logits
    top_k: int，保留概率最高的top_k个token
    temperature: float，温度系数，控制采样的随机程度，temperature越低越确定
    
    返回：采样得到的token索引（int）
    """
    # 1. 温度调节
    logits = logits / temperature

    # 2. 只保留top_k个logits，其他置为极小值（-inf）
    top_k = min(top_k, logits.size(-1))  # 防止top_k大于词表大小
    values, indices = torch.topk(logits, top_k)
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits[indices] = logits[indices]

    # 3. 计算softmax概率
    probs = F.softmax(filtered_logits, dim=-1)

    # 4. 根据概率分布采样
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token.item()



def generate_stream_sft(
    model, 
    prompt, 
    config=MASTER_CONFIG, 
    max_new_tokens=128,
):
    tokenizer= AutoTokenizer.from_pretrained('./minimind_tokenizer')
    messages = []
    messages.append({"role": "user", "content": prompt})
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )[-config['context_window']:] 

    idx = torch.tensor(tokenizer(new_prompt)['input_ids'], device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = model(idx[:, -config['context_window']:])
        last_time_step_logits = logits[:, -1, :]

        p = F.softmax(last_time_step_logits, dim=-1)
        idx_next = torch.argmax(p, dim=-1, keepdim=True)

        # idx_next=top_k_sampling(last_time_step_logits, top_k=10, temperature=1.0)
        if idx_next.item()==2:
            break

        idx = torch.cat([idx, idx_next], dim=-1)
        
        new_token = tokenizer.decode(idx_next.tolist()[0])
        
        yield new_token  # 返回解码后的文本



if __name__=='__main__':
    model = Llama(MASTER_CONFIG).to(device)

    # 加载参数，注意设备类型，如果训练和推理的设备不同过需要显示指定
    model.load_state_dict(torch.load("full_sft_512.pth",map_location='cuda:0'))
    model.eval()

    # res=generate(model,"介绍一下北京",max_new_tokens=100)
    # print(res)

    for token_text in generate_stream_sft(model, "请介绍一下自己。", max_new_tokens=200):
        print(token_text, end='', flush=True) 
    print('')
