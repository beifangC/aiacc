from dataset import SFTDataset
from model import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import time


def train(model, optimizer, train_loader,scheduler=None, config=MASTER_CONFIG, print_logs=False):
    start_time = time.time()
    total_set=len(train_loader)

    for step, (X, Y) in enumerate(train_loader):
        optimizer.zero_grad()
        xs = X.to(device=device)
        ys = Y.to(device=device)

        logits, loss = model(xs, targets=ys)

        # 反向传播更新权重参数，更新学习率优化器
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()
 
        if 0 == 0:
            batch_time = time.time() - start_time
            if print_logs:
                print(f"Epoch {1} | Step {step}/{total_set} | val loss {loss:.3f} | Time {batch_time:.3f}")
            start_time = time.time()
        break
        # if scheduler:
        #     print("lr: ", scheduler.get_lr())
    return 

if __name__=='__main__':
    llama = Llama(MASTER_CONFIG).to(device)
    ckp = 'model_weights.pth'
    state_dict = torch.load(ckp, map_location=device)
    tokenizer = AutoTokenizer.from_pretrained('minimind_tokenizer')
    optimizer = torch.optim.Adam(
        llama.parameters(),   
        lr=0.0001  
    )

    train_ds = SFTDataset('sft_mini_512.jsonl', tokenizer, max_length=MASTER_CONFIG['context_window'])

    train_loader = DataLoader(
        train_ds,
        batch_size=MASTER_CONFIG['batch_size'],
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    print('start sft train')
    train(llama, optimizer,train_loader,print_logs=True)
    torch.save(llama.state_dict(), 'full_sft_512.pth')
