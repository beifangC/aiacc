from model import *
from matplotlib import pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
from dataset import PretrainDataset
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


# 构建训练函数
def train(model, optimizer, train_loader,scheduler=None, config=MASTER_CONFIG, print_logs=False):
    start_time = time.time()
    total_set=len(train_loader)
    if scheduler:
        print('scheduler on')
    for step, (X, Y) in enumerate(train_loader):
        optimizer.zero_grad()
        xs = X.to(device=device)
        ys = Y.to(device=device)

        logits, loss = model(xs, targets=ys)

        # 反向传播更新权重参数，更新学习率优化器
        loss.backward()
        optimizer.step()

        # 学习率调度器
        if scheduler:
            scheduler.step()
        # log
        if step % config['log_interval'] == 0:
            batch_time = time.time() - start_time
            if print_logs:
                print(f"Epoch {1} | Step {step}/{total_set} | val loss {loss:.3f} | Time {batch_time:.3f}")
            start_time = time.time()

        # if scheduler:
        #     print("lr: ", scheduler.get_lr())

    return 


if __name__=='__main__':
    llama = Llama(MASTER_CONFIG).to(device)
    tokenizer = AutoTokenizer.from_pretrained('minimind_tokenizer')

    optimizer = torch.optim.Adam(
        llama.parameters(),   
        lr=0.001  
    )

    train_ds = PretrainDataset('pretrain_hq.jsonl', tokenizer, max_length=MASTER_CONFIG['context_window'])

    train_loader = DataLoader(
        train_ds,
        batch_size=MASTER_CONFIG['batch_size'],
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    T_max = 30000  # 周期长度（总epoch数或step数）
    eta_min = 0.0001  # 最低学习率η_min
    # 余弦退火
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    # 启动训练
    print('tarin start')
    train(llama, optimizer,train_loader,scheduler,print_logs=True)
    print('train end')
    # 保存仅参数
    torch.save(llama.state_dict(), 'model_weights.pth')