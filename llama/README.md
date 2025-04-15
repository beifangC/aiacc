# 大模型学习

模仿llama，对原始的transformer的一些改进
- 使用旋转编码代替正余弦编码
- 使用RMSNorm
- SwiGLU代替ReLU

训练数据
https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files


以及配套tokenizer
./minimind_tokenizer




# 0412模型参数
```python
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
```