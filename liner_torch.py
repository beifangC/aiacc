import torch
import matplotlib.pyplot as plt

# 使用torch写的线性回归，可控制的内容更多


plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统常用字体

# 设置随机种子（保证可复现）
torch.manual_seed(123)

# 生成数据
X = torch.linspace(0, 5, 100).reshape(-1, 1)  # 特征 (100个样本, 1维)
true_weight = 2.0
true_bias = 1.0
y = true_weight * X + true_bias + torch.randn(X.shape) * 0.5  # 添加高斯噪声

# 可视化数据
# plt.scatter(X.numpy(), y.numpy(), label='原始数据')
# plt.plot(X.numpy(), (true_weight * X + true_bias).numpy(), 'r--', label='真实关系')
# plt.legend()
# plt.show()



class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)  # 输入1维，输出1维

    def forward(self, x):
        return self.linear(x)

# 初始化模型
model = LinearRegression()


criterion = torch.nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 随机梯度下降


num_epochs = 500
loss_history = []

for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # 反向传播与优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    # 记录损失
    loss_history.append(loss.item())
    
    # 每10轮打印一次
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


        # 获取训练后的参数
weight = model.linear.weight.item()
bias = model.linear.bias.item()
print(f"训练后的权重: {weight:.2f}, 偏置: {bias:.2f}")
print(f"真实权重: {true_weight}, 真实偏置: {true_bias}")

# 绘制拟合直线
predicted = model(X).detach().numpy()
plt.scatter(X.numpy(), y.numpy(), label='原始数据')
plt.plot(X.numpy(), predicted, 'g-', label='模型预测')
plt.legend()
plt.show()

# 绘制损失下降曲线
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('loss变化')
plt.show()