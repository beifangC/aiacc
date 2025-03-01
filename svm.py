import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 加载数据并预处理
data = load_breast_cancer()
X, y = data.data, data.target

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 转换为 PyTorch Tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # 形状 (n_samples, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 定义 SVM 模型
class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super(LinearSVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 单层线性模型

    def forward(self, x):
        return self.linear(x)  # 输出原始分数（未经过 Sigmoid）

# 3. 自定义 Hinge Loss（支持 L2 正则化）
def hinge_loss(output, target, model, C=1.0):
    # Hinge Loss: max(0, 1 - y * f(x)) 的平均值
    loss = torch.mean(torch.clamp(1 - target * output, min=0))
    
    # L2 正则化（对应 SVM 的 1/(2C) ||w||^2）
    l2_reg = 0.5 * torch.sum(model.linear.weight**2)
    
    return loss + l2_reg / C

# 初始化模型和优化器
input_dim = X.shape[1]
model = LinearSVM(input_dim)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练循环
epochs = 1000
C = 1.0  # 正则化参数（类似 sklearn 的 C）

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(X_train)
    loss = hinge_loss(outputs, y_train, model, C=C)
    
    # 反向传播与优化
    loss.backward()
    optimizer.step()
    
    # 每 100 轮打印损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 预测与评估
model.eval()
with torch.no_grad():
    train_output = model(X_train)
    test_output = model(X_test)
    train_pred = (train_output > 0).float()  # 分数 > 0 视为正类
    test_pred = (test_output > 0).float()

# 计算准确率
train_acc = (train_pred == y_train).float().mean()
test_acc = (test_pred == y_test).float().mean()

print(f'Train Accuracy: {train_acc.item() * 100:.2f}%')
print(f'Test Accuracy: {test_acc.item() * 100:.2f}%')

