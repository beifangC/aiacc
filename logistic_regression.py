import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据,乳腺癌数据集
data = load_breast_cancer()
X = data.data
# 良性（0）还是恶性（1）
y = data.target.reshape(-1, 1)  # 调整形状为 (n_samples, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建 DataLoader
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 单层线性层

    def forward(self, x):
        # 使用 Sigmoid 函数输出概率
        return torch.sigmoid(self.linear(x))   # 输出形状 (batch_size, 1)

# 初始化模型
input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)

# 损失函数：二元交叉熵损失（内置 Sigmoid）
criterion = nn.BCELoss()  # 或者使用 nn.BCEWithLogitsLoss（不需要手动加 Sigmoid）

# 优化器：随机梯度下降（SGD）或 Adam
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-2)  # weight_decay 是 L2 正则化参数



num_epochs = 1000

for epoch in range(num_epochs):
    model.train()  # 训练模式
    for batch_X, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每 100 个 epoch 打印损失
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")



        model.eval()  # 评估模式
with torch.no_grad():
    y_pred_prob = model(X_test_tensor).numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)  # 阈值设为0.5

# 计算准确率和混淆矩阵
from sklearn.metrics import accuracy_score, confusion_matrix
print("准确率:", accuracy_score(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))