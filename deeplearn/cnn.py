import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 超参数
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.002

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
# 60,000 张彩色图片，分为 ​10 个类别，每个类别 6,000 张。32x32 像素
# 类别：飞机（airplane）、汽车（automobile）、鸟（bird）、猫（cat）、鹿（deer）、狗（dog）、青蛙（frog）、马（horse）、船（ship）、卡车（truck）。
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True)

# 测试数据集
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False)

# 定义简单CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)       # 输入通道3，输出通道6，卷积核5x5
        self.pool = nn.MaxPool2d(2, 2)        # 2x2最大池化
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)     # 全连接层计算
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # 展平
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型、损失函数、优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 训练循环
train_losses, val_accs = [], []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 每个epoch结束后验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    
    # 记录数据
    train_loss = running_loss / len(trainloader)
    train_losses.append(train_loss)
    val_accs.append(acc)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, Val Acc: {acc:.2f}%")


# 保存整个模型（结构+参数）
# torch.save(model, './cifar10_model.pth')

# 或仅保存模型参数（推荐方式，兼容性更好）
torch.save(model.state_dict(), './cifar10_model_weights.pth')

# 可视化结果
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(val_accs, label='Validation Accuracy')
plt.legend()
plt.show()