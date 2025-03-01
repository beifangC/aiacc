import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义类别名称（CIFAR-10的10个类别）
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 数据预处理（注意：需要反向归一化以还原图像）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
])

# 加载数据集（仅加载训练集用于展示）
# 速度慢，使用其他方式提前下载
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True)  # 批量大小为4，随机打乱

# 定义一个反向归一化的函数（将张量从[-1,1]还原到[0,1]）
def imshow(img):
    img = img / 2 + 0.5     # 反向归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 将PyTorch的通道优先格式转为Matplotlib的HWC格式
    plt.axis('off')

# 从数据加载器中获取一个批量（4张图片）
dataiter = iter(trainloader)
images, labels = next(dataiter)  # 或 dataiter.next()（旧版本PyTorch）

# 显示图片和标签
plt.figure(figsize=(8, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    imshow(images[i])
    plt.title(f'Label: {classes[labels[i]]}')
plt.tight_layout()
plt.show()