import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 定义 WideResNet 模型
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.equalInOut = in_planes == out_planes
        self.conv_shortcut = None if self.equalInOut else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropRate = dropRate

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.relu1(self.bn1(x))
            out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = self.conv2(out)
        if self.equalInOut:
            return x + out
        else:
            return self.conv_shortcut(x) + out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(nb_layers):
            if i == 0:
                layers.append(block(in_planes, out_planes, stride, dropRate))
            else:
                layers.append(block(out_planes, out_planes, 1, dropRate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropRate=0.3):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert ((depth - 4) % 6 == 0), 'Depth should be 6n+4'
        n = (depth - 4) // 6

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], BasicBlock, stride=1, dropRate=dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], BasicBlock, stride=2, dropRate=dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], BasicBlock, stride=2, dropRate=dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, out.size(1))
        return self.fc(out)

def load_model(model_path, device):
    net = WideResNet(depth=28, widen_factor=10, num_classes=10, dropRate=0.3).to(device)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    state_dict = torch.load(model_path, map_location=device)

    # 移除 state_dict 中可能存在的 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # 移除 'module.'
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    return net

def imshow(img, mean, std):
    # denormalize
    img = img.numpy().transpose((1, 2, 0))
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    plt.imshow(img)

if __name__ == '__main__':
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据增强与归一化
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 加载测试数据集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=True, num_workers=2)

    # 加载模型
    model_path = 'wideresnet1.pth'
    net = load_model(model_path, device)

    # 类别名称
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 从测试集中获取一批数据
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # 显示图片
    plt.figure(figsize=(10, 4))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        imshow(images[i].cpu(), mean, std)
        plt.axis('off')
    plt.suptitle('Input Images', fontsize=16)
    plt.show()

    # 将图片传入模型进行预测
    images = images.to(device)
    with torch.no_grad():
        outputs = net(images)
        _, predicted = outputs.max(1)

    # 输出真实标签与预测标签
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(5)))
    print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(5)))

    # 展示预测结果图
    plt.figure(figsize=(10, 4))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        imshow(images[i].cpu(), mean, std)
        plt.title(classes[predicted[i]])
        plt.axis('off')
    plt.suptitle('Predicted Classes', fontsize=16)
    plt.show()