import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os

# 定义 WideResNet 模型（与训练时定义的模型结构相同）
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

        # 1st conv
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # Blocks
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], BasicBlock, stride=1, dropRate=dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], BasicBlock, stride=2, dropRate=dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], BasicBlock, stride=2, dropRate=dropRate)
        # Global average pooling and classifier
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

def evaluate():
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据归一化
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 数据加载
    batch_size = 100
    num_workers = 4  # 根据CPU核心数调整

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    # 实例化模型
    net = WideResNet(depth=28, widen_factor=10, num_classes=10, dropRate=0.3)

    # 使用 nn.DataParallel 包裹模型
    if device == 'cuda':
        net = nn.DataParallel(net)

    net = net.to(device)

    # 加载模型权重
    model_path = 'wideresnet1.pth'
    if os.path.isfile(model_path):
        net.load_state_dict(torch.load(model_path))
        print('模型已加载')
    else:
        print('模型文件不存在，请检查路径')
        return

    # 评估
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total
    print(f'Test Accuracy: {test_acc:.2f}%')

if __name__ == '__main__':
    evaluate()



    # # 实例化模型
    # net = WideResNet(depth=28, widen_factor=10, num_classes=10, dropRate=0.3)
    # net = net.to(device)
    #
    # # 加载模型权重
    # model_path = './checkpoint/wideresnet.pth'
    # if os.path.isfile(model_path):
    #     net.load_state_dict(torch.load(model_path))
    #     print('模型已加载')
    # else:
    #     print('模型文件不存在，请检查路径')
    #     return