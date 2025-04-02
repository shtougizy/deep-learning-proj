#
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# from torch.optim.lr_scheduler import CosineAnnealingLR
#
# # 定义用于 CIFAR-10 的 ResNet 模型
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes * BasicBlock.expansion, kernel_size=3, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(planes * BasicBlock.expansion)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes * BasicBlock.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_planes,
#                     planes * BasicBlock.expansion,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(planes * BasicBlock.expansion),
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
# class ResNet_CIFAR(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet_CIFAR, self).__init__()
#         self.in_planes = 16
#
#         self.conv1 = nn.Conv2d(
#             3, 16, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(16)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#         self.linear = nn.Linear(64 * block.expansion, num_classes)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(
#                 block(self.in_planes, planes, stride)
#             )
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
#
# def ResNet110():
#     return ResNet_CIFAR(BasicBlock, [18, 18, 18])
#
# # 主程序
# def main():
#     # 使用 GPU 加速
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f'Using device: {device}')
#
#     # 数据增强和归一化
#     mean = (0.4914, 0.4822, 0.4465)
#     std  = (0.2023, 0.1994, 0.2010)
#
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])
#
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])
#
#     # 加载数据集
#     batch_size = 128
#
#     trainset = torchvision.datasets.CIFAR10(
#         root='./data', train=True, download=True, transform=transform_train
#     )
#     trainloader = torch.utils.data.DataLoader(
#         trainset, batch_size=batch_size, shuffle=True, num_workers=2
#     )
#
#     testset = torchvision.datasets.CIFAR10(
#         root='./data', train=False, download=True, transform=transform_test
#     )
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=100, shuffle=False, num_workers=2
#     )
#
#     # 定义模型
#     model = ResNet110().to(device)
#     # 打印模型参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f'Total parameters: {total_params}')
#
#     # 定义损失函数和优化器
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(
#         model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
#     )
#     scheduler = CosineAnnealingLR(optimizer, T_max=50)
#
#     # 为混合精度训练准备
#     scaler = torch.cuda.amp.GradScaler()
#
#     # 存储训练过程中的损失和准确率
#     train_losses = []
#     train_accuracies = []
#     test_accuracies = []
#
#     num_epochs = 50
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         for batch_idx, (inputs, targets) in enumerate(trainloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#
#             optimizer.zero_grad()
#
#             # 混合精度训练
#             with torch.cuda.amp.autocast():
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#
#         scheduler.step()
#
#         train_loss = running_loss / len(trainloader)
#         train_acc = 100. * correct / total
#         train_losses.append(train_loss)
#         train_accuracies.append(train_acc)
#
#         # 在测试集上评估
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for batch_idx, (inputs, targets) in enumerate(testloader):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()
#
#         test_acc = 100. * correct / total
#         test_accuracies.append(test_acc)
#
#         print(f'Epoch: {epoch+1}/{num_epochs} | '
#               f'Train Loss: {train_loss:.4f} | '
#               f'Train Acc: {train_acc:.2f}% | '
#               f'Test Acc: {test_acc:.2f}%')
#
#     # 绘制损失曲线
#     plt.figure()
#     plt.plot(train_losses, label='Train Loss')
#     plt.title('Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()
#
#     # 绘制训练和测试准确率曲线
#     plt.figure()
#     plt.plot(train_accuracies, label='Train Accuracy')
#     plt.plot(test_accuracies, label='Test Accuracy')
#     plt.title('Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()
#     plt.show()
#
#     # 打印最终测试准确率
#     print(f'Final Test Accuracy: {test_accuracies[-1]:.2f}%')
#
# if __name__ == '__main__':
#     main()
#
# ============================================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
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

if __name__ == '__main__':
    def main():
        # 检查设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 数据增广
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪
            transforms.RandomHorizontalFlip(),     # 随机水平翻转
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),  # 自动数据增强
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # 数据加载
        batch_size = 128
        num_workers = 6  # 根据CPU核心数调整

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

        # 实例化模型
        net = WideResNet(depth=28, widen_factor=10, num_classes=10, dropRate=0.3)
        net = net.to(device)

        if device == 'cuda':
            net = nn.DataParallel(net)  # 多GPU并行
            cudnn.benchmark = True

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

        # AMP训练
        scaler = GradScaler()

        # 训练
        num_epochs = 150

        train_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(num_epochs):
            net.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                with autocast():
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            epoch_loss = running_loss / len(trainset)
            epoch_acc = 100. * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            # 在测试集上评估
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
            test_accuracies.append(test_acc)

            print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%')

        # 绘制训练日志
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('train_loss.png')

        plt.figure()
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig('accuracy.png')

        # 保存模型
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), './checkpoint/wideresnet.pth')
        print('模型已保存至 ./checkpoint/wideresnet.pth')

    main()

# import torch
# from torch.utils.tensorboard.summary import image
# import torchvision
# import torch.nn.functional as F
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torch.optim as optim
#
# from torch.utils.tensorboard import SummaryWriter
#
# myWriter = SummaryWriter('./tensorboard/log/')
#
# myTransforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#
# #  load
# train_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=True, download=True,
#                                              transform=myTransforms)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
#
# test_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=False, download=True,
#                                             transform=myTransforms)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)
#
# # 定义模型
# myModel = torchvision.models.resnet50(pretrained=True)
# # 将原来的ResNet18的最后两层全连接层拿掉,替换成一个输出单元为10的全连接层
# inchannel = myModel.fc.in_features
# myModel.fc = nn.Linear(inchannel, 10)
#
# # 损失函数及优化器
# # GPU加速
# myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# myModel = myModel.to(myDevice)
#
# learning_rate = 0.001
# myOptimzier = optim.SGD(myModel.parameters(), lr=learning_rate, momentum=0.9)
# myLoss = torch.nn.CrossEntropyLoss()
#
# for _epoch in range(10):
#     training_loss = 0.0
#     for _step, input_data in enumerate(train_loader):
#         image, label = input_data[0].to(myDevice), input_data[1].to(myDevice)  # GPU加速
#         predict_label = myModel.forward(image)
#
#         loss = myLoss(predict_label, label)
#
#         myWriter.add_scalar('training loss', loss, global_step=_epoch * len(train_loader) + _step)
#
#         myOptimzier.zero_grad()
#         loss.backward()
#         myOptimzier.step()
#
#         training_loss = training_loss + loss.item()
#         if _step % 10 == 0:
#             print('[iteration - %3d] training loss: %.3f' % (_epoch * len(train_loader) + _step, training_loss / 10))
#             training_loss = 0.0
#             print()
#     correct = 0
#     total = 0
#     # torch.save(myModel, 'Resnet50_Own.pkl') # 保存整个模型
#     myModel.eval()
#     for images, labels in test_loader:
#         # GPU加速
#         images = images.to(myDevice)
#         labels = labels.to(myDevice)
#         outputs = myModel(images)  # 在非训练的时候是需要加的，没有这句代码，一些网络层的值会发生变动，不会固定
#         numbers, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Testing Accuracy : %.3f %%' % (100 * correct / total))
#     myWriter.add_scalar('test_Accuracy', 100 * correct / total)