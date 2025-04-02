# 深度学习-技术文档

仓库名：deep-learning-proj  
时间：2024.11-2024.12  
在这段时间内完成了合计五个完整的实验，本篇readme将给出所有实验的实验要求，并完成完整的技术文档撰写。  
将持续更新直至完成。  
最后撰写日期：2025.4.2


---

所有实验的实验环境：

GTX4060laptop（8gb显存） i5-13500H  ddr5内存32g  Windows11  anaconda-python3.11  pycharm community edition

环境配置方法：

访问anaconda官网，下载适用于windows11的64位安装包，选择python3.11   
下载完成后双击安装包打开，选择非系统盘（D盘），路径中检查后发现没有汉字  
勾选“Add Anaconda to my PATH environment variable”，或将安装完成后的/bin文件夹添加至环境变量。  
在cmd中输入以下命令以检验是否安装成功：  
     conda --version  # 显示版本号即表示成功  
     conda list        # 查看已安装的包  
在pycharm中，选择settings-python interpreter，如果上述的anaconda环境安装成功的话，那么可以选择anaconda环境下的python3.11解释器了。选择该解释器，即环境配置完成。

---
## 实验一
CIFAR-10 数据集分类实现 95% 以上准确率   
1. 数据集：CIFAR-10（包含 10 类彩色图像，每类 6000 张图片，32x32 分辨率）。    
2. 要求：基于所学的网络设计（可选择ResNet、VGG、或自行设计的卷积网络结构）在CIFAR-10上 训练出一个模型，验证集/测试集准确率达到95%以上。  
3. 技术点： • 数据加载与增广（随机裁剪、翻转、归一化）。 • 使用GPU加速、AMP训练。 • 使用学习率调度器提高收敛速度。 • 根据需要选择合适的优化器（如Adam或SGD+Momentum）。  
4. 输出：最终模型的测试集准确率，以及训练日志（loss、accuracy 曲线）。

数据集：Tiny ImageNet 数据集

### 一、代码结构概述

本代码基于PyTorch框架实现了一个WideResNet模型。为完成题目的要求，代码设计包含以下核心组件：

**1.模型架构**：
1. 残差块 `BasicBlock`：通过`conv_shortcut`实现跳跃连接解决梯度消失问题，支持了输入输出维度不一致时的通道调整  
2. 网络块结构 `NetworkBlock`：堆叠多个残差块，每个块的首层可设置步长以进行下采样  
3. 宽度因子`widen_factor`：增加了通道数，以提升模型容量  
4. 丢弃神经元 `Dropout`：为防止过拟合，在残差块中随机丢弃部分神经元  

**2.数据增强与预处理：**
- 使用了`transforms.Compose`以组合以下操作：  
	随机水平翻转 `RandomHorizontalFlip`和随机裁剪 `RandomCrop`以增加数据多样性  
	为加速训练的收敛，归一化 `Normalize`将像素值映射到[-1，1]区间  
	
**3.训练优化技术：**
1. 混合精度训练AMP ：通过`autocast、GradScaler`减少显存占用、加速计算  
2. 学习率调度器：采用余弦退火 `CosineAnnealingLR` 动态调整学习率以避免陷入局部最优解  
3. 优化器：使用 `SGD`优化器并配置动量 `momentum`和权重衰减 `weight_decay`以平衡收敛速度与稳定性  
4. GPU加速：设置`cudnn.benchmark=True`以启动了CuDNN自动优化  

### 二、核心技术分析

#### 1. 残差块 `BasicBlock`的实现


```
class BasicBlock(nn.Module):   
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):   
        super().__init__()    
        self.equalInOut = in_planes == out_planes  
        # 通道调整（当输入输出维度不一致时）  
        self.conv_shortcut = None if self.equalInOut else \  
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)  
        # 主路径  
        self.bn1 = nn.BatchNorm2d(in_planes)  
        self.relu1 = nn.ReLU(inplace=True)  
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2d(out_planes)  
        self.relu2 = nn.ReLU(inplace=True)  
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)  
        self.dropRate = dropRate

    def forward(self, x):  
        if not self.equalInOut:  
            # 输入输出通道不一致时，先做BN+ReLU再卷积  
            x = self.relu1(self.bn1(x))  
            out = self.conv1(x)  
        else:  
            # 通道一致时，直接通过主路径  
            out = self.relu1(self.bn1(x))  
            out = self.conv1(out)  
        # 第二层处理  
        out = self.relu2(self.bn2(out))  
        if self.dropRate > 0:  
            out = F.dropout(out, p=self.dropRate, training=self.training)  
        out = self.conv2(out)  
        # 跳跃连接（通过conv_shortcut调整维度）  
        return self.conv_shortcut(x) + out if not self.equalInOut else x + out
 ```


残差块的核心是跳跃连接 `Skip Connection`，其输出公式为： 输出=主路径+跳跃连接。在前向传播过程中，把`ReLU`激活函数置于`BatchNorm`的后面，这种顺序的目的是让卷积层的输入数据先经过标准化和非线性激活，从而保证卷积操作的输入分布更稳定。如果先使用 `ReLU `再 `BN`，`ReLU` 会先将负值归零，导致 `BN` 的输入数据分布偏向非对称，这会削弱` BN `的标准化效果，因为 `BN `依赖数据的均值和方差来归一化。总结来说引入残差块的原因就是为了稳定输入分布、保护跳跃链接信息、并减少死神经元。

1. 在`BasicBlock`中，输入特征通过两个卷积层（`conv1`、`conv2`）处理，并通过跳跃连接与原始输入相加。若输入输出通道数不同（`equalInOut=False`），则通过`conv_shortcut`调整通道维度，以实现恒等映射
2. 残差连接：最终输出为主路径输出 + 跳跃连接，确保梯度直接回传
3. 在第二个激活后随机丢弃神经元（`dropRate=0.3`），增强泛化性

#### 2. 网络宽度扩展`WideResNet`

```
class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropRate=0.3):
    super().__init__()
    self.in_planes = 16
    nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]  # 宽度扩展关键
    # 初始卷积层
    self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
    # 构建3个阶段（stage）
    self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], BasicBlock, 1, dropRate)
    self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], BasicBlock, 2, dropRate)
    self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], BasicBlock, 2, dropRate)
    # 全局池化与分类层
    self.bn1 = nn.BatchNorm2d(nChannels[3])
    self.relu = nn.ReLU(inplace=True)
    self.fc = nn.Linear(nChannels[3], num_classes)`
```

由于模型深度的继续增加在精度方面已经不能取得良好的回报，因此研究者们就开始从模型的宽度方面开始思索，最后发现了减少模型深度的同时增加模型的宽度也可以提高模型的精度，本实验中遵循了该设计理念。
-  `widen_factor`会将通道数扩展为原来的倍数。
- `depth`：总层数计算为`(28-4)/6=4`，每个阶段包含4个残差块


#### 3. 混合精度训练`AMP`

```
`scaler = GradScaler()  # 梯度缩放器
`#训练循环片段
`with autocast():  # 自动混合精度上下文
  ``  outputs = model(inputs)
   `` loss = criterion(outputs, targets)
`scaler.scale(loss).backward()  # 缩放损失
`scaler.step(optimizer)  # 更新参数
`scaler.update()  # 调整缩放因子
```

混合精度训练，指的是单精度`float`和半精度 `float16` 混合。FP16与FP32（单精度）相比，优点在于内存占用更少、由于GPU的适配而计算更快，但会出现下溢出现象。所以要在
- `autocast()`：在前向计算中自动将部分运算转为FP16（如卷积），其余保持FP32
- `GradScaler`：反向传播时放大梯度（对计算出来的`loss`值进行`scale`），防止FP16精度下的小梯度值下溢

#### 4.  学习率调度器 `CosineAnnealingLR`

```
`optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
`scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  # 200个epoch
`scheduler.step()#每个epoch后更新
```

学习率是控制每次迭代更新中梯度下降步幅大小的参数。在反向传播过程中，模型通过计算损失函数的梯度来更新参数，而学习率决定了沿梯度方向迈出多大的一步。本实验中，使用余弦退火法`CosineAnnealing`调整学习率时，在训练过程中，学习率会以余弦曲线的方式逐渐增加，接近训练结束时再慢慢回到一个较低的学习率，这种非线性的调整方式有助于在训练的不同阶段更好地探索参数空间，从而找到更好的全局最优解。
- 初始学习率设为0.1，按余弦曲线从最大值衰减到0。
- 同时配合SGD的动量（`momentum=0.9`）避免震荡


#### 5.数据增强的实现

```
`transform_train = transforms.Compose([
	`transforms.RandomCrop(32, padding=4),      # 随机裁剪（数据增广）
	`transforms.RandomHorizontalFlip(),        # 水平翻转
    `transforms.ToTensor(),
    `transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 归一化
`])
```

数据增强是依赖从现有数据生成新的数据点来人为地增加数据量的过程，增加训练集的多样性，同时也提升了训练出的模型的性能。本实验中使用了随机裁剪`RandomCrop`与水平翻转`Normalize`方法。
- `RandomCrop`：在32x32图像边缘填充4像素后随机裁剪，模拟物体位置变化
- `Normalize`：根据CIFAR-10数据集的均值和标准差标准化输入，加速收敛



###  三、实验总结

![](https://github.com/shtougizy/deep-learning-proj/blob/main/deeplearning_1/pic/6a47ad5ec60cc4bb80ce2613e4943530.png)
![](https://github.com/shtougizy/deep-learning-proj/blob/main/deeplearning_1/pic/93993eef081e5c2298dd1d870ac07821.png?raw=true)
![](https://github.com/shtougizy/deep-learning-proj/blob/main/deeplearning_1/pic/a3205c28d460a2fe191540dc3405c7ab.png?raw=true)
![](https://github.com/shtougizy/deep-learning-proj/blob/main/deeplearning_1/pic/accuracy.png?raw=true)
![](https://github.com/shtougizy/deep-learning-proj/blob/main/deeplearning_1/pic/train_loss.png?raw=true)
总结来说，本实验在150个epochs内达到了最高97%的准确率，虽然仍有较大的进步空间，但圆满地完成了既定的95%准确率的要求。


