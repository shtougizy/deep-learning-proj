###### 深度学习-技术文档

仓库名：deep-learning-proj
时间：2024.11-2024.12
在这段时间内完成了合计五个完整的实验，本篇readme将给出所有实验的实验要求，并完成完整的技术文档撰写。
将持续更新直至完成。
最后撰写日期：2025.3.14


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
#### 实验一
基于ResNet 的自编码器实现 

• 目标：设计一个基于ResNet架构的自编码器网络，完成图像降维与重建。
• 具体要求：
1. 编码器网络基于ResNet，将输入图像的特征降维到 16 维瓶颈表示。
2. 解码器网络将16维特征重建为原始图像大小（64×64）。 
3. 使用 PSNR（峰值信噪比）作为重建精度的评估指标。
• 数据：Tiny ImageNet 数据集。
• 损失函数：使用均方误差（MSE）作为重建损失。
• 优化器：选择合适的优化器（如Adam），设置合理的学习率和训练周期。

数据集：Tiny ImageNet 数据集
