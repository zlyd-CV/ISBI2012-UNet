# ISBI2012-UNet

## 一、项目介绍
+ 本项目基于Pytorch构建了3个模型，分别为ResNet-34、UNet、UNet++，针对原论文中提到的ISBI2012竞赛。
+ 本项目不包含模型部署，仅使用训练好的模型权重对测试集（不含标签）的图像进行预测。
+ 数据集展示

![图像](https://github.com/zlyd-CV/Photos-Are-Used-To-Others-Repository/blob/3de129a4f320bd29a0c5d747dc66be4e426e9cc6/ISBI2012-UNet/%E8%AE%AD%E7%BB%83%E9%9B%86%E5%9B%BE%E5%83%8F.png)
![掩码](https://github.com/zlyd-CV/Photos-Are-Used-To-Others-Repository/blob/3de129a4f320bd29a0c5d747dc66be4e426e9cc6/ISBI2012-UNet/%E6%8E%A9%E7%A0%81%E5%9B%BE%E5%83%8F.png)
+ 论文图和模型示意图：

![原论文结构图](https://github.com/zlyd-CV/Photos-Are-Used-To-Others-Repository/blob/827f7dc20d8d88025fbb534298b8f85cf0c1bbf4/ISBI2012-UNet/UNet.png)

## 二、内容介绍
+ 目录结构
```python
ISBI-2012/
├── dataset/
│   └── images/
│       ├── train/
│       │   ├── images/    # 训练图像 (30张)
│       │   └── label/     # 训练标签 (30张)
│       └── test/          # 测试图像 (30张)
├── model/
│   ├── config.py          # 配置文件
│   ├── get_dataset.py     # 数据加载
│   ├── networks.py        # 3种网络结构
│   ├── train.py           # 训练函数
│   ├── test.py            # 测试函数
│   └── main.py            # 主程序入口(全部功能只需要运行该行代码即可)
├── model_checkpoints/     # 模型权重保存目录(权重字典太大已剔除)
├── test_predict/          # 测试结果保存目录
└── requirements.txt       # 依赖包列表
```
+ 本项目包含：
+ requirements.txt：需要的包版本，建议使用虚拟环境
 ```txt
pip install -r requirement.txt
```

## 三、结果展示
+ UNet(两张图片上下分别对应SGD和Adam结果)
![UNet+SGD](https://github.com/zlyd-CV/Photos-Are-Used-To-Others-Repository/blob/827f7dc20d8d88025fbb534298b8f85cf0c1bbf4/ISBI2012-UNet/best_UNet_training_curves_SGD.png)
![UNet+Adam](https://github.com/zlyd-CV/Photos-Are-Used-To-Others-Repository/blob/827f7dc20d8d88025fbb534298b8f85cf0c1bbf4/ISBI2012-UNet/best_UNet_training_curves_Adam.png)
+ UNet++(两张图片上下分别对应SGD和Adam结果)
![UNet++ +SGD](https://github.com/zlyd-CV/Photos-Are-Used-To-Others-Repository/blob/827f7dc20d8d88025fbb534298b8f85cf0c1bbf4/ISBI2012-UNet/best_UNetPlusPlus_training_curves_SGD.png)
![UNet++ +Adam](https://github.com/zlyd-CV/Photos-Are-Used-To-Others-Repository/blob/827f7dc20d8d88025fbb534298b8f85cf0c1bbf4/ISBI2012-UNet/best_UNetPlusPlus_training_curves_Adam.png)
+ ResNet34(两张图片上下分别对应SGD和Adam结果)
![ResNet34+SGD](https://github.com/zlyd-CV/Photos-Are-Used-To-Others-Repository/blob/827f7dc20d8d88025fbb534298b8f85cf0c1bbf4/ISBI2012-UNet/best_ResNet34_training_curves_SGD.png)
![ResNet34+Adam](https://github.com/zlyd-CV/Photos-Are-Used-To-Others-Repository/blob/827f7dc20d8d88025fbb534298b8f85cf0c1bbf4/ISBI2012-UNet/best_ResNet34_training_curves_Adam.png)

## 四、部分资源下载地址
+ pytorch官网下载带cuda的pytorch：https://pytorch.org
+ Anaconda官网下载地址：https://anaconda.org/anaconda/conda
+ 原论文地址：[Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation[C]//International Conference on Medical image computing and computer-assisted intervention. Cham: Springer international publishing, 2015: 234-241.](https://arxiv.org/pdf/1505.04597)


## 五、本项目结论总结
+ UNet具有较高的性能，处理每个epoch的速度几乎比UNet++和ResNet34快一倍，并且取得了训练数据集上的最优性能。
+ 使用动量SGD和Adam二者结果很接近，没有优劣一说。
+ Adam在大多数情况下学习率为0.0001和0.00005会有较好效果，通常不需要配备学习率调度器；而SGD初始值可设置大一些(0.1)，采样固定步长衰减(尽管本代码采取了其他方式)来逐步逼近局部最优值。
