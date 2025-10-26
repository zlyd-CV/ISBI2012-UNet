import torch
from torchvision.transforms import transforms
from get_dataset import *
from config import Config
from networks import *
from train import *
from test import *

if __name__ == "__main__":
    """
    在运行前您需要检查以下内容：
    1. 在config.py中正确设置了Config类的各项参数(尤其是MODEL)
    1. 数据集路径(Config.DATA_DIR)是否正确，确保包含训练和测试数据。
    2. 模型类型(Config.MODEL)是否设置为 'UNet', 'ResNet34' 或 'UNetPlusPlus'。
    3. 检查train_model和plot_training_curves函数里的use_SGD参数与传入的优化器对应，这会直接影响保存的路径。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader = get_train_dataloader(Config.DATA_DIR, batch_size=Config.BATCH_SIZE, transform=transform)
    test_loader = get_test_dataloader(Config.DATA_DIR, batch_size=Config.BATCH_SIZE, transform=transform)
    if Config.MODEL == 'ResNet34':
        model = ResNet34(BasicBlock, [3, 4, 6, 3],num_classes=1)
    elif Config.MODEL == 'UNet':
        model = UNet(in_channels=1, out_channels=1)
    elif Config.MODEL == 'UNetPlusPlus':
        model = UNetPlusPlus(in_channels=1, out_channels=1)
    elif Config.MODEL == 'ResNet34':
        model = ResNet34(BasicBlock, [3, 4, 6, 3], num_classes=1)
    else:
        raise ValueError("Unsupported model type. Choose either 'UNet', 'ResNet34', or 'UNetPlusPlus'.") # 添加错误处理
    criterion = torch.nn.BCELoss()
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE_SGD, momentum=0.9, weight_decay=5e-4)  # 原论文中使用0.99的动量,但实践不如0.9好
    scheduler_SGD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_SGD, mode='min', factor=0.5, patience=3, min_lr=0.00001) # 当指标停止提升时，减少学习率
    optimizer_Adam = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE_Adam, weight_decay=5e-4)
    scheduler_Adam = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_Adam, T_max=10, eta_min=0.00001) # 余弦退火学习率调度器
    use_SGD = True  # 设置为True使用SGD优化器，False使用Adam优化器
    if use_SGD:
        train_loss_list,train_acc_list = train_model(model, train_loader, criterion, optimizer_SGD,scheduler_SGD, Config.DEVICE, Config.EPOCHS,use_SGD=use_SGD)
    else:
        train_loss_list,train_acc_list = train_model(model, train_loader, criterion, optimizer_Adam,scheduler_Adam, Config.DEVICE, Config.EPOCHS,use_SGD=use_SGD)
    test_model(model, test_loader, Config.DEVICE, Config.TEST_PREDICT_DIR)
    plot_training_curves(train_loss_list, train_acc_list,use_SGD=use_SGD)

