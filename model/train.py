import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from config import Config


def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs,use_SGD=False):
    """训练模型的函数"""
    model.to(device)
    model.train()  # 设置模型为训练模式
    # 初始化2个列表，用于存储每个epoch的损失和像素准确率
    train_losses = []
    train_pixel_accuracies = []

    for epoch in range(num_epochs):
        print('-' * 50)

        best_loss = float('inf')

        if os.path.exists(Config.MODEL_SAVE_PATH) and ((epoch % 5) - 1 == 0 or epoch == 0):
            model_parameters = torch.load(Config.MODEL_SAVE_PATH)
            model.load_state_dict(model_parameters['model'])
            # optimizer.load_state_dict(model_parameters['optimizer'])
            # scheduler.load_state_dict(model_parameters['scheduler'])
            best_loss = model_parameters['best_loss']
            print(f"加载历史最优模型成功,最优损失{best_loss:.4f}")
        else:
            print(f"没有找到保存的模型或者没到加载计时({epoch % 5 - 1}/{5})，加载失败")

        epoch_loss = 0.0
        epoch_pixel_acc = 0.0  # 累计像素准确率

        train_pbar = tqdm(
            train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', leave=False)
        for index, sample in enumerate(train_pbar):
            # sample是一个包含图像和标签的字典
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            optimizer.zero_grad()  # 清除之前的梯度

            outputs = model(image)  # 前向传播
            loss = criterion(outputs, label)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            # 获取逐像素的语义准确率
            # 对于二分类分割，将输出转为二值预测 (阈值0.5)
            preds = (outputs > 0.5).float()
            pixel_acc = (preds == label).float().mean().item()

            epoch_loss += loss.item()
            epoch_pixel_acc += pixel_acc  # 累加像素准确率
            train_pbar.set_postfix({'loss': loss.item(), 'pixel_acc': pixel_acc})

        # 计算平均损失和平均像素准确率
        epoch_loss = epoch_loss / len(train_loader)
        epoch_pixel_acc = epoch_pixel_acc / len(train_loader)

        # 由于SGD和Adam使用不同的学习率调度器，分别处理
        if use_SGD:
            scheduler.step(epoch_loss)  # 在每个epoch结束后更新学习率,基于训练损失(报错好像是传了带epoch的参数)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        tqdm.write(
            f'\nEpoch [{epoch + 1}/{num_epochs}] : Average Loss: {epoch_loss:.4f}, Pixel Accuracy: {epoch_pixel_acc:.4f} ({epoch_pixel_acc * 100:.2f}%), Learning Rate: {current_lr:.6f}')

        # 将每个epoch的损失和像素准确率添加到列表中
        train_losses.append(epoch_loss)
        train_pixel_accuracies.append(epoch_pixel_acc)

        if epoch_loss <= best_loss and epoch % 5 == 0:  # 每5次保留一次,根据use_SGD选择保存路径
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': epoch_loss,
            }, f"{Config.MODEL_SAVE_PATH+('SGD.pth' if use_SGD else 'Adam.pth')}")
            print(f"最优模型已保存")
        else:
            print(f"当前epoch测试损失升高，或者没到保存计时({epoch % 5}/{5})，即将加载历史最优模型...")
        print('-' * 50)


    print('Training finished.')
    return train_losses, train_pixel_accuracies

# 使用matplotlib绘制训练损失和像素准确率曲线
def plot_training_curves(train_losses, train_pixel_accuracies,use_SGD=False):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.xlim(1, len(train_losses))
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制像素准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_pixel_accuracies, 'r-', label='Pixel Accuracy')
    plt.xlim(1, len(train_pixel_accuracies))
    plt.title('Pixel Accuracy over Epochs')
    plt.ylim(top=1.0) # 设置y轴上限为1.0
    plt.xlabel('Epochs')
    plt.ylabel('Pixel Accuracy')
    plt.legend()

    # 保存图像
    if use_SGD:
        plt.savefig(f'{Config.MODEL_SAVE_PATH}training_curves_SGD.png')
    else:
        plt.savefig(f'{Config.MODEL_SAVE_PATH}training_curves_Adam.png')

    plt.tight_layout()
    plt.show()
