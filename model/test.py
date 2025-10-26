# 预测test里面的数据集，将结果保存在test_predict文件夹中
import os
import torch
from torchvision import transforms
from get_dataset import ISBIDataset
from config import Config
from networks import UNet
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

def test_model(model, test_loader, device, save_dir):
    """测试模型的函数"""
    model.to(device)
    model.eval()  # 设置模型为评估模式

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for index, sample in tqdm(enumerate(test_loader)):
            image = sample['image'].to(device)
            image_name = 'test_image_' + str(index) + '.png'  # 假设测试集图像按顺序命名

            outputs = model(image)  # 前向传播
            preds = (outputs > 0.5).float()  # 二值化预测结果

            # 将预测结果转换为PIL图像并保存
            pred_image = preds.squeeze().cpu().numpy() * 255  # 转为0-255范围
            pred_image = Image.fromarray(pred_image.astype('uint8'))
            pred_image.save(os.path.join(save_dir, image_name))


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = ISBIDataset(Config.DATA_DIR, mode='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1)
    model_parameters = torch.load(Config.MODEL_SAVE_PATH)
    model.load_state_dict(model_parameters['model'])
    print("加载模型成功")

    test_model(model, test_loader, Config.DEVICE, Config.TEST_PREDICT_DIR)