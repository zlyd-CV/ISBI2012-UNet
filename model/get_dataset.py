import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import transforms
from config import Config


class ISBIDataset(Dataset):
    """ISBI 2012 神经元分割数据集"""

    def __init__(self, dataset_path, mode='train', transform=None):
        """
        初始化数据集

        Args:
            dataset_path: 数据根目录，例如 'D:/Projects/Python/ISBI-2012/dataset/images'
            mode: 'train' 或 'test'
            transform: 数据增强/预处理函数
        """
        self.data_dir = dataset_path
        self.mode = mode
        self.transform = transform

        if mode == 'train':
            self.images_dir = os.path.join(dataset_path, 'train', 'images')
            self.labels_dir = os.path.join(dataset_path, 'train', 'label')
            # 获取训练集所有图像文件名
            self.image_files = sorted(
                [f for f in os.listdir(self.images_dir) if f.endswith('.tif')])
        elif mode == 'test':
            self.images_dir = os.path.join(dataset_path, 'test')
            self.labels_dir = None
            # 获取测试集所有图像文件名
            self.image_files = sorted(
                [f for f in os.listdir(self.images_dir) if f.endswith('.tif')])
        else:
            raise ValueError("mode 必须是 'train' 或 'test'")

        print(f"{mode.upper()} 数据集: 找到 {len(self.image_files)} 张图像")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """获取单个样本"""
        # 读取图像
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('L')  # 转为灰度图

        if self.mode == 'train':
            # 读取标签
            label_path = os.path.join(self.labels_dir, img_name)
            label = Image.open(label_path).convert('L')

            # 应用数据增强（在转换为numpy之前）
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)
                # 确保标签是二值的 (0 或 1)
                label = (label > 0.5).float()
            else:
                # 如果没有transform，手动转换为tensor
                image = np.array(image, dtype=np.float32)
                label = np.array(label, dtype=np.float32)
                image = torch.from_numpy(image).unsqueeze(0) / 255.0
                label = torch.from_numpy(label).unsqueeze(0) / 255.0
                # 确保标签是二值的 (0 或 1)
                label = (label > 0.5).float()

            # 返回图像和标签的字典
            return {'image': image, 'label': label}
        else:
            # 测试集只返回图像
            if self.transform:
                image = self.transform(image)
            else:
                # 如果没有transform，手动转换为tensor
                image = np.array(image, dtype=np.float32)
                image = torch.from_numpy(image).unsqueeze(0) / 255.0

            return {'image': image}


def get_train_dataloader(dataset_path, batch_size=1, shuffle=True, transform=None):
    """获取训练数据加载器"""
    train_dataset = ISBIDataset(
        dataset_path, mode='train', transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )
    return train_loader


def get_test_dataloader(dataset_path, batch_size=1, shuffle=False, transform=None):
    """获取测试数据加载器"""
    test_dataset = ISBIDataset(dataset_path, mode='test', transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )
    return test_loader


if __name__ == '__main__':
    # 测试数据加载
    data_dir = Config.DATA_DIR
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("-" * 50)
    print("训练集数据加载")
    print("-" * 50)
    train_loader = get_train_dataloader(
        data_dir, batch_size=2, transform=transform)

    for index, train_iter in enumerate(train_loader):
        images = train_iter['image']
        labels = train_iter['label']

        print(f"  图像形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  图像值范围: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  标签值范围: [{labels.min():.3f}, {labels.max():.3f}]")

        if index >= 0:  # 只显示前3个batch
            break

    print("-" * 50)
    print("测试集数据加载")
    print("-" * 50)
    test_loader = get_test_dataloader(
        data_dir, batch_size=2, transform=transform)

    for index, test_iter in enumerate(test_loader):
        images = test_iter['image']

        print(f"图像形状: {images.shape}")
        print(f"图像值范围: [{images.min():.3f}, {images.max():.3f}]")

        if index >= 0:  # 只显示前3个batch
            break
