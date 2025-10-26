import torch


class Config:
    DATA_DIR = '../dataset/images/'  # 数据集路径
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL = 'UNet'  # 选择模型: 'UNet', 'ResNet34', 'UNetPlusPlus'
    BATCH_SIZE = 1
    EPOCHS = 30
    LEARNING_RATE_SGD = 0.1
    LEARNING_RATE_Adam = 0.0001
    TEST_PREDICT_DIR = '../test_predict'
    MODEL_SAVE_PATH = f'../model_checkpoints/best_{MODEL}_'  # 模型保存路径
    PLOT_SAVE_PATH = f'../model_checkpoints/{MODEL}_'  # 训练曲线保存路径
