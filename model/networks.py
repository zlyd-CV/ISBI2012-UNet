import torch
import torch.nn as nn
from torchinfo import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class TransposeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransposeConv, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)  # 瓶颈层，位于编码器和解码器之间，是网络的最深层次

        self.up4 = TransposeConv(1024, 512)
        self.up_conv4 = DoubleConv(1024, 512)
        self.up3 = TransposeConv(512, 256)
        self.up_conv3 = DoubleConv(512, 256)
        self.up2 = TransposeConv(256, 128)
        self.up_conv2 = DoubleConv(256, 128)
        self.up1 = TransposeConv(128, 64)
        self.up_conv1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 在每个阶段下采样之前，保存特征图用于跳跃连接
        # 阶段 1: 512x512 -> 256x256
        skip1 = self.down1(x)  # x:(1, 1, 512, 512) -> skip1:(1, 64, 512, 512)
        pool1 = self.pool(skip1)  # skip1:(1, 64, 512, 512) -> pool1:(1, 64, 256, 256)

        # 阶段 2: 256x256 -> 128x128
        skip2 = self.down2(pool1)  # pool1:(1, 64, 256, 256) -> skip2:(1, 128, 256, 256)
        pool2 = self.pool(skip2)  # skip2:(1, 128, 256, 256) -> pool2:(1, 128, 128, 128)

        # 阶段 3: 128x128 -> 64x64
        skip3 = self.down3(pool2)  # pool2:(1, 128, 128, 128) -> skip3:(1, 256, 128, 128)
        pool3 = self.pool(skip3)  # skip3:(1, 256, 128, 128) -> pool3:(1, 256, 64, 64)

        # 阶段 4: 64x64 -> 32x32
        skip4 = self.down4(pool3)  # pool3:(1, 256, 64, 64) -> skip4:(1, 512, 64, 64)
        pool4 = self.pool(skip4)  # skip4:(1, 512, 64, 64) -> pool4:(1, 512, 32, 32)

        # 瓶颈层
        bottle = self.bottleneck(pool4)  # pool4:(1, 512, 32, 32) -> bottle:(1, 1024, 32, 32)

        # 在每个阶段，进行上采样，然后与编码器对应层的输出进行拼接
        # 解码阶段 4: 32x32 -> 64x64(上采样)
        up4_out = self.up4(bottle)  # bottle:(1, 1024, 32, 32) -> up4_out:(1, 512, 64, 64)
        merged4 = torch.cat([up4_out, skip4],
                            dim=1)  # up4_out:(1, 512, 64, 64) + skip4:(1, 512, 64, 64) -> merged4:(1, 1024, 64, 64)
        up_conv4_out = self.up_conv4(merged4)  # merged4:(1, 1024, 64, 64) -> up_conv4_out:(1, 512, 64, 64)

        # 解码阶段 3: 64x64 -> 128x128
        up3_out = self.up3(up_conv4_out)  # up_conv4_out:(1, 512, 64, 64) -> up3_out:(1, 256, 128, 128)
        merged3 = torch.cat([up3_out, skip3],
                            dim=1)  # up3_out:(1, 256, 128, 128) + skip3:(1, 256, 128, 128) -> merged3:(1, 512, 128, 128)
        up_conv3_out = self.up_conv3(merged3)  # merged3:(1, 512, 128, 128) -> up_conv3_out:(1, 256, 128, 128)

        # 解码阶段 2: 128x128 -> 256x256
        up2_out = self.up2(up_conv3_out)  # up_conv3_out:(1, 256, 128, 128) -> up2_out:(1, 128, 256, 256)
        merged2 = torch.cat([up2_out, skip2],
                            dim=1)  # up2_out:(1, 128, 256, 256) + skip2:(1, 128, 256, 256) -> merged2:(1, 256, 256, 256)
        up_conv2_out = self.up_conv2(merged2)  # merged2:(1, 256, 256, 256) -> up_conv2_out:(1, 128, 256, 256)

        # 解码阶段 1: 256x256 -> 512x512
        up1_out = self.up1(up_conv2_out)  # up_conv2_out:(1, 128, 256, 256) -> up1_out:(1, 64, 512, 512)
        merged1 = torch.cat([up1_out, skip1],
                            dim=1)  # up1_out:(1, 64, 512, 512) + skip1:(1, 64, 512, 512) -> merged1:(1, 128, 512, 512)
        up_conv1_out = self.up_conv1(merged1)  # merged1:(1, 128, 512, 512) -> up_conv1_out:(1, 64, 512, 512)

        # 使用1x1卷积将通道数调整为最终的类别数 (out_channels)，然后通过sigmoid激活函数得到每个像素的概率
        return torch.sigmoid(
            self.final_conv(up_conv1_out))  # up_conv1_out:(1, 64, 512, 512) -> final output:(1, out_channels, 512, 512)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetPlusPlus, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器（不变）
        self.down1 = DoubleConv(in_channels, 64)   # (64, 512, 512)
        self.down2 = DoubleConv(64, 128)          # (128, 256, 256)
        self.down3 = DoubleConv(128, 256)         # (256, 128, 128)
        self.down4 = DoubleConv(256, 512)         # (512, 64, 64)
        self.bottleneck = DoubleConv(512, 1024)   # (1024, 32, 32)

        # -------------------------- 修正嵌套块的输入通道数 --------------------------
        # 路径1：bottleneck → down4 → down3 → down2 → down1
        self.conv4_4 = DoubleConv(512 + 512, 512)  # up4(512) + x4(512) = 1024
        self.conv4_3 = DoubleConv(256 + 256, 256)  # up4_3(256) + x3(256) = 512
        self.conv4_2 = DoubleConv(128 + 128, 128)  # up4_2(128) + x2(128) = 256
        self.conv4_1 = DoubleConv(64 + 64, 64)     # up4_1(64) + x1(64) = 128

        # 路径2：down4 → down3 → down2 → down1
        self.conv3_3 = DoubleConv(256 + 256, 256)  # up3(256) + x3(256) = 512
        self.conv3_2 = DoubleConv(128 + 128, 128)  # up3_2(128) + x2(128) = 256
        self.conv3_1 = DoubleConv(64 + 64, 64)     # up3_1(64) + x1(64) = 128

        # 路径3：down3 → down2 → down1
        self.conv2_2 = DoubleConv(128 + 128, 128)  # up2(128) + x2(128) = 256
        self.conv2_1 = DoubleConv(64 + 64, 64)     # up2_1(64) + x1(64) = 128

        # 路径4：down2 → down1
        self.conv1_1 = DoubleConv(64 + 64, 64)     # up1(64) + x1(64) = 128

        # 上采样模块（不变）
        self.up4 = TransposeConv(1024, 512)   # 32x32 → 64x64
        self.up3 = TransposeConv(512, 256)    # 64x64 → 128x128
        self.up2 = TransposeConv(256, 128)    # 128x128 → 256x256
        self.up1 = TransposeConv(128, 64)     # 256x256 → 512x512

        # 辅助上采样（不变）
        self.up4_3 = TransposeConv(512, 256)  # conv4_4 → conv4_3
        self.up4_2 = TransposeConv(256, 128)  # conv4_3 → conv4_2
        self.up4_1 = TransposeConv(128, 64)   # conv4_2 → conv4_1
        self.up3_2 = TransposeConv(256, 128)  # conv3_3 → conv3_2
        self.up3_1 = TransposeConv(128, 64)   # conv3_2 → conv3_1
        self.up2_1 = TransposeConv(128, 64)   # conv2_2 → conv2_1

        # 最终输出层（不变）
        self.final_conv = nn.Conv2d(64 * 4, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    # forward方法不变（通道数已修正匹配）
    def forward(self, x):
        # 编码器特征提取
        x1 = self.down1(x)          # (64, 512, 512)
        x2 = self.down2(self.pool(x1))  # (128, 256, 256)
        x3 = self.down3(self.pool(x2))  # (256, 128, 128)
        x4 = self.down4(self.pool(x3))  # (512, 64, 64)
        x5 = self.bottleneck(self.pool(x4))  # (1024, 32, 32)

        # 路径1
        up4 = self.up4(x5)              # (512, 64, 64)
        merge4_4 = torch.cat([up4, x4], dim=1)  # 512+512=1024
        conv4_4 = self.conv4_4(merge4_4)        # 输入1024→输出512

        up4_3 = self.up4_3(conv4_4)     # (256, 128, 128)
        merge4_3 = torch.cat([up4_3, x3], dim=1)  # 256+256=512
        conv4_3 = self.conv4_3(merge4_3)        # 输入512→输出256

        up4_2 = self.up4_2(conv4_3)     # (128, 256, 256)
        merge4_2 = torch.cat([up4_2, x2], dim=1)  # 128+128=256
        conv4_2 = self.conv4_2(merge4_2)        # 输入256→输出128

        up4_1 = self.up4_1(conv4_2)     # (64, 512, 512)
        merge4_1 = torch.cat([up4_1, x1], dim=1)  # 64+64=128
        conv4_1 = self.conv4_1(merge4_1)        # 输入128→输出64

        # 路径2
        up3 = self.up3(x4)              # (256, 128, 128)
        merge3_3 = torch.cat([up3, x3], dim=1)  # 256+256=512
        conv3_3 = self.conv3_3(merge3_3)        # 输入512→输出256

        up3_2 = self.up3_2(conv3_3)     # (128, 256, 256)
        merge3_2 = torch.cat([up3_2, x2], dim=1)  # 128+128=256
        conv3_2 = self.conv3_2(merge3_2)        # 输入256→输出128

        up3_1 = self.up3_1(conv3_2)     # (64, 512, 512)
        merge3_1 = torch.cat([up3_1, x1], dim=1)  # 64+64=128
        conv3_1 = self.conv3_1(merge3_1)        # 输入128→输出64

        # 路径3
        up2 = self.up2(x3)              # (128, 256, 256)
        merge2_2 = torch.cat([up2, x2], dim=1)  # 128+128=256
        conv2_2 = self.conv2_2(merge2_2)        # 输入256→输出128

        up2_1 = self.up2_1(conv2_2)     # (64, 512, 512)
        merge2_1 = torch.cat([up2_1, x1], dim=1)  # 64+64=128
        conv2_1 = self.conv2_1(merge2_1)        # 输入128→输出64

        # 路径4
        up1 = self.up1(x2)              # (64, 512, 512)
        merge1_1 = torch.cat([up1, x1], dim=1)  # 64+64=128
        conv1_1 = self.conv1_1(merge1_1)        # 输入128→输出64

        # 融合输出
        final_merge = torch.cat([conv4_1, conv3_1, conv2_1, conv1_1], dim=1)  # 64*4=256
        out = self.final_conv(final_merge)
        return self.sigmoid(out)


class BasicBlock(nn.Module):
    """保持原结构不变"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.ReLU1 = nn.ReLU(inplace=True)

        self.Conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.ReLU2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.ReLU1(x)
        x = self.Conv2(x)
        x = self.BN2(x)
        x += self.shortcut(identity)
        x = self.ReLU2(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet34, self).__init__()
        self.in_channels = 64

        # 输入通道保持1（ISBI是单通道灰度图）
        self.Conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.BN1 = nn.BatchNorm2d(64)
        self.ReLU1 = nn.ReLU(inplace=True)

        # 主干网络结构完全不变
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=2)  # 512->256（下采样1次）
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)  # 256->128（下采样2次）
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)  # 128->64（下采样3次）
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)  # 64->32（下采样4次）

        # 解码器上采样参数不变（stride=2，每次上采样尺寸翻倍）
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 32->64（与x3的64匹配）
        self.conv_up1 = block(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 64->128（与x2的128匹配）
        self.conv_up2 = block(128 + 128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 128->256（与x1的256匹配）
        self.conv_up3 = block(64 + 64, 64)

        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 256->512（与x0的512匹配）
        self.conv_up4 = block(64 + 64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 编码器：提取多尺度特征（保留跳跃连接）
        x0 = self.ReLU1(self.BN1(self.Conv1(x)))  # x0: (B,64,512,512) （最浅层特征）
        x1 = self.layer1(x0)  # x1: (B,64,256,256)
        x2 = self.layer2(x1)  # x2: (B,128,128,128)
        x3 = self.layer3(x2)  # x3: (B,256,64,64)
        x4 = self.layer4(x3)  # x4: (B,512,32,32) （最深层特征）

        # 解码器：上采样+融合逻辑不变，仅卷积块换为残差块（改动2，对应conv_up*）
        up1 = self.up1(x4)  # (B,256,64,64)
        merge1 = torch.cat([up1, x3], dim=1)  # (B,512,64,64)
        dec1 = self.conv_up1(merge1)  # 残差块处理（B,256,64,64）

        up2 = self.up2(dec1)  # (B,128,128,128)
        merge2 = torch.cat([up2, x2], dim=1)  # (B,256,128,128)
        dec2 = self.conv_up2(merge2)  # 残差块处理（B,128,128,128）

        up3 = self.up3(dec2)  # (B,64,256,256)
        merge3 = torch.cat([up3, x1], dim=1)  # (B,128,256,256)
        dec3 = self.conv_up3(merge3)  # 残差块处理（B,64,256,256）

        up4 = self.up4(dec3)  # (B,64,512,512)
        merge4 = torch.cat([up4, x0], dim=1)  # (B,128,512,512)
        dec4 = self.conv_up4(merge4)  # 残差块处理（B,64,512,512）

        out = self.final_conv(dec4)  # (B,1,512,512)
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=1).to(device)
    summary(model, input_size=(4, 1, 512, 512))
    model = ResNet34(BasicBlock, [3, 4, 6, 3], num_classes=1).to(device)
    summary(model, input_size=(4, 1, 512, 512))
    model = UNetPlusPlus(in_channels=1, out_channels=1).to(device)
    summary(model, input_size=(4, 1, 512, 512))
