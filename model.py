import torch
import torch.nn as nn
import torch.nn.functional as F

# 畳み込みブロック (Conv -> BN -> ReLU -> Conv -> BN -> ReLU)
class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1) # padding=1でサイズ維持
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

# アップサンプリングブロック (拡大して結合) 
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) 
        # ここでチャンネル数を調整するための畳み込み

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

# 本体: U-Netモデル
class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_channels = 3
        self.n_classes = n_classes

        # Encoder (下り)
        self.TCB1 = TwoConvBlock(self.n_channels, 64, 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)

        # Pooling
        self.maxpool = nn.MaxPool2d(2, stride=2)
        
        # Decoder (上り)
        self.UC1 = UpConv(1024, 512) 
        self.TCB6 = TwoConvBlock(1024, 512, 512)

        self.UC2 = UpConv(512, 256) 
        self.TCB7 = TwoConvBlock(512, 256, 256)

        self.UC3 = UpConv(256, 128) 
        self.TCB8 = TwoConvBlock(256, 128, 128)

        self.UC4= UpConv(128, 64)
        self.TCB9 = TwoConvBlock(128, 64, 64)

        # 出力層
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.TCB1(x)
        x_pool1 = self.maxpool(x1)

        x2 = self.TCB2(x_pool1)
        x_pool2 = self.maxpool(x2)

        x3 = self.TCB3(x_pool2)
        x_pool3 = self.maxpool(x3)

        x4 = self.TCB4(x_pool3)
        x_pool4 = self.maxpool(x4)

        x5 = self.TCB5(x_pool4)

        # Decoder (Skip Connectionあり)
        x = self.UC1(x5)
        # ここでサイズが合うように結合 (x4とxを結合)
        x = torch.cat([x4, x], dim=1) 
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.TCB9(x)

        # 最終出力
        logits = self.out_conv(x)
        return logits