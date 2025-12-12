import torch
import torch.nn as nn
import torchvision.models as models


class ResidualConv(nn.Module):
    """Residual Convolution Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualConv, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.conv_skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.conv_skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        self.relu = nn.SiLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv_block(x) + self.conv_skip(x))


class Upsample(nn.Module):
    """Upsample Block with Residual Connection"""
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x):
        return self.upsample(x)


class ResUNet(nn.Module):
    """ResU-Net with ResNet-50 pretrained encoder"""
    def __init__(self, num_classes=1, pretrained=True):
        super(ResUNet, self).__init__()
        
        # ResNet-50エンコーダをロード（事前学習済み）
        resnet = models.resnet50(pretrained=pretrained)
        
        # エンコーダ部分（ResNet-50の各層を使用）
        self.encoder1 = nn.Sequential(
            resnet.conv1,      # 7x7 conv, 64 channels
            resnet.bn1,
            resnet.relu
        )
        self.pool1 = resnet.maxpool  # 3x3 maxpool, stride=2
        
        self.encoder2 = resnet.layer1  # 256 channels
        self.encoder3 = resnet.layer2  # 512 channels
        self.encoder4 = resnet.layer3  # 1024 channels
        self.encoder5 = resnet.layer4  # 2048 channels
        
        # ブリッジ（ボトルネック）
        self.bridge = ResidualConv(2048, 2048)
        
        # デコーダ部分（アップサンプリング + スキップ接続）
        self.upsample5 = Upsample(2048, 1024)
        self.decoder5 = ResidualConv(1024 + 1024, 1024)  # スキップ接続: encoder4
        
        self.upsample4 = Upsample(1024, 512)
        self.decoder4 = ResidualConv(512 + 512, 512)  # スキップ接続: encoder3
        
        self.upsample3 = Upsample(512, 256)
        self.decoder3 = ResidualConv(256 + 256, 256)  # スキップ接続: encoder2
        
        self.upsample2 = Upsample(256, 64)
        self.decoder2 = ResidualConv(64 + 64, 64)  # スキップ接続: encoder1
        
        self.upsample1 = Upsample(64, 32)
        self.decoder1 = ResidualConv(32, 32)
        
        # 最終出力層
        self.output = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        # エンコーダパス（ダウンサンプリング + 特徴抽出）
        enc1 = self.encoder1(x)      # 64 channels, H/2 x W/2
        enc1_pooled = self.pool1(enc1)
        
        enc2 = self.encoder2(enc1_pooled)  # 256 channels, H/4 x W/4
        enc3 = self.encoder3(enc2)         # 512 channels, H/8 x W/8
        enc4 = self.encoder4(enc3)         # 1024 channels, H/16 x W/16
        enc5 = self.encoder5(enc4)         # 2048 channels, H/32 x W/32
        
        # ブリッジ
        bridge = self.bridge(enc5)
        
        # デコーダパス（アップサンプリング + スキップ接続）
        dec5 = self.upsample5(bridge)                      # 1024 channels, H/16 x W/16
        dec5 = torch.cat([dec5, enc4], dim=1)             # スキップ接続
        dec5 = self.decoder5(dec5)                        # 1024 channels
        
        dec4 = self.upsample4(dec5)                       # 512 channels, H/8 x W/8
        dec4 = torch.cat([dec4, enc3], dim=1)             # スキップ接続
        dec4 = self.decoder4(dec4)                        # 512 channels
        
        dec3 = self.upsample3(dec4)                       # 256 channels, H/4 x W/4
        dec3 = torch.cat([dec3, enc2], dim=1)             # スキップ接続
        dec3 = self.decoder3(dec3)                        # 256 channels
        
        dec2 = self.upsample2(dec3)                       # 64 channels, H/2 x W/2
        dec2 = torch.cat([dec2, enc1], dim=1)             # スキップ接続
        dec2 = self.decoder2(dec2)                        # 64 channels
        
        dec1 = self.upsample1(dec2)                       # 32 channels, H x W
        dec1 = self.decoder1(dec1)
        
        # 最終出力
        out = self.output(dec1)
        
        return out


# 使用例
if __name__ == "__main__":
    # モデルの作成（バイナリセグメンテーションの場合）
    model = ResUNet(num_classes=1, pretrained=True)
    
    # マルチクラスセグメンテーションの場合
    # model = ResUNet(num_classes=21, pretrained=True)  # 例: PASCAL VOC
    
    # モデルのサマリー
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # テスト用の入力
    x = torch.randn(2, 3, 256, 256)  # バッチサイズ2, RGB, 256x256
    
    # フォワードパス
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # (2, 1, 256, 256)