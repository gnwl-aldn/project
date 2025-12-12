import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import fashionpedia as fp
from resunet import ResUNet
import wandb
from torchmetrics.functional import jaccard_index
from torchmetrics import JaccardIndex
import os
from dice_loss import DiceLoss

# デバイス設定
# GPUがあればGPUで学習、なければCPUで実行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_cpus = os.cpu_count()
print(f"Number of CPU cores available: {num_cpus}")

use_class_ids = list(range(0,26))
print(f"Using class IDs: {use_class_ids}")
classes = len(use_class_ids) + 1  # 背景クラスを追加
print(f"Number of classes: {classes}")

iou = JaccardIndex(task ="multiclass", num_classes=classes).to(device)

# wandb の初期化

wandb.login()

wandb.init(
    project="FashionSegmentation",
    name="test1",
    tags=["ResUNet",
          "Fashionpedia",
          "normalization",
          "decoder with SiLU"
          "ce + dice loss"
          ],
    config={
        "epochs": 10,
        "batch_size": 64*4,
        "learning_rate": 1e-4,
    },
)

config = wandb.config

# データセット・データローダーの準備
IMG_SIZE = 224
base_path = "/root/.cache/kagglehub/competitions/imaterialist-fashion-2020-fgvc7"
TRAIN_CSV = base_path + "/train.csv"
IMG_DIR = base_path + "/train"

train_dataset = fp.Dataset(IMG_SIZE, TRAIN_CSV, IMG_DIR, use_class_ids)

train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=16,
                          pin_memory=True
                          )

print("dataset and dataloader ready")

# モデル・最適化・損失関数の設定
unet = ResUNet(classes, pretrained = True).to(torch.bfloat16).to(device)
optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()
dice = DiceLoss()


# 学習ループ
print("start training")
unet.train()
for epoch in range(config.epochs):
    unet.train()
    for data in train_loader:
        inputs, labels = data
        inputs = inputs.to(torch.bfloat16)
        inputs, labels = inputs.to(device), labels.to(device)
        print("data loaded")
        optimizer.zero_grad()
        outputs = unet(inputs)
        loss = criterion(outputs, labels) + dice(outputs, labels)
        loss.backward()
        optimizer.step()
        print("batch trained")

        # バッチごとの学習損失をwandbに記録
        wandb.log({"train_loss": loss.item(),
                    "iou": iou(torch.argmax(outputs, dim=1), labels).item()})

    # モデル保存
    torch.save(unet.state_dict(), f"./train_{epoch+1}.pth")

print("Finished Training")
wandb.finish()