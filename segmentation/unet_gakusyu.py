import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import fashionpedia
from model import UNet
import wandb
import os


# デバイス設定
# GPUがあればGPUで学習、なければCPUで実行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# wandb の初期化
wandb.init(
    project="unet_training",
    name="unet_run1",
    config={
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 1e-2,
    },
)

config = wandb.config
BATCH_SIZE = config.batch_size


# データセット・データローダーの準備
IMG_SIZE = (256, 256)
TRAIN_CSV = "./train.csv"
VAL_CSV = "./val.csv"
IMG_DIR = "./train/"

train_dataset = fashionpedia.Dataset(IMG_SIZE, TRAIN_CSV, IMG_DIR)
val_dataset = fashionpedia.Dataset(IMG_SIZE, VAL_CSV, IMG_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# モデル・最適化・損失関数の設定
unet = UNet().to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()


# 学習ループ
for epoch in range(config.epochs):
    train_loss = 0
    val_loss = 0
    # n = 0   # 学習バッチ数
    # m = 0   # 検証バッチ数

    unet.train()
    for i, data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = unet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()
        # n += 1
        # バッチごとの学習損失をwandbに記録
        wandb.log({"train_loss_batch": loss.item()})

    # # 学習損失の平均を計算してログに記録
    # avg_train_loss = train_loss / n
    # print(f"epoch:{epoch+1} train_loss:{avg_train_loss:.5f}")
    # wandb.log({"train_loss": avg_train_loss, "epoch": epoch+1})
    
    

    # 検証ループ
    unet.eval()
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = unet(inputs)
            loss = criterion(outputs, labels)

            # val_loss += loss.item()
            # m += 1

            # バッチごとの検証損失をwandbに記録
            wandb.log({"val_loss_batch": loss.item()})

    # # 検証損失の平均を計算してログに記録
    # avg_val_loss = val_loss / m
    # print(f"epoch:{epoch+1} val_loss:{avg_val_loss:.5f}")


    # モデル保存
    torch.save(unet.state_dict(), f"./train_{epoch+1}.pth")

print("Finished Training")
wandb.finish()