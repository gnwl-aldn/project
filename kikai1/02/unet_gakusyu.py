import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import wandb

# GPU・最適化アルゴリズムの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = UNet_2D().to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# wandb の初期化
wandb.init(
    project="unet_training",
    name="unet_run1",
    config={
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 1e-2,
    },)


# wandb の初期化
wandb.init(
    project="unet_training",
    name="unet_run1"
    config={
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 1e-2,
    },)

config = wandb.config

n = 0
m = 0


# 学習ループ
for epoch in range(config.epochs):
    train_loss = 0
    val_loss = 0

    unet.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data["img"].to(device), data["label"].to(device)
        optimizer.zero_grad()
        outputs = unet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n += 1

        wandb.log({"train_loss_batch": loss.item()})

        if i % ((len(df)//BATCH_SIZE)//10) == (len(df)//BATCH_SIZE)//10 - 1:
            print(f"{epoch+1} index:{i+1} train_loss:{train_loss/n:.5f}")

            wandb.log({"train_loss": avg_loss, "epoch": epoch+1})

            n = 0
            train_loss = 0
            train_acc = 0

    
    unet.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data["img"].to(device), data["label"].to(device)
            outputs = unet(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            m += 1

            wandb.log({"val_loss_batch": loss.item()})

            if i % (len(val_df)//BATCH_SIZE) == len(val_df)//BATCH_SIZE - 1:
                print(f"epoch:{epoch+1} index:{i+1} val_loss:{val_loss/m:.5f}")

                wandb.log({"val_loss": avg_val_loss, "epoch": epoch+1})

                m = 0
                val_loss = 0
                val_acc = 0


    # モデル保存
    torch.save(unet.state_dict(), f"./train_{epoch+1}.pth")

print("Finished Training")
wandb.finish()