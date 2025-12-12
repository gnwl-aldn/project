import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=None):
        """
        Args:
            smooth: 数値安定性のためのsmoothing term
            ignore_index: 無視するクラスのインデックス（背景クラスなど）
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - ネットワークの出力（softmax前）
            targets: (B, H, W) - 正解ラベル（整数）
        Returns:
            dice_loss: スカラー
        """
        # Softmaxで確率に変換
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        
        # targetsをone-hot encodingに変換
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # クラスごとにDice係数を計算
        dims = (0, 2, 3)  # バッチ、高さ、幅で集約
        
        intersection = torch.sum(probs * targets_one_hot, dim=dims)  # (C,)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)  # (C,)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # ignore_indexがある場合は除外
        if self.ignore_index is not None:
            mask = torch.ones(num_classes, device=dice_score.device)
            mask[self.ignore_index] = 0
            dice_score = dice_score * mask
            dice_loss = 1.0 - dice_score.sum() / mask.sum()
        else:
            dice_loss = 1.0 - dice_score.mean()
        
        return dice_loss


# 使用例
if __name__ == "__main__":
    # ダミーデータ
    batch_size, num_classes, height, width = 4, 10, 256, 256
    logits = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # 損失計算
    dice_loss = DiceLoss(smooth=1.0)
    loss = dice_loss(logits, targets)
    print(f"Dice Loss: {loss.item()}")
    
    # CEとの組み合わせ
    ce_loss = F.cross_entropy(logits, targets)
    combined_loss = ce_loss + loss
    print(f"CE Loss: {ce_loss.item()}")
    print(f"Combined Loss: {combined_loss.item()}")