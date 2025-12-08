import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap

def visualize_segmentation(image, mask, alpha=0.5, num_classes=None):
    """
    画像にセグメンテーションマスクをオーバーレイして可視化
    
    Args:
        image: PIL Image
        mask: np.array (H, W) クラスIDの2次元配列
        alpha: オーバーレイの透明度 (0-1)
        num_classes: クラス数（Noneの場合は自動検出）
    """
    # 画像をnumpy配列に変換
    img_array = np.array(image)
    
    # クラス数を取得
    if num_classes is None:
        num_classes = mask.max() + 1
    
    # カラーマップを生成（背景は透明、他のクラスは色付き）
    colors = plt.cm.get_cmap('tab20', num_classes)
    cmap = ListedColormap(colors(range(num_classes)))
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 元画像
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # マスクのみ
    im = axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=num_classes-1)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # オーバーレイ
    axes[2].imshow(img_array)
    axes[2].imshow(mask, cmap=cmap, alpha=alpha, vmin=0, vmax=num_classes-1)
    axes[2].set_title(f'Overlay (alpha={alpha})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def quick_check(image, mask, class_names=None):
    """
    クイックチェック用の簡易版
    
    Args:
        image: PIL Image
        mask: np.array (H, W)
        class_names: クラス名のリスト（オプション）
    """
    print(f"Image size: {image.size}")
    print(f"Mask shape: {mask.shape}")
    print(f"Unique classes: {np.unique(mask)}")
    print(f"Class distribution:")
    
    unique, counts = np.unique(mask, return_counts=True)
    for cls, count in zip(unique, counts):
        percentage = count / mask.size * 100
        name = class_names[cls] if class_names and cls < len(class_names) else f"Class {cls}"
        print(f"  {name}: {count} pixels ({percentage:.2f}%)")
    
    visualize_segmentation(image, mask)

# 使用例
# image = Image.open("path/to/image.jpg")
# mask = np.load("path/to/mask.npy")
# visualize_segmentation(image, mask, alpha=0.5)
# または
# quick_check(image, mask, class_names=["background", "person", "car", ...])