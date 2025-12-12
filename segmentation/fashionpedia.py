import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List

class Resizer:
    def __init__(self, size):
        self.size = size
        self.img_resize = transforms.Resize(size)
        # マスクは最近隣補間 (NEAREST) を使用
        # カテゴリIDが補間によって変わってしまうのを防ぐため
        self.mask_resize = transforms.Resize(size, interpolation=Image.NEAREST)

    def __call__(self, image, mask):
        # 1. 画像のリサイズ
        image_resized = self.img_resize(image)
        
        # 2. マスクのリサイズ
        # NumPy配列を一度PIL Imageに戻す
        mask_pil = Image.fromarray(mask.astype(np.uint8)) 
        mask_resized_pil = self.mask_resize(mask_pil)
        
        # 3. 再び NumPy配列に戻す
        mask_resized_np = np.array(mask_resized_pil).astype(np.int64)
        
        return image_resized, mask_resized_np

def rle_decode(rle, H, W, fill_value=1):
    """
    RLE (Run-Length Encoding) をデコードしてバイナリマスク (NumPy配列) を生成します。

    Args:
        rle (str): RLE文字列 (例: '1 3 10 5 ...')
        H (int): マスクの高さ
        W (int): マスクの幅
        fill_value (int): RLEによって指定される領域に割り当てる値 (通常は1)

    Returns:
        np.ndarray: デコードされたマスク (H x W)
    """
    if pd.isna(rle):
        return np.zeros((H, W), dtype=np.uint8)

    # スペースで分割し、整数に変換
    parts = np.array(rle.split()).astype(int)
    
    # RLEは [開始位置, 長さ, 開始位置, 長さ, ...] の形式
    starts = parts[::2] - 1  # 1-indexedなので0-indexedに変換
    lengths = parts[1::2]
    
    mask = np.zeros(H * W, dtype=np.uint8)
    
    # RLEセグメントを塗りつぶし
    for start, length in zip(starts, lengths):
        mask[start:start + length] = fill_value
    
    # H x W の形状に戻し、転置して (H, W) 形式にする (Fashion-Pediaの形式に合わせる)
    # RLEは通常、列方向に走査される <- これ正しい？
    return mask.reshape((W,H)).T

def create_segmentation_mask(df_rows, img_path, use_ids:List[int]=[]):
    """
    画像に存在する全てのアノテーションを統合し、カテゴリIDのマスクを作成します。

    Args:
        df_rows (pd.DataFrame): 特定のImageIdに対応するアノテーション行
        img_path (str): 画像ファイルのパス (サイズ取得用)

    Returns:
        np.ndarray: 統合されたカテゴリIDマスク (H x W, dtype=np.int64)
    """
    # 画像サイズを取得
    img = Image.open(img_path)
    W, H = img.size
    
    # 全てゼロ（背景）で初期化されたマスク。U-Netのターゲットは通常、
    # [H, W] の形状で、各ピクセルがカテゴリIDを持ちます。
    # ここでは、背景をクラスID n+1 とします。
    background_id = len(use_ids)
    mask = np.full((H, W), background_id, dtype=np.int64)
    
    # 各アノテーション行を反復処理
    # (注意: DataFrameの行は元のCSVの順番で処理されます)
    for _, row in df_rows.iterrows():
        rle = row['EncodedPixels']
        class_id = int(row['ClassId'])

        if not (class_id in use_ids):
            continue
        
        # RLEをデコードしてバイナリマスクを取得
        # fill_value に class_id を渡すことで、デコード時にそのカテゴリIDで領域を塗りつぶします
        binary_mask = rle_decode(rle, H, W, fill_value=class_id)
        
        # 既存のマスクに上書きします。これにより、後に処理されたRLEが優先されます。
        # この処理は、オーバーラップ処理の最もシンプルな方法です。
        # (binary_mask != 0) は、その領域が現在のカテゴリでアノテーションされていることを示します。
        mask = np.where(binary_mask != 0, binary_mask, mask)
        
    return mask

class Dataset(Dataset):
    def __init__(self, img_size:int, csv_file, img_dir, use_ids:List[int]=[]):
        """
        Args:
            img_size (int): 画像とマスクのリサイズ後のサイズ (例: 256)
            csv_file (str): アノテーションCSVファイルへのパス ('train.csv')
            img_dir (str): 画像ディレクトリへのパス ('train/')
        """
        self.data_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.use_ids = use_ids
        self.resizer = Resizer((img_size, img_size))
        
        # ImageIdごとにアノテーションをグループ化し、一意の画像IDのリストを作成
        # これがデータセットのインデックスに対応します
        self.image_ids = self.data_df['ImageId'].unique().tolist()
        
        # 画像IDごとのグループ化されたデータをキャッシュ
        self.grouped = self.data_df.groupby('ImageId')
        
        self.to_tensor = transforms.ToTensor()
        self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1. 画像IDの取得
        image_id = self.image_ids[idx]
        img_file = f"{image_id}.jpg"
        img_path = os.path.join(self.img_dir, img_file)
        
        # 2. 画像の読み込み
        image = Image.open(img_path).convert("RGB")
        
        # 3. マスクの生成
        # そのImageIdに対応する全てのアノテーション行を取得
        df_rows = self.grouped.get_group(image_id)
        # RLEデコードと統合により、セグメンテーションマスク (H x W, np.int64) を作成
        mask_np = create_segmentation_mask(df_rows, img_path, use_ids=self.use_ids)  # np.int64
        
        # 4. 画像への変換の適用
        if self.resizer:
            image, mask_np = self.resizer(image, mask_np)
        
        # 画像のテンソル化
        image_tensor = self.to_tensor(image)  # [C, H, W], float32
        image_tensor = self.normalization(image_tensor)

        # 5. マスクのテンソル化
        # マスクは [H, W] の形状で、型は LongTensor (カテゴリID用) にします。
        # U-Netの出力は [B, C, H, W]、ターゲットは [B, H, W] または [B, 1, H, W] が一般的です。
        mask_tensor = torch.from_numpy(mask_np).long() # [H, W]
        
        # (補足) U-Net学習のための推奨される出力形式:
        # 画像: Tensor[C, H, W] (float32, [0, 1]または正規化済み)
        # マスク: Tensor[H, W] (long/int64, カテゴリID)
        return image_tensor, mask_tensor