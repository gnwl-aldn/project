from PIL import Image

class ClothingItem:
    def __init__(self, name, item_type, category, image_path):
        self.name = name
        self.item_type = item_type 
        self.category = category
        self.image_path = image_path
        
        # 色彩分析の結果を保存する場所を初期化
        self.hsv = None
        self.rgb = None

    def __repr__(self):
        return f"ClothingItem(name='{self.name}', type='{self.item_type}', category='{self.category}')"
    
    def open_image(self):
        return Image.open(self.image_path)

def initialize_closet():
    closet_data = [
        ClothingItem("黒のダウンジャケット", "ダウンジャケット", "Outerwear", "closet_images/kuro_down.jpg"),
        ClothingItem("グレーのセーター", "セーター", "Tops", "closet_images/grey_sweater.jpg"),
        ClothingItem("ベージュのコート", "コート", "Outerwear", "closet_images/beige_coat.jpg"),
        ClothingItem("お気に入りのパーカー", "パーカー", "Tops", "closet_images/fav_parka.jpg"),
        ClothingItem("白のTシャツ", "Tシャツ", "Tops", "closet_images/white_tshirt.jpg"),
        ClothingItem("ブルージーンズ", "ジーンズ", "Bottoms", "closet_images/blue_jeans.jpg"),
        ClothingItem("黒のスキニーパンツ", "ロングパンツ", "Bottoms", "closet_images/black_skinny.jpg"),
        ClothingItem("白のスニーカー", "スニーカー", "Other", "closet_images/white_sneaker.jpg"),
        ClothingItem("赤のマフラー", "マフラー", "Accessory", "closet_images/red_muffler.jpg"),
        ClothingItem("デニムのショートパンツ", "ショートパンツ", "Bottoms", "closet_images/denim_shorts.jpg"),
        ClothingItem("麦わら帽子", "麦わら帽子", "Accessory", "closet_images/straw_hat.jpg"),
        ClothingItem("Acneのマフラー", "マフラー", "Accessory", "closet_images/1.jpg"),
        ClothingItem("青いセーター", "セーター", "Tops", "closet_images/2.jpeg")
    ]
    print(f"\n--- [Class] {len(closet_data)} 点のアイテムをクラスリストから初期化しました ---")
    return closet_data