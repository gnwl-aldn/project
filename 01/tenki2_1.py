import json

def load_json_data(file_path):
    """
    指定されたJSONファイルを読み込むヘルパー関数
    - file_path: 読み込むJSONファイルのパス
    - 戻り値: 読み込んだデータ (失敗した場合は None)
    """
    print(f"\n--- データをロード中: {file_path} ---")
    try:
        # 'utf-8' を指定してファイルを開く
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("  -> 読み込み成功")
            return data
    except FileNotFoundError:
        print(f"  [エラー] {file_path} が見つかりません。")
        return None
    except json.JSONDecodeError:
        print(f"  [エラー] {file_path} のJSON形式が正しくありません。")
        return None
    except Exception as e:
        print(f"  [エラー] {file_path} の読み込み中に予期せぬ問題が発生しました: {e}")
        return None

# 関数部分 

def get_allowed_clothing_types(temperature, rules):
    print(f"\n--- 関数: 気温{temperature}度に合う服の「種類」を検索 ---")
    
    allowed_types = []
    
    for item_type, rule in rules.items():
        min_temp = rule.get('min')
        max_temp = rule.get('max')
        
        is_min_ok = False
        if min_temp is None:
            is_min_ok = True 
        elif temperature >= min_temp:
            is_min_ok = True
            
        is_max_ok = False
        if max_temp is None:
            is_max_ok = True  
        elif temperature <= max_temp:
            is_max_ok = True
            
        if is_min_ok and is_max_ok:
            allowed_types.append(item_type)
            
    print(f"  -> 許可された服のタイプ (全{len(allowed_types)}種)")
    return allowed_types

def filter_my_closet(allowed_types, closet):
    print("\n--- 関数: 自分の服と「許可された種類」を照合 ---")
    
    allowed_types_set = set(allowed_types)
    
    wearable_clothes = []
    for item in closet:
        if item['type'] in allowed_types_set:
            wearable_clothes.append(item)
            print(f"  [OK] {item['name']} (タイプ: {item['type']})")
        else:
            print(f"  [NG] {item['name']} (タイプ: {item['type']}) - 気温に合いません")

    return wearable_clothes

# --- メインの実行部分 ---
if __name__ == "__main__":
    
    ALL_ITEMS_RULES = load_json_data("rules.json")
    my_closet = load_json_data("closet.json")

    if ALL_ITEMS_RULES is None or my_closet is None:
        print("\nデータファイルの読み込みに失敗したため、処理を終了します。")
    else:
        current_temperature = 8 
        print(f"\n現在の気温: {current_temperature}度")
        
        allowed_types_1 = get_allowed_clothing_types(current_temperature, ALL_ITEMS_RULES)
        wearable_clothes_1 = filter_my_closet(allowed_types_1, my_closet)
        
        print("\n--- 最終結果 ---")
        print(f"気温{current_temperature}度で着られるあなたの服:")
        for item in wearable_clothes_1:
            print(f" - {item['name']} ({item['type']})")
            
        current_temperature = 28 
        print(f"*** 現在の気温: {current_temperature}度 ***")
        
        allowed_types_2 = get_allowed_clothing_types(current_temperature, ALL_ITEMS_RULES)
        wearable_clothes_2 = filter_my_closet(allowed_types_2, my_closet)
        
        print("\n--- 最終結果 ---")
        print(f"気温{current_temperature}度で着られるあなたの服:")
        for item in wearable_clothes_2:
            print(f" - {item['name']} ({item['type']})")