import cv2
import numpy as np
import json
import sys
from collections import defaultdict
import os
from typing import List, Dict, Optional, Any, Tuple
from closet import ClothingItem, initialize_closet

# 色彩分析関数群
def average(values, weights):
    """ 通常の加重平均 """
    return np.sum(values * weights) / np.sum(weights)

def circular_average(h_values, weights):
    """ 色相 (0-179) のための加重平均 """
    h_deg = h_values * 2
    h_rad = np.deg2rad(h_deg)
    x_mean = np.sum(np.cos(h_rad) * weights)
    y_mean = np.sum(np.sin(h_rad) * weights)
    angle = np.rad2deg(np.arctan2(y_mean, x_mean)) % 360
    return angle / 2

def hue_distance(h1, h2):
    """ 色相の距離を計算 (0-360°スケール) """
    d = abs(h1 - h2)
    return min(d, 360 - d)

def tone_score(s1, v1, s2, v2):
    """ トーン（彩度・明度）の類似度を計算 (0-1スケール) """
    s_diff = abs(s1 - s2)
    v_diff = abs(v1 - v2)
    score = 1 - (s_diff + v_diff) / 2.0
    return max(0, score)

def color_score(hsv1, hsv2, w_h=0.7, w_t=0.3):
    """ 総合的な色相性スコアを計算 (0-1スケール) """
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2
    hue_sim = 1 - (hue_distance(h1, h2) / 180.0)
    hue_sim = max(0, hue_sim)
    tone_sim = tone_score(s1, v1, s2, v2)
    total_score = w_h * hue_sim + w_t * tone_sim
    return round(max(0, min(1, total_score)), 3)

def extract_representative_color(image_path):
    """ 画像から中央重点の加重平均で代表色を抽出する """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = cv2.split(hsv_image)
    h, w, _ = hsv_image.shape
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    sigma_x, sigma_y = w / 4, h / 4
    weight = np.exp(-(((x - cx) ** 2) / (2 * sigma_x ** 2) + ((y - cy) ** 2) / (2 * sigma_y ** 2)))
    weight_flat = weight.flatten()

    H_mean = circular_average(H.flatten(), weight_flat)
    S_mean = average(S.flatten(), weight_flat)
    V_mean = average(V.flatten(), weight_flat)

    hsv_normalized = (H_mean * 2, S_mean / 255.0, V_mean / 255.0)
    hsv_opencv = np.uint8([[[H_mean, S_mean, V_mean]]])
    rep_color_bgr = cv2.cvtColor(hsv_opencv, cv2.COLOR_HSV2BGR)[0][0]
    rgb_color = rep_color_bgr[::-1].tolist()

    return {'hsv': hsv_normalized, 'rgb': rgb_color}

# フィルタリング関数群
def load_rules_data(file_path):
    print(f"\n--- データをロード中: {file_path} ---")
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print("  -> ファイルが空か見つかりません。")
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("  -> 読み込み成功")
            return data
    except FileNotFoundError:
        print(f"  [エラー] {file_path} が見つかりません。")
    except json.JSONDecodeError:
        print(f"  [エラー] {file_path} のJSON形式が正しくありません。")
    except Exception as e:
        print(f"  [エラー] 読み込み中に予期せぬ問題が発生: {e}")
    return None

def get_allowed_clothing_types(temperature, rules):
    print(f"\n--- 関数: 体感温度 {temperature}度 に合う服の「種類」を検索 ---")
    allowed_types = []
    for item_type, rule in rules.items():
        min_temp = rule.get('min')
        max_temp = rule.get('max')
        is_min_ok = (min_temp is None) or (temperature >= min_temp)
        is_max_ok = (max_temp is None) or (temperature <= max_temp)
        if is_min_ok and is_max_ok:
            allowed_types.append(item_type)
    print(f"  -> 許可された服のタイプ (全{len(allowed_types)}種)")
    return allowed_types

def filter_closet_objects(allowed_types, closet_list: List[ClothingItem]) -> List[ClothingItem]:
    print("\n--- 関数: 自分の服と「許可された種類」を照合 ---")
    allowed_types_set = set(allowed_types)
    wearable_clothes = []
    
    for item in closet_list: 
        if item.item_type in allowed_types_set:
            wearable_clothes.append(item)
            print(f"  [OK] {item.name} (タイプ: {item.item_type})")
        else:
            print(f"  [NG] {item.name} (タイプ: {item.item_type}) - 体感温度に合いません")
            
    return wearable_clothes

#可視化・出力関数群 (ヘルパー関数)
def print_color_info(label: str, item: ClothingItem):
    """ Item オブジェクトを引数に取る """
    h, s, v = item.hsv
    rgb = item.rgb
    print(f"----- {label}: {item.name} -----")
    print(f"  HSV: (H={h:.1f}°, S={s:.2f}, V={v:.2f})")
    print(f"  RGB: {rgb}")

def show_final_outfit(items: List[ClothingItem]):
    """ Item オブジェクトのリストを引数に取る """
    color_blocks = []
    labels = []
    
    for item in items:
        if item:
            rgb_color = item.rgb
            block = np.full((150, 150, 3), np.array(rgb_color, dtype=np.uint8), dtype=np.uint8)
            color_blocks.append(block)
            labels.append(item.category)

    if not color_blocks:
        print("表示する色がありません。")
        return

    combined_image = np.hstack(color_blocks)
    window_title_short = ' & '.join(labels)
    if len(window_title_short) > 100:
        window_title_short = f"{len(labels)} items"
        
    cv2.imshow(f'Final Outfit: {window_title_short}', combined_image)
    print("\n代表色のウィンドウが表示されました。キーを押すと終了します。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#メインの提案関数 (ご指定の関数)
def suggest(items: List[ClothingItem], weather: str, temp: int):
    RULES_FILE = "rules.json" 
    # 天気による体感温度の調整
    weather_adjustment = {"晴れ": 2, "曇り": 0, "雨": -2, "雪": -5}
    adjustment = weather_adjustment.get(weather, 0)
    ADJUSTED_TEMPERATURE = temp + adjustment

    print(f"現在の設定気温: {temp}度")
    print(f"天気: {weather} (体感調整: {adjustment:+}度)")
    print(f" -> 体感温度: {ADJUSTED_TEMPERATURE}度 で服を検索します")
    
    ALL_ITEMS_RULES = load_rules_data(RULES_FILE)
    
    # (クローゼットのリスト 'items' は引数で渡されている)
    if ALL_ITEMS_RULES is None or not items:
        print("\n[エラー] rules.json またはクローゼットにアイテムがありません。")
        return 
    allowed_types = get_allowed_clothing_types(ADJUSTED_TEMPERATURE, ALL_ITEMS_RULES)
    
    wearable_clothes = filter_closet_objects(allowed_types, items)

    if not wearable_clothes:
        print(f"\n[結果] 体感温度{ADJUSTED_TEMPERATURE}度で着られる服はありません。")
        return

    print("\n--- 3. 着用可能な服の代表色を抽出中... ---")
    categorized_closet = defaultdict(list)
    
    for item in wearable_clothes:
        try:
            color_data = extract_representative_color(item.image_path)
            item.hsv = color_data['hsv']
            item.rgb = color_data['rgb']
            categorized_closet[item.category].append(item)
            print(f"  [成功] {item.name} ({item.category}) の色を抽出完了。")
            
        except FileNotFoundError:
            print(f"  [エラー] {item.name} の画像 ({item.image_path}) が見つかりません。")
        except Exception as e:
            print(f"  [エラー] {item.name} の処理中に問題発生: {e}")

    wearable_tops = categorized_closet.get('Tops', [])
    wearable_bottoms = categorized_closet.get('Bottoms', [])
    wearable_outers = categorized_closet.get('Outerwear', [])
    wearable_others = categorized_closet.get('Other', [])
    wearable_accessories = categorized_closet.get('Accessory', [])

    if not wearable_tops or not wearable_bottoms:
        print("\n[エラー] 着用可能なトップス、またはボトムスが見つかりませんでした。")
        return
    else:
        print(f"\n--- 4. 最適なトップスとボトムスのペアを検索 ---")
        print(f" (トップス {len(wearable_tops)}着, ボトムス {len(wearable_bottoms)}着 で総当たり)")

    best_pair = None
    best_pair_score = -1.0 

    for top in wearable_tops:
        for bottom in wearable_bottoms:
            score = color_score(top.hsv, bottom.hsv)
            if score > best_pair_score:
                best_pair_score = score
                best_pair = {'top': top, 'bottom': bottom}

    print(f"\n--- 5. 検索完了: ベストペア (スコア: {best_pair_score:.3f}) ---")
    print_color_info("Tops", best_pair['top'])
    print_color_info("Bottoms", best_pair['bottom'])

    best_outer = None
    if not wearable_outers:
        print("\n--- 6. アウター検索 ---")
        print("  -> 体感温度に合う着用可能なアウターがありません。アウターなしを推奨します。")
    else:
        print(f"\n--- 6. ペアに合う最適なアウターを検索 ---")
        print(f" (アウター {len(wearable_outers)}着 で総当たり)")
        best_outer_score = -1.0
        chosen_top_hsv = best_pair['top'].hsv
        chosen_bottom_hsv = best_pair['bottom'].hsv
        for outer in wearable_outers:
            score_with_top = color_score(outer.hsv, chosen_top_hsv)
            score_with_bottom = color_score(outer.hsv, chosen_bottom_hsv)
            combined_score = (score_with_top + score_with_bottom) / 2.0
            print(f"  - 試行: {outer.name} (vs Top: {score_with_top:.3f}, vs Bottom: {score_with_bottom:.3f}) -> 総合: {combined_score:.3f}")
            if combined_score > best_outer_score:
                best_outer_score = combined_score
                best_outer = outer
        if best_outer:
            print(f"\n--- 6. 検索完了: ベストなアウター (スコア: {best_outer_score:.3f}) ---")
            print_color_info("Outerwear", best_outer)
        else:
            print("\n--- 6. 検索完了: マッチするアウターなし ---")

    print("\n--- コーデの基本色を確定 ---")
    chosen_hsv_list = [best_pair['top'].hsv, best_pair['bottom'].hsv]
    print(f"  - Tops: {best_pair['top'].name}")
    print(f"  - Bottoms: {best_pair['bottom'].name}")
    if best_outer:
        chosen_hsv_list.append(best_outer.hsv)
        print(f"  - Outerwear: {best_outer.name}")

    best_accessory = None
    if not wearable_accessories:
        print("\n--- 8. アクセサリー検索 ---")
        print("  -> 体感温度に合う着用可能なアクセサリーがありません。")
    else:
        print(f"\n--- 8. コーデに合う最適なアクセサリーを検索 ---")
        print(f" (アクセサリー {len(wearable_accessories)}点 で総当たり)")
        best_accessory_score = -1.0
        for accessory in wearable_accessories:
            total_score = 0
            for hsv in chosen_hsv_list:
                total_score += color_score(accessory.hsv, hsv)
            avg_score = total_score / len(chosen_hsv_list)
            print(f"  - 試行: {accessory.name} (平均スコア: {avg_score:.3f})")
            if avg_score > best_accessory_score:
                best_accessory_score = avg_score
                best_accessory = accessory
        if best_accessory: 
            print(f"\n--- 8. 検索完了: ベストなアクセサリー (スコア: {best_accessory_score:.3f}) ---")
            print_color_info("Accessory", best_accessory)
        else:
            print("\n--- 8. 検索完了: マッチするアクセサリーなし ---")

    # --- 9. 最適な「その他」を探す ---
    best_other = None
    if not wearable_others:
        print("\n--- 9. 「その他」アイテム検索 ---")
        print("  -> 体感温度に合う着用可能な「その他」アイテムがありません。")
    else:
        print(f"\n--- 9. コーデに合う最適な「その他」アイテムを検索 ---")
        print(f" (アイテム {len(wearable_others)}点 で総当たり)")
        best_other_score = -1.0
        for other_item in wearable_others:
            total_score = 0
            for hsv in chosen_hsv_list:
                total_score += color_score(other_item.hsv, hsv)
            avg_score = total_score / len(chosen_hsv_list)
            print(f"  - 試行: {other_item.name} (平均スコア: {avg_score:.3f})")
            if avg_score > best_other_score:
                best_other_score = avg_score
                best_other = other_item
        if best_other: 
            print(f"\n--- 9. 検索完了: ベストな「その他」 (スコア: {best_other_score:.3f}) ---")
            print_color_info("Other", best_other)
        else:
            print("\n--- 9. 検索完了: マッチする「その他」アイテムなし ---")

    # --- 10. 最終結果を辞書として返す ---
    
    # ★ 最終的な結果 (ClothingItem オブジェクトまたは None) を
    #   辞書にまとめて return する
    
    result = {
        "Tops": best_pair['top'],
        "Bottoms": best_pair['bottom'],
        "Outerwear": best_outer,      # None の場合あり
        "Accessory": best_accessory,  # None の場合あり
        "Other": best_other,        # None の場合あり
    }
    
    return result

# ====== 5. メイン実行ロジック ======
if __name__ == "__main__":
    
    # 1. closet.py からクローゼットの全アイテムを取得
    my_full_closet = initialize_closet()
    
    # 2. 条件を定義
    BASE_TEMPERATURE = 30
    CURRENT_WEATHER = "雨"
    
    # 3. メインの提案関数を実行
    suggest(items=my_full_closet, weather=CURRENT_WEATHER, temp=BASE_TEMPERATURE)