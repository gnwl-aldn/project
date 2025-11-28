import cv2
import numpy as np
import json
import sys
from collections import defaultdict
from typing import List

# ===================================================================
# ====== 1. 色彩分析関数群 (Script 1 ベース) ======
# ===================================================================

# (変更なし ... )
# 加重平均
def average(values, weights):
    """ 通常の加重平均 """
    return np.sum(values * weights) / np.sum(weights)

# 色相（円データ）の加重平均
def circular_average(h_values, weights):
    """ 色相 (0-179) のための加重平均 """
    # 0-358°に変換
    h_deg = h_values * 2
    h_rad = np.deg2rad(h_deg)

    # xとyの平均を計算
    x_mean = np.sum(np.cos(h_rad) * weights)
    y_mean = np.sum(np.sin(h_rad) * weights)

    # 角度に戻し、% 360で範囲内に収める
    angle = np.rad2deg(np.arctan2(y_mean, x_mean)) % 360
    # OpenCVスケール (0-179) に戻す
    return angle / 2

# --- 色の距離とスコア ---

def hue_distance(h1, h2):
    """ 
    色相の距離を計算 (0-360°スケール)
    h1, h2: 0-360°の値
    """
    d = abs(h1 - h2)
    return min(d, 360 - d)

def tone_score(s1, v1, s2, v2):
    """ 
    トーン（彩度・明度）の類似度を計算 (0-1スケール)
    s, v: 0.0 - 1.0 の値
    """
    # 差の絶対値を計算 (最大1.0)
    s_diff = abs(s1 - s2)
    v_diff = abs(v1 - v2)
    # 差の平均を1から引く (0.0 - 1.0 のスコア)
    score = 1 - (s_diff + v_diff) / 2.0
    return max(0, score)

def color_score(hsv1, hsv2, w_h=0.7, w_t=0.3):
    """ 
    総合的な色相性スコアを計算 (0-1スケール)
    hsv: (H: 0-360°, S: 0-1, V: 0-1) のタプル
    w_h: 色相(Hue)の重み
    w_t: トーン(Tone)の重み
    """
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2

    # 色相の類似度 (0-1)
    # 距離 (最大180) を 180 で割って反転
    hue_sim = 1 - (hue_distance(h1, h2) / 180.0)
    hue_sim = max(0, hue_sim)

    # トーンの類似度 (0-1)
    tone_sim = tone_score(s1, v1, s2, v2)

    # 総合スコア（加重平均）
    total_score = w_h * hue_sim + w_t * tone_sim
    # 0-1の範囲に丸める
    return round(max(0, min(1, total_score)), 3)

# --- 代表色抽出 (ロジックを簡略化) ---

def extract_representative_color(image_path):
    """
    画像から中央重点の加重平均で代表色を抽出する
    - image_path: 画像ファイルのパス
    - 戻り値: 色情報を含む辞書
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    # BGRからHSV (float32) へ変換
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = cv2.split(hsv_image)

    # 画像サイズ
    h, w, _ = hsv_image.shape
    y, x = np.ogrid[:h, :w]

    # 重み設定（中央重視のガウス分布）
    cx, cy = w / 2, h / 2
    # 標準偏差を小さくすると、より中央が重視される
    sigma_x, sigma_y = w / 4, h / 4
    weight = np.exp(-(((x - cx) ** 2) / (2 * sigma_x ** 2) + ((y - cy) ** 2) / (2 * sigma_y ** 2)))
    # 重みをフラット化
    weight_flat = weight.flatten()

    # 各チャンネルの加重平均
    # H (0-179), S (0-255), V (0-255)
    H_mean = circular_average(H.flatten(), weight_flat)
    S_mean = average(S.flatten(), weight_flat)
    V_mean = average(V.flatten(), weight_flat)

    # スコア計算用の標準フォーマット (H: 0-360, S: 0-1, V: 0-1)
    hsv_normalized = (
        H_mean * 2,     # 0-360°
        S_mean / 255.0, # 0-1
        V_mean / 255.0  # 0-1
    )

    # RGB変換用のHSV (OpenCVスケール)
    hsv_opencv = np.uint8([[[H_mean, S_mean, V_mean]]])
    rep_color_bgr = cv2.cvtColor(hsv_opencv, cv2.COLOR_HSV2BGR)[0][0]
    
    # BGR -> RGB
    rgb_color = rep_color_bgr[::-1].tolist()

    return {
        'hsv': hsv_normalized,
        'rgb': rgb_color
    }

# ====== 2. JSON & 服装フィルタリング関数群 (Script 2) ======

def load_json_data(file_path):
    " 指定されたJSONファイルを読み込む "
    print(f"\n--- データをロード中: {file_path} ---")
    try:
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
    " 気温に基づいて着用可能な服の「タイプ」リストを返す "
    print(f"\n--- 関数: 体感温度 {temperature}度 に合う服の「種類」を検索 ---")
    allowed_types = []
    for item_type, rule in rules.items():
        min_temp = rule.get('min')
        max_temp = rule.get('max')
        
        # Noneは「制限なし」として扱う
        is_min_ok = (min_temp is None) or (temperature >= min_temp)
        is_max_ok = (max_temp is None) or (temperature <= max_temp)
            
        if is_min_ok and is_max_ok:
            allowed_types.append(item_type)
            
    print(f"  -> 許可された服のタイプ (全{len(allowed_types)}種)")
    return allowed_types

def filter_my_closet(allowed_types, closet):
    " 自分のクローゼットを「許可されたタイプ」で絞り込む "
    print("\n--- 関数: 自分の服と「許可された種類」を照合 ---")
    allowed_types_set = set(allowed_types)
    wearable_clothes = []
    
    for item in closet:
        # JSONに 'type' キーがあるか確認
        if 'type' not in item:
            print(f"  [警告] {item.get('name', '不明なアイテム')} に 'type' がありません。スキップします。")
            continue
            
        if item['type'] in allowed_types_set:
            wearable_clothes.append(item)
            print(f"  [OK] {item['name']} (タイプ: {item['type']})")
        else:
            print(f"  [NG] {item['name']} (タイプ: {item['type']}) - 体感温度に合いません")
            
    return wearable_clothes

# ====== 3. 可視化・出力関数群 (Script 1 ベース) ======

def print_color_info(label, item):
    " アイテムの色情報を表示 "
    h, s, v = item['hsv']
    rgb = item['rgb']
    print(f"----- {label}: {item['name']} -----")
    print(f"  HSV: (H={h:.1f}°, S={s:.2f}, V={v:.2f})")
    print(f"  RGB: {rgb}")

def show_final_outfit(items):
    " 最終的なコーディネートの色を並べて表示 items: 代表色情報('rgb')を含むアイテム辞書のリスト"
    color_blocks = []
    labels = []
    
    for item in items:
        if item: # itemがNoneでないことを確認
            rgb_color = item['rgb']
            block = np.full((150, 150, 3), np.array(rgb_color, dtype=np.uint8), dtype=np.uint8)
            color_blocks.append(block)
            labels.append(item['category'])

    if not color_blocks:
        print("表示する色がありません。")
        return

    # 画像を水平に連結
    combined_image = np.hstack(color_blocks)
    
    # ウィンドウタイトルが長くなりすぎるのを防ぐ
    window_title_short = ' & '.join(labels)
    if len(window_title_short) > 100:
        window_title_short = f"{len(labels)} items"
        
    cv2.imshow(f'Final Outfit: {window_title_short}', combined_image)
    print("\n代表色のウィンドウが表示されました。キーを押すと終了します。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ====== . メイン実行ロジック ======
class Item():
    pass

def suggest(items:List[Item], weather:str, temp: int):
    # --- 1. 設定とデータ読み込み ---
    CLOSET_FILE = "closet.json"
    RULES_FILE = "rules.json"
    
    # 天気による体感温度の調整
    weather_adjustment = {
        "晴れ": 2,  
        "曇り": 0,   
        "雨": -2,   
        "雪": -3 
    }
    
    adjustment = weather_adjustment.get(CURRENT_WEATHER, 0)
    
    # 最終的に使用する「体感温度」
    ADJUSTED_TEMPERATURE = BASE_TEMPERATURE + adjustment


    print(f"*** ファッションコーディネート提案 ***")
    print(f"現在の設定気温: {BASE_TEMPERATURE}度")
    print(f"天気: {CURRENT_WEATHER} (体感調整: {adjustment:+}度)")
    print(f" -> 体感温度: {ADJUSTED_TEMPERATURE}度 で服を検索します")
    
    ALL_ITEMS_RULES = load_json_data(RULES_FILE)

    if ALL_ITEMS_RULES is None or items is None:
        print("\n[エラー] データファイルの読み込みに失敗したため、処理を終了します。")
        sys.exit() # プログラムを終了

    # 2. 気温に基づく服の絞り込み 
    allowed_types = get_allowed_clothing_types(ADJUSTED_TEMPERATURE, ALL_ITEMS_RULES)
    wearable_clothes = filter_my_closet(allowed_types, items) # 関数の入力形式を合わせる

    if not wearable_clothes:
        #  メッセージを体感温度基準に変更
        raise f"\n[結果] 体感温度{ADJUSTED_TEMPERATURE}度で着られる服はありません。"

    print("\n--- 3. 着用可能な服の代表色を抽出中... ---")
    
    categorized_closet = defaultdict(list)
    
    for item in wearable_clothes:
        # 必要なキーのチェック
        if 'category' not in item or 'image_path' not in item:
            print(f"  [警告] {item['name']} に 'category' または 'image_path' がありません。スキップします。")
            continue
            
        try:
            # 色情報を抽出し、アイテム辞書に追加
            color_data = extract_representative_color(item['image_path'])
            item.update(color_data) # 'hsv' と 'rgb' キーが追加される
            
            # カテゴリ別に分類
            categorized_closet[item['category']].append(item)
            print(f"  [成功] {item['name']} ({item['category']}) の色を抽出完了。")
            
        except FileNotFoundError:
            print(f"  [エラー] {item['name']} の画像 ({item['image_path']}) が見つかりません。")
        except Exception as e:
            print(f"  [エラー] {item['name']} の処理中に問題発生: {e}")

    # 分類結果を取得
    wearable_tops = categorized_closet.get('Tops', [])
    wearable_bottoms = categorized_closet.get('Bottoms', [])
    wearable_outers = categorized_closet.get('Outerwear', [])
    wearable_others = categorized_closet.get('Other', [])
    # ★★★ 変更点 1: Accessory リストを取得 ★★★
    wearable_accessories = categorized_closet.get('Accessory', [])

    # --- 4. 必須アイテム（トップス・ボトムス）の確認 ---
    if not wearable_tops or not wearable_bottoms:
        print("\n[エラー] 着用可能なトップス、またはボトムスが見つかりませんでした。")
        print(f"  (トップス: {len(wearable_tops)}着, ボトムス: {len(wearable_bottoms)}着)")
        sys.exit()
    else:
        print(f"\n--- 4. 最適なトップスとボトムスのペアを検索 ---")
        print(f" (トップス {len(wearable_tops)}着, ボトムス {len(wearable_bottoms)}着 で総当たり)")

    # --- 5. 最高のトップスxボトムス ペアを探す ---
    best_pair = None
    best_pair_score = -1.0 

    for top in wearable_tops:
        for bottom in wearable_bottoms:
            score = color_score(top['hsv'], bottom['hsv'])
            
            if score > best_pair_score:
                best_pair_score = score
                best_pair = {'top': top, 'bottom': bottom}

    print(f"\n--- 5. 検索完了: ベストペア (スコア: {best_pair_score:.3f}) ---")
    print_color_info("Tops", best_pair['top'])
    print_color_info("Bottoms", best_pair['bottom'])

    # --- 6. 最高のペアに合うアウターを探す ---
    best_outer = None
    
    if not wearable_outers:
        print("\n--- 6. アウター検索 ---")
        print("  -> 体感温度に合う着用可能なアウターがありません。アウターなしを推奨します。")
    else:
        print(f"\n--- 6. ペアに合う最適なアウターを検索 ---")
        print(f" (アウター {len(wearable_outers)}着 で総当たり)")
        
        best_outer_score = -1.0
        chosen_top_hsv = best_pair['top']['hsv']
        chosen_bottom_hsv = best_pair['bottom']['hsv']

        for outer in wearable_outers:
            score_with_top = color_score(outer['hsv'], chosen_top_hsv)
            score_with_bottom = color_score(outer['hsv'], chosen_bottom_hsv)
            combined_score = (score_with_top + score_with_bottom) / 2.0
            
            print(f"  - 試行: {outer['name']} (vs Top: {score_with_top:.3f}, vs Bottom: {score_with_bottom:.3f}) -> 総合: {combined_score:.3f}")

            if combined_score > best_outer_score:
                best_outer_score = combined_score
                best_outer = outer
        
        if best_outer:
            print(f"\n--- 6. 検索完了: ベストなアウター (スコア: {best_outer_score:.3f}) ---")
            print_color_info("Outerwear", best_outer)
        else:
            print("\n--- 6. 検索完了: マッチするアウターなし ---")

    # --- 7. (新設) 決定済みコーデのHSVリストを作成 ---
    # (Step 8 と 9 で共通して使用する)
    print("\n--- コーデの基本色を確定 ---")
    chosen_hsv_list = [
        best_pair['top']['hsv'],
        best_pair['bottom']['hsv']
    ]
    print(f"  - Tops: {best_pair['top']['name']}")
    print(f"  - Bottoms: {best_pair['bottom']['name']}")
    
    if best_outer:
        chosen_hsv_list.append(best_outer['hsv'])
        print(f"  - Outerwear: {best_outer['name']}")
    

    # ★★★ 変更点 2: Accessory のスコア計算ロジックを追加 ★★★
    # --- 8. コーデに合う最適なアクセサリーを探す ---
    best_accessory = None
    if not wearable_accessories:
        print("\n--- 8. アクセサリー検索 ---")
        print("  -> 体感温度に合う着用可能なアクセサリーがありません。")
    else:
        # 1つでも複数でもスコア計算を実行
        print(f"\n--- 8. コーデに合う最適なアクセサリーを検索 ---")
        print(f" (アクセサリー {len(wearable_accessories)}点 で総当たり)")
        
        best_accessory_score = -1.0

        for accessory in wearable_accessories:
            total_score = 0
            # 決定したコーデの各アイテム(Top, Bottom, Outer)とのスコアを合計
            for hsv in chosen_hsv_list:
                total_score += color_score(accessory['hsv'], hsv)
            
            # 平均スコアで評価
            avg_score = total_score / len(chosen_hsv_list)
            
            print(f"  - 試行: {accessory['name']} (平均スコア: {avg_score:.3f})")

            if avg_score > best_accessory_score:
                best_accessory_score = avg_score
                best_accessory = accessory
        
        if best_accessory: # 見つかった場合のみ表示
            print(f"\n--- 8. 検索完了: ベストなアクセサリー (スコア: {best_accessory_score:.3f}) ---")
            print_color_info("Accessory", best_accessory)
        else:
            print("\n--- 8. 検索完了: マッチするアクセサリーなし ---")


    # --- 9. (旧 8) コーデに合う「その他」を探す ---
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
                total_score += color_score(other_item['hsv'], hsv)
            
            avg_score = total_score / len(chosen_hsv_list)
            
            print(f"  - 試行: {other_item['name']} (平均スコア: {avg_score:.3f})")

            if avg_score > best_other_score:
                best_other_score = avg_score
                best_other = other_item
        
        if best_other: 
            print(f"\n--- 9. 検索完了: ベストな「その他」 (スコア: {best_other_score:.3f}) ---")
            print_color_info("Other", best_other)
        else:
            print("\n--- 9. 検索完了: マッチする「その他」アイテムなし ---")
    
    # --- 10. (旧 9) 最終結果の表示 ---
    print("\n===================================")
    print(f"     気温 {BASE_TEMPERATURE}度 ({CURRENT_WEATHER}) のおすすめコーデ")
    print(f"     (体感温度 {ADJUSTED_TEMPERATURE}度 基準)")
    print("===================================")
    
    final_items = [] 
    
    print(f"  Tops:    {best_pair['top']['name']}")
    final_items.append(best_pair['top'])
    
    print(f"  Bottoms: {best_pair['bottom']['name']}")
    final_items.append(best_pair['bottom'])
    
    if best_outer:
        print(f"  Outer:   {best_outer['name']}")
        final_items.append(best_outer)
    else:
        print("  Outer:   (アウターなし)")
    
    if best_other:
        print(f"  Other:   {best_other['name']}")
        final_items.append(best_other)
    else:
        print("  Other:   (なし)")
        
    print("===================================")
    
    if best_accessory:
        print(f"  気温に適した Accessory: {best_accessory['name']}")
        final_items.append(best_accessory)
    else:
        print("  Accessory: (なし)")

    # 最終的な色の組み合わせを可視化
    try:
        show_final_outfit(final_items)
    except Exception as e:
        print(f"\n[警告] 色の可視化ウィンドウの表示に失敗しました: {e}")
        print("（OpenCVがGUI環境で実行されていない可能性があります）")
    return


if __name__ == "__main__":
    
    # --- 1. 設定とデータ読み込み ---
    
    BASE_TEMPERATURE = 7    # (例) 基準の気温
    CURRENT_WEATHER = "雨"   # "晴れ", "曇り", "雨" などを指定
    CLOSET_FILE = "closet.json"
    RULES_FILE = "rules.json"
    
    # 天気による体感温度の調整
    weather_adjustment = {
        "晴れ": 2,  
        "曇り": 0,   
        "雨": -2,   
        "雪": -3 
    }
    
    adjustment = weather_adjustment.get(CURRENT_WEATHER, 0)
    
    # 最終的に使用する「体感温度」
    ADJUSTED_TEMPERATURE = BASE_TEMPERATURE + adjustment

    print(f"*** ファッションコーディネート提案 ***")
    print(f"現在の設定気温: {BASE_TEMPERATURE}度")
    print(f"天気: {CURRENT_WEATHER} (体感調整: {adjustment:+}度)")
    print(f" -> 体感温度: {ADJUSTED_TEMPERATURE}度 で服を検索します")
    
    ALL_ITEMS_RULES = load_json_data(RULES_FILE)
    my_closet = load_json_data(CLOSET_FILE)

    if ALL_ITEMS_RULES is None or my_closet is None:
        print("\n[エラー] データファイルの読み込みに失敗したため、処理を終了します。")
        sys.exit() # プログラムを終了

    # 2. 気温に基づく服の絞り込み 
    allowed_types = get_allowed_clothing_types(ADJUSTED_TEMPERATURE, ALL_ITEMS_RULES)
    wearable_clothes = filter_my_closet(allowed_types, my_closet)

    if not wearable_clothes:
        #  メッセージを体感温度基準に変更
        print(f"\n[結果] 体感温度{ADJUSTED_TEMPERATURE}度で着られる服はありません。")
        sys.exit()

    print("\n--- 3. 着用可能な服の代表色を抽出中... ---")
    
    categorized_closet = defaultdict(list)
    
    for item in wearable_clothes:
        # 必要なキーのチェック
        if 'category' not in item or 'image_path' not in item:
            print(f"  [警告] {item['name']} に 'category' または 'image_path' がありません。スキップします。")
            continue
            
        try:
            # 色情報を抽出し、アイテム辞書に追加
            color_data = extract_representative_color(item['image_path'])
            item.update(color_data) # 'hsv' と 'rgb' キーが追加される
            
            # カテゴリ別に分類
            categorized_closet[item['category']].append(item)
            print(f"  [成功] {item['name']} ({item['category']}) の色を抽出完了。")
            
        except FileNotFoundError:
            print(f"  [エラー] {item['name']} の画像 ({item['image_path']}) が見つかりません。")
        except Exception as e:
            print(f"  [エラー] {item['name']} の処理中に問題発生: {e}")

    # 分類結果を取得
    wearable_tops = categorized_closet.get('Tops', [])
    wearable_bottoms = categorized_closet.get('Bottoms', [])
    wearable_outers = categorized_closet.get('Outerwear', [])
    wearable_others = categorized_closet.get('Other', [])
    # ★★★ 変更点 1: Accessory リストを取得 ★★★
    wearable_accessories = categorized_closet.get('Accessory', [])

    # --- 4. 必須アイテム（トップス・ボトムス）の確認 ---
    if not wearable_tops or not wearable_bottoms:
        print("\n[エラー] 着用可能なトップス、またはボトムスが見つかりませんでした。")
        print(f"  (トップス: {len(wearable_tops)}着, ボトムス: {len(wearable_bottoms)}着)")
        sys.exit()
    else:
        print(f"\n--- 4. 最適なトップスとボトムスのペアを検索 ---")
        print(f" (トップス {len(wearable_tops)}着, ボトムス {len(wearable_bottoms)}着 で総当たり)")

    # --- 5. 最高のトップスxボトムス ペアを探す ---
    best_pair = None
    best_pair_score = -1.0 

    for top in wearable_tops:
        for bottom in wearable_bottoms:
            score = color_score(top['hsv'], bottom['hsv'])
            
            if score > best_pair_score:
                best_pair_score = score
                best_pair = {'top': top, 'bottom': bottom}

    print(f"\n--- 5. 検索完了: ベストペア (スコア: {best_pair_score:.3f}) ---")
    print_color_info("Tops", best_pair['top'])
    print_color_info("Bottoms", best_pair['bottom'])

    # --- 6. 最高のペアに合うアウターを探す ---
    best_outer = None
    
    if not wearable_outers:
        print("\n--- 6. アウター検索 ---")
        print("  -> 体感温度に合う着用可能なアウターがありません。アウターなしを推奨します。")
    else:
        print(f"\n--- 6. ペアに合う最適なアウターを検索 ---")
        print(f" (アウター {len(wearable_outers)}着 で総当たり)")
        
        best_outer_score = -1.0
        chosen_top_hsv = best_pair['top']['hsv']
        chosen_bottom_hsv = best_pair['bottom']['hsv']

        for outer in wearable_outers:
            score_with_top = color_score(outer['hsv'], chosen_top_hsv)
            score_with_bottom = color_score(outer['hsv'], chosen_bottom_hsv)
            combined_score = (score_with_top + score_with_bottom) / 2.0
            
            print(f"  - 試行: {outer['name']} (vs Top: {score_with_top:.3f}, vs Bottom: {score_with_bottom:.3f}) -> 総合: {combined_score:.3f}")

            if combined_score > best_outer_score:
                best_outer_score = combined_score
                best_outer = outer
        
        if best_outer:
            print(f"\n--- 6. 検索完了: ベストなアウター (スコア: {best_outer_score:.3f}) ---")
            print_color_info("Outerwear", best_outer)
        else:
            print("\n--- 6. 検索完了: マッチするアウターなし ---")

    # --- 7. (新設) 決定済みコーデのHSVリストを作成 ---
    # (Step 8 と 9 で共通して使用する)
    print("\n--- コーデの基本色を確定 ---")
    chosen_hsv_list = [
        best_pair['top']['hsv'],
        best_pair['bottom']['hsv']
    ]
    print(f"  - Tops: {best_pair['top']['name']}")
    print(f"  - Bottoms: {best_pair['bottom']['name']}")
    
    if best_outer:
        chosen_hsv_list.append(best_outer['hsv'])
        print(f"  - Outerwear: {best_outer['name']}")
    

    # ★★★ 変更点 2: Accessory のスコア計算ロジックを追加 ★★★
    # --- 8. コーデに合う最適なアクセサリーを探す ---
    best_accessory = None
    if not wearable_accessories:
        print("\n--- 8. アクセサリー検索 ---")
        print("  -> 体感温度に合う着用可能なアクセサリーがありません。")
    else:
        # 1つでも複数でもスコア計算を実行
        print(f"\n--- 8. コーデに合う最適なアクセサリーを検索 ---")
        print(f" (アクセサリー {len(wearable_accessories)}点 で総当たり)")
        
        best_accessory_score = -1.0

        for accessory in wearable_accessories:
            total_score = 0
            # 決定したコーデの各アイテム(Top, Bottom, Outer)とのスコアを合計
            for hsv in chosen_hsv_list:
                total_score += color_score(accessory['hsv'], hsv)
            
            # 平均スコアで評価
            avg_score = total_score / len(chosen_hsv_list)
            
            print(f"  - 試行: {accessory['name']} (平均スコア: {avg_score:.3f})")

            if avg_score > best_accessory_score:
                best_accessory_score = avg_score
                best_accessory = accessory
        
        if best_accessory: # 見つかった場合のみ表示
            print(f"\n--- 8. 検索完了: ベストなアクセサリー (スコア: {best_accessory_score:.3f}) ---")
            print_color_info("Accessory", best_accessory)
        else:
            print("\n--- 8. 検索完了: マッチするアクセサリーなし ---")


    # --- 9. (旧 8) コーデに合う「その他」を探す ---
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
                total_score += color_score(other_item['hsv'], hsv)
            
            avg_score = total_score / len(chosen_hsv_list)
            
            print(f"  - 試行: {other_item['name']} (平均スコア: {avg_score:.3f})")

            if avg_score > best_other_score:
                best_other_score = avg_score
                best_other = other_item
        
        if best_other: 
            print(f"\n--- 9. 検索完了: ベストな「その他」 (スコア: {best_other_score:.3f}) ---")
            print_color_info("Other", best_other)
        else:
            print("\n--- 9. 検索完了: マッチする「その他」アイテムなし ---")
    
    # --- 10. (旧 9) 最終結果の表示 ---
    print("\n===================================")
    print(f"     気温 {BASE_TEMPERATURE}度 ({CURRENT_WEATHER}) のおすすめコーデ")
    print(f"     (体感温度 {ADJUSTED_TEMPERATURE}度 基準)")
    print("===================================")
    
    final_items = [] 
    
    print(f"  Tops:    {best_pair['top']['name']}")
    final_items.append(best_pair['top'])
    
    print(f"  Bottoms: {best_pair['bottom']['name']}")
    final_items.append(best_pair['bottom'])
    
    if best_outer:
        print(f"  Outer:   {best_outer['name']}")
        final_items.append(best_outer)
    else:
        print("  Outer:   (アウターなし)")
    
    if best_other:
        print(f"  Other:   {best_other['name']}")
        final_items.append(best_other)
    else:
        print("  Other:   (なし)")
        
    print("===================================")
    
    if best_accessory:
        print(f"  気温に適した Accessory: {best_accessory['name']}")
        final_items.append(best_accessory)
    else:
        print("  Accessory: (なし)")

    # 最終的な色の組み合わせを可視化
    try:
        show_final_outfit(final_items)
    except Exception as e:
        print(f"\n[警告] 色の可視化ウィンドウの表示に失敗しました: {e}")
        print("（OpenCVがGUI環境で実行されていない可能性があります）")