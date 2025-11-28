import cv2
import numpy as np


# ====== 基本関数群 ======

# 加重平均
def average(values, weights):
    return np.sum(values * weights) / np.sum(weights) 

# 色相環の加重平均
def circular_average(h_values, weights):
    h_deg = h_values * 2  # OpenCVではH:0–179 → 0–358°に変換
    h_rad = np.deg2rad(h_deg)

    x_mean = np.sum(np.cos(h_rad) * weights) 
    y_mean = np.sum(np.sin(h_rad) * weights) 

    angle = np.rad2deg(np.arctan2(y_mean, x_mean)) % 360
    return angle / 2  # OpenCVスケールに戻す



# ======= 色の距離とスコア ======

# 色相距離（度単位）
def hue_distance(h1, h2):
    d = abs(h1 - h2)
    return min(d, 360 - d)

# トーン（彩度・明度）スコア
def tone_score(s1, v1, s2, v2):
    s_diff = abs(s1 - s2)
    v_diff = abs(v1 - v2)
    score = 1 - (s_diff + v_diff) / 2
    return max(0, score)

# 総合スコア（色相＋トーン）
def color_score(hsv1, hsv2, w_h=0.7, w_t=0.3):
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2

    hue_sim = 1 - (hue_distance(h1, h2) / 180)
    hue_sim = max(0, hue_sim)
    tone_sim = tone_score(s1, v1, s2, v2)

    # 総合スコア
    total_score = w_h * hue_sim + w_t * tone_sim
    return round(max(0, min(1, total_score)), 3) # 小数点3位



# ======== 代表色抽出======

def extract_representative_color(image_path, reference_hsv=None):
    # 画像の読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    # BGRからHSVへの変換
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = cv2.split(hsv_image)

    # 画像サイズ
    h, w, _ = hsv_image.shape
    y, x = np.ogrid[:h, :w]

    # 重み設定（中央重視のガウス分布）
    cx, cy = w / 2, h / 2
    sigma_x, sigma_y = w / 4, h / 4  # 中央の重み範囲
    weight = np.exp(-(((x - cx) ** 2) / (2 * sigma_x ** 2) + ((y - cy) ** 2) / (2 * sigma_y ** 2)))
    weight /= np.sum(weight)

    # 各チャンネルの加重平均
    H_mean = circular_average(H, weight)
    S_mean = average(S, weight)
    V_mean = average(V, weight)

    # HSV正規化
    H_deg = H_mean * 2  # 0-360度
    S_norm = S_mean / 255.0  # 0-1
    V_norm = V_mean / 255.0  # 0-1
    hsv_color = (H_deg, S_norm, V_norm)

    # RGB変換
    rep_color_bgr = cv2.cvtColor(np.uint8([[[H_mean, S_mean, V_mean]]]), cv2.COLOR_HSV2BGR)[0][0]
    rgb_color = rep_color_bgr[::-1].tolist()  # BGR→RGB

    # 比較がある場合
    hue_dist = tone_s = color_s = None
    if reference_hsv is not None:
        ref_h, ref_s, ref_v = reference_hsv
        hue_dist = hue_distance(H_deg, ref_h)
        tone_s = tone_score(S_norm, V_norm, ref_s, ref_v)
        color_s = color_score(hsv_color, reference_hsv)

    return {
        'hsv': hsv_color,
        'rgb': rgb_color,
        'hue_distance': hue_dist,
        'tone_score': tone_s,
        'color_score': color_s
    }



# ====== 可視化・出力 ======

# 代表色の画像を作成
def show_color(rgb_color, window_name='Representative Color'):
    img = np.full((100, 100, 3), np.array(rgb_color,dtype=np.uint8) , dtype=np.uint8)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 結果を表示
def print_color_info(label, color_dict):
    h, s, v = color_dict['hsv']
    rgb = color_dict['rgb']
    hue_dist = color_dict['hue_distance']
    tone = color_dict['tone_score']
    color = color_dict['color_score']

    print(f"----- {label} -----")
    print(f"HSV: (H={h:.1f}°, S={s:.2f}, V={v:.2f})")
    print(f"RGB: {rgb}")

    if any(v is not None for v in [hue_dist, tone, color]):
        print ("\n----- 比較結果 -----")
    if hue_dist is not None:
        print(f"・色相距離: {hue_dist:.1f}°")
    if tone is not None:
        print(f"・トーン統一スコア: {tone:.2f}")
    if color is not None:
        print(f"・総合 色相性スコア: {color:.2f}")
    print("-------------------\n")

# 代表色を並べて表示
def compare_colors(rgb_a, rgb_b, label_a='Item 1', label_b='Item 2'):
    color_a = np.full((150, 150, 3), np.array(rgb_a,dtype=np.uint8) , dtype=np.uint8)
    color_b = np.full((150, 150, 3), np.array(rgb_b,dtype=np.uint8) , dtype=np.uint8)

    combined = np.hstack((color_a, color_b))
    cv2.imshow(f'{label_a} vs {label_b}', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# ====== 実行部分 ======
# 一枚目の画像抽出
color_1 = extract_representative_color('project_photo/test1.jpg')
print_color_info("Item 1", color_1)

# 二枚目の画像抽出
color_2 = extract_representative_color('project_photo/test3.jpg', reference_hsv=color_1['hsv'])
print_color_info("Item 2", color_2)

# 評価コメント
color_sim = color_2["color_score"]
if color_sim is not None:
    if color_sim >= 0.8:
        print("→ Good Match!")
    elif color_sim >= 0.5:
        print("→ Moderate Match.")
    else:
        print("→ Poor Match.")
else:
    print("→ 比較スコアが計算できませんでした。")

# 色の比較表示
compare_colors(color_1['rgb'], color_2['rgb'], "Item 1", "Item 2")