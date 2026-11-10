# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""植被率推定の計算コア（純関数群）。

RGB画像から植生マスクを作り、植被率(FVC)を算出するための関数を集約する。
GUIにもワーカープロセスにも依存しない。

v3 の方針:
- 学習器は Random Forest、特徴量は Lab 色空間の3値に固定する（選択肢を持たない）
- アルファ（透明）を持つ画像では、透明な画素を「存在しない画素」として扱う。
  植被率の分母にも学習データにも入れない（QGIS 等で区画を斜めに切り出した画像への対応）
- v2 以前のスクリプトとの互換のため、旧API の get_pixel_features /
  segment_image_with_method は残す（tests/equivalence_harness.py も使う）。アプリ本体はこれらを使わない
"""

import cv2
import numpy as np
from PIL import Image

# --- 固定パラメータ（再現性のため、ここ以外で値を持たない） ---
FEATURE_MODE = "Lab"
ALGO = "RF"
RF_N_ESTIMATORS = 200
RANDOM_STATE = 42
ALPHA_VALID_THRESHOLD = 128          # アルファがこの値以上の画素を「有効」とする
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
OVERLAY_RGBA = (255, 0, 0, 80)       # 判定結果の植生色（赤・半透明）
INVALID_RGB = (128, 128, 128)        # 出力画像で ROI 外（透明画素）を塗る色


def imread_unicode(path):
    """日本語を含むパスでも読める imread（BGR のみ。代表画像選抜など軽い用途向け）。

    cv2.imread は非ASCIIパスで例外を出さずに None を返すため、
    np.fromfile + cv2.imdecode を使う（研究データのフォルダ名は日本語率が高い）。
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _has_alpha(im):
    return im.mode in ("RGBA", "LA", "PA", "RGBa") or (im.mode == "P" and "transparency" in im.info)


def load_image(path):
    """画像を BGR uint8 配列と有効画素マスクに分けて読み込む。

    Returns:
        (img_bgr, valid): valid は bool (H, W)。アルファを持たない画像では None（全画素有効）。
        アルファを持つ画像（RGBA / LA / 透明色つきパレット）では alpha >= 128 の画素だけが有効。
        16bit グレースケール TIFF は 8bit に落として3チャンネル化する。
    """
    with Image.open(path) as im:
        im.load()
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(im).astype(np.float32)
            if arr.max() > 255:
                arr = arr * (255.0 / 65535.0)
            gray = np.clip(arr, 0, 255).astype(np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), None
        if _has_alpha(im):
            rgba = np.asarray(im.convert("RGBA"))
            valid = rgba[..., 3] >= ALPHA_VALID_THRESHOLD
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), valid
        rgb = np.asarray(im.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None


def lab_features(img_bgr):
    """アプリ本体の特徴量: Lab 色空間の (L, a, b) 3値（uint8）。画素ごとに1行"""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).reshape(-1, 3)


def calculate_exg_mask(img_bgr):
    """Excess Green Index (2G - R - B) + Otsu二値化で植生マスクを生成する（研究用の比較手法）"""
    b_channel, g_channel, r_channel = cv2.split(img_bgr)
    exg = (2 * g_channel.astype(np.int32)) - r_channel.astype(np.int32) - b_channel.astype(np.int32)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
    exg_uint8 = exg_norm.astype(np.uint8)
    otsu_threshold, binary = cv2.threshold(
        exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask = binary == 255
    return mask, otsu_threshold


def create_overlay_from_mask(mask, valid=None):
    """2値マスクから RGBA のオーバーレイ画像を生成する（植生=赤・半透明）。
    valid を渡すと、無効画素には何も描かない"""
    if valid is not None:
        mask = mask & valid
    overlay_data = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay_data[mask] = OVERLAY_RGBA
    return Image.fromarray(overlay_data)  # uint8 の H×W×4 なので RGBA になる


def create_overlay_from_mask_colored(mask, rgba=(255, 255, 0, 90)):
    """2値マスクから色指定RGBAのオーバーレイ画像を生成する（研究用）"""
    overlay_data = np.zeros((*mask.shape, 4), dtype=np.uint8)
    r, g, b, a = rgba
    overlay_data[mask] = [r, g, b, a]
    return Image.fromarray(overlay_data)  # uint8 の H×W×4 なので RGBA になる


def calculate_coverage(mask, valid=None):
    """マスクから植被率(%)を算出する。

    valid を渡すと、分母は有効画素数、分子は「植生かつ有効」の画素数になる。
    有効画素が1つも無い画像では None を返す（植被率は定義できない）。
    """
    if valid is None:
        plant_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
    else:
        plant_pixels = np.count_nonzero(mask & valid)
        total_pixels = np.count_nonzero(valid)
    if total_pixels == 0:
        return 0.0 if valid is None else None
    return (plant_pixels / total_pixels) * 100.0


def valid_fraction(valid, shape=None):
    """有効画素の割合(%)。valid が None なら 100.0"""
    if valid is None:
        return 100.0
    if valid.size == 0:
        return 0.0
    return float(np.count_nonzero(valid)) / float(valid.size) * 100.0


def predict_mask(img_bgr, model):
    """学習済みモデルで植生マスク（bool, H×W）を推定する。

    Lab の3値は BGR 画素値の純関数なので、まず BGR でユニーク色に畳み、ユニーク色だけ
    特徴量計算・予測して全画素に引き直す。画素数百万に対し色は数万通りしかなく、
    predict が数十倍速くなる（結果は全画素予測と完全に同一）。
    """
    if model is None:
        raise ValueError("モデルが学習されていません。")
    flat_bgr = img_bgr.reshape(-1, img_bgr.shape[2])
    uniq_bgr, inverse = _unique_rows(flat_bgr)
    features_u = lab_features(uniq_bgr.reshape(-1, 1, img_bgr.shape[2]))
    predictions = _predict_pixels(model, features_u)[inverse]
    return predictions.reshape(img_bgr.shape[:2]) == 1


def segment_image(img_bgr, model, valid=None):
    """推定マスクに有効画素マスクを掛けて返す（無効画素は常に False）"""
    mask = predict_mask(img_bgr, model)
    if valid is not None:
        mask = mask & valid
    return mask


def compose_result_image(img_bgr, mask, valid=None):
    """確認用の判定画像（PIL RGB）を作る。植生は赤を半透明で重ね、無効画素はグレーで塗る。
    RGB 画像では v2 までの出力（alpha_composite）とビット単位で同じになる"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if valid is not None:
        rgb = rgb.copy()
        rgb[~valid] = INVALID_RGB
    base = Image.fromarray(rgb).convert("RGBA")  # uint8 の H×W×3 なので RGB になる
    overlay = create_overlay_from_mask(mask, valid)
    return Image.alpha_composite(base, overlay).convert("RGB")


def extract_training_pixels(annotations, loader=load_image):
    """教師点から学習用の特徴量行列を作る。

    annotations: {image_path: [{'nx': 0-1, 'ny': 0-1, 'class': 'plant'|'soil', ...}, ...]}
    点は正規化座標なので、画像を読み込んだ解像度でその位置の1画素を採る。
    無効画素（透明）に打たれた点は除外する。

    順序は (画像パス, クラス, ny, nx) で固定する。Random Forest はサンプルの並びにも
    依存するため、同じ教師点からは常に同じ行列（＝同じモデル）が得られるようにする。

    Returns:
        X (N,3) uint8, y (N,) int (1=plant, 0=soil), stats dict
    """
    plant_rows, soil_rows = [], []
    stats = {"n_images": 0, "n_plant": 0, "n_soil": 0, "n_skipped_invalid": 0, "n_failed_images": 0}
    for path in sorted(annotations.keys()):
        points = annotations[path]
        if not points:
            continue
        try:
            img_bgr, valid = loader(path)
        except Exception:
            stats["n_failed_images"] += 1
            continue
        stats["n_images"] += 1
        h, w = img_bgr.shape[:2]
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        for ann in sorted(points, key=lambda a: (a.get('class', ''), float(a['ny']), float(a['nx']))):
            x = min(w - 1, max(0, int(float(ann['nx']) * w)))
            y = min(h - 1, max(0, int(float(ann['ny']) * h)))
            if valid is not None and not valid[y, x]:
                stats["n_skipped_invalid"] += 1
                continue
            if ann.get('class') == 'plant':
                plant_rows.append(lab[y, x])
                stats["n_plant"] += 1
            else:
                soil_rows.append(lab[y, x])
                stats["n_soil"] += 1
    if plant_rows or soil_rows:
        X = np.array(plant_rows + soil_rows, dtype=np.uint8)
    else:
        X = np.zeros((0, 3), dtype=np.uint8)
    y = np.array([1] * len(plant_rows) + [0] * len(soil_rows), dtype=int)
    return X, y, stats


def train_rf(X, y):
    """固定ハイパーパラメータの Random Forest を学習する（seed 固定で決定的）"""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model


# ==========================================================
# 互換API（v2 以前のスクリプトと tests/equivalence_harness.py が使う。アプリ本体は使わない）
# ==========================================================

def segment_image_with_method(img_bgr, mode, model, segmentation_method, include_exg):
    """指定手法でマスクを生成し、付随情報を返す（旧API）"""
    if segmentation_method == "exg":
        mask, threshold = calculate_exg_mask(img_bgr)
        return mask, {'otsu_threshold': threshold}
    if model is None:
        raise ValueError("モデルが学習されていません。")
    flat_bgr = img_bgr.reshape(-1, img_bgr.shape[2])
    uniq_bgr, inverse = _unique_rows(flat_bgr)
    features_u = get_pixel_features(uniq_bgr.reshape(-1, 1, img_bgr.shape[2]), mode, include_exg)
    predictions = _predict_pixels(model, features_u)[inverse]
    mask = predictions.reshape(img_bgr.shape[:2]) == 1
    return mask, {}


def _unique_rows(features):
    """整数特徴量のユニーク行と逆引きインデックスを返す。

    3〜4列はビットパッキング、それ以上はバイト列ビューで np.unique にかける。
    """
    f = np.ascontiguousarray(features)
    n_cols = f.shape[1]
    if f.dtype == np.uint8 and n_cols <= 4:
        packed = np.zeros(f.shape[0], dtype=np.uint32)
        for c in range(n_cols):
            packed = (packed << 8) | f[:, c].astype(np.uint32)
        uniq, inverse = np.unique(packed, return_inverse=True)
        cols = []
        for c in range(n_cols):
            shift = 8 * (n_cols - 1 - c)
            cols.append((uniq >> shift) & 255)
        uniq_rows = np.stack(cols, axis=1).astype(f.dtype)
        return uniq_rows, inverse
    void_view = f.view([("", f.dtype)] * n_cols).ravel()
    _, first_idx, inverse = np.unique(void_view, return_index=True, return_inverse=True)
    return f[first_idx], inverse


def _predict_pixels(model, features):
    """ピクセル特徴量をモデルで分類する。整数特徴量ならユニーク色で予測回数を削減する"""
    if features.dtype.kind in "ui" and features.shape[0] > features.shape[1]:
        uniq_rows, inverse = _unique_rows(features)
    else:
        uniq_rows, inverse = features, None
    try:
        pred = model.predict(uniq_rows.astype(np.float32))
    except Exception:
        pred = model.predict(uniq_rows)
    # OpenCV の場合は (retval, results) が返る
    if isinstance(pred, tuple) and len(pred) == 2:
        _, results = pred
        predictions = np.asarray(results).ravel()
    else:
        predictions = np.asarray(pred).ravel()
    if inverse is not None:
        predictions = predictions[inverse]
    return predictions


def get_pixel_features(img_bgr, mode, include_exg):
    """指定されたモードに応じてピクセルの特徴量を取得する（旧API: Lab / HSV / All + ExG）"""
    if mode == "Lab":
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        feats = img_lab.reshape(-1, 3)
    elif mode == "HSV":
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        feats = img_hsv.reshape(-1, 3)
    else:  # All (RGB + Lab + HSV)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, w = img_rgb.shape[:2]
        feats = np.empty((h * w, 9), dtype=img_rgb.dtype)
        feats[:, 0:3] = img_rgb.reshape(-1, 3)
        feats[:, 3:6] = img_lab.reshape(-1, 3)
        feats[:, 6:9] = img_hsv.reshape(-1, 3)

    if include_exg:
        b_channel, g_channel, r_channel = cv2.split(img_bgr)
        exg = (2 * g_channel.astype(np.int32)) - r_channel.astype(np.int32) - b_channel.astype(np.int32)
        exg_uint8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        exg_col = exg_uint8.reshape(-1, 1)
        feats = np.concatenate([feats, exg_col], axis=1)

    return feats
