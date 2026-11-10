# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""並列処理ワーカー。

multiprocessing.Pool の initializer でモデルをプロセスごとに一度だけ受け取り、
1枚ずつ画像を処理する。関数はモジュールトップレベルに置くこと（spawn 方式の
Windows では pickle 経由で参照されるため）。
"""

import multiprocessing
import os

import cv2
import numpy as np
from PIL import Image

from adaptfvc.core import (
    calculate_coverage,
    compose_result_image,
    create_overlay_from_mask,
    load_image,
    segment_image,
    segment_image_with_method,
    valid_fraction,
)


def prepare_resized_task(args):
    """1枚の画像の縮小キャッシュを作る（フォルダ読み込み時の並列処理用）。

    アルファを持つ画像は RGBA のまま縮小して保存し、透明（ROI 外）の情報を失わない。

    Returns:
        (src, dst, error_message)  成功時は error_message=None
    """
    src, dst, long_side = args
    try:
        with Image.open(src) as img:
            w, h = img.size
            scale = long_side / float(max(w, h))
            has_alpha = img.mode in ("RGBA", "LA", "PA", "RGBa") or (
                img.mode == "P" and "transparency" in img.info)
            out = img.convert("RGBA" if has_alpha else "RGB").resize(
                (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
            out.save(dst, format="PNG")
        return (src, dst, None)
    except Exception as e:
        return (src, dst, str(e))


# --- Global variables for multiprocessing ---
_global_model = None
_global_mode = "Lab"
_global_segmentation_method = "svm"
_global_include_exg = False


def _init_worker(model, mode="Lab", segmentation_method="svm", include_exg=False):
    """ワーカープロセスの初期化関数。モデルをグローバル変数として保持する。
    mode / segmentation_method / include_exg は旧 process_image_task（研究用）だけが使う"""
    global _global_model, _global_mode, _global_segmentation_method, _global_include_exg
    # ワーカー内の predict は 1 スレッドで十分（joblib がデーモンプロセス内で並列化できず
    # 毎回 UserWarning を出すため、ワーカー側のコピーだけ n_jobs=1 にする。結果は不変）
    if model is not None and hasattr(model, "n_jobs") and multiprocessing.current_process().daemon:
        try:
            model.n_jobs = 1
        except Exception:
            pass
    _global_model = model
    _global_mode = mode
    _global_segmentation_method = segmentation_method
    _global_include_exg = include_exg


def process_image_v3(image_path, output_folder):
    """1枚の画像を処理して判定画像を保存する（アプリ本体用・アルファ対応）。

    Returns:
        (image_path, coverage_rate, error_message, valid_fraction_percent)
        成功時: error_message=None。有効画素がゼロの画像は coverage=None でエラー扱い
    """
    try:
        img_bgr, valid = load_image(image_path)
        mask = segment_image(img_bgr, _global_model, valid)
        coverage_rate = calculate_coverage(mask, valid)
        vf = valid_fraction(valid)
        if coverage_rate is None:
            return (image_path, None, f"エラー ({os.path.basename(image_path)}): 有効画素がありません（全面が透明）", vf)
        result_img = compose_result_image(img_bgr, mask, valid)
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        result_img.save(output_path)
        return (image_path, coverage_rate, None, vf)
    except Exception as e:
        return (image_path, None, f"エラー ({os.path.basename(image_path)}): {e}", 0.0)


def process_image_task(image_path, output_folder):
    """旧API（研究スクリプト用）: mode / 手法 / ExG をグローバルから受けて処理する。

    Returns:
        (image_path, coverage_rate, error_message)
    """
    try:
        original_img_pil = Image.open(image_path).convert("RGBA")
        original_img_np = np.array(original_img_pil)
        img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGBA2BGR)
        mask, _ = segment_image_with_method(
            img_bgr, _global_mode, _global_model, _global_segmentation_method, _global_include_exg)
        coverage_rate = calculate_coverage(mask)
        overlay_img = create_overlay_from_mask(mask)
        result_img = Image.alpha_composite(original_img_pil, overlay_img)
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        result_img.convert("RGB").save(output_path)
        return (image_path, coverage_rate, None)
    except Exception as e:
        return (image_path, None, f"エラー ({os.path.basename(image_path)}): {e}")
