# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""画像フォルダの走査と縮小キャッシュ、出力フォルダ構成。GUI と CLI の両方が使う。

出力はすべて <画像フォルダ>/adaptFVC/ 配下に置き、原本には一切触れない:
    cache/      縮小版 (resized_<長辺>/) と resize_choice.json
    results/    全画像の処理の結果一式 (run_<日時>/。v3.1 以前は runs/)
    previews/   プレビュー
    models/     自動保存のモデルと教師点
    logs/       作業ログ (session_*.jsonl)
"""

import glob
import json
import multiprocessing
import os
import shutil
import time

from PIL import Image

from adaptfvc.core import IMAGE_EXTENSIONS
from adaptfvc.workers import prepare_resized_task

APP_DIR_NAME = "adaptFVC"
CACHE_SIZE_CHOICES = (256, 512, 1024)
CACHE_SIZE_DEFAULT = 1024


def app_dir(image_folder, *parts, make=False):
    p = os.path.join(image_folder, APP_DIR_NAME, *parts)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def discover_images(image_folder):
    """フォルダ直下の画像を拡張子ごとにソートして返す（大文字小文字は区別しない）"""
    found = []
    for name in os.listdir(image_folder):
        if name.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(image_folder, name)):
            found.append(os.path.join(image_folder, name))
    # 旧版と同じ並び: .jpg → .png の順にそれぞれソート。それ以外の拡張子は末尾
    def key(p):
        ext = os.path.splitext(p)[1].lower()
        rank = {".jpg": 0, ".jpeg": 0, ".png": 1}.get(ext, 2)
        return (rank, p)
    return sorted(found, key=key)


def migrate_legacy_outputs(image_folder):
    """旧配置 (results/ 直下) の縮小キャッシュと選択を adaptFVC/cache/ へ移す（rename のみ）"""
    old_root = os.path.join(image_folder, "results")
    if not os.path.isdir(old_root):
        return
    try:
        cache = app_dir(image_folder, "cache", make=True)
        oldc = os.path.join(old_root, "resize_choice.json")
        newc = os.path.join(cache, "resize_choice.json")
        if os.path.exists(oldc) and not os.path.exists(newc):
            shutil.move(oldc, newc)
        for d in glob.glob(os.path.join(old_root, "resized_*")):
            nd = os.path.join(cache, os.path.basename(d))
            if os.path.isdir(d) and not os.path.exists(nd):
                shutil.move(d, nd)
    except Exception as e:
        print(f"旧キャッシュの移行に失敗（無視して続行）: {e}")


def load_resize_choice(image_folder):
    cfg = app_dir(image_folder, "cache", "resize_choice.json")
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            v = int(json.load(f).get("long_side"))
        if v in CACHE_SIZE_CHOICES:
            return v
    except Exception:
        pass
    return None


def save_resize_choice(image_folder, v):
    cfg = app_dir(image_folder, "cache", "resize_choice.json")
    os.makedirs(os.path.dirname(cfg), exist_ok=True)
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"long_side": int(v)}, f)


def clear_resize_choice(image_folder):
    cfg = app_dir(image_folder, "cache", "resize_choice.json")
    if os.path.exists(cfg):
        os.remove(cfg)


def scan_dimensions(image_paths):
    """(path, 長辺px) のリスト。開けない画像は除く"""
    dims = []
    for p in image_paths:
        try:
            with Image.open(p) as im:
                dims.append((p, max(im.size)))
        except Exception:
            continue
    return dims


def prepare_resized_cache(image_folder, image_paths, long_side, progress=None):
    """長辺が long_side を超える画像の縮小版を adaptFVC/cache/resized_<long_side>/ に用意し、
    差し替え後のパス一覧を返す。

    - 原本は読み取りのみ。縮小版は無損失PNG（アルファ付きなら RGBA のまま）
    - 2回目以降は mtime 比較で再生成をスキップ
    Returns:
        (new_paths, stats dict)
    """
    dims = scan_dimensions(image_paths)
    mapped, targets = {}, []
    cache_dir = app_dir(image_folder, "cache", f"resized_{long_side}")
    for p, d in dims:
        if d <= long_side:
            continue
        base = os.path.basename(p)
        if not base.lower().endswith(".png"):
            base += ".png"
        dst = os.path.join(cache_dir, base)
        mapped[p] = dst
        if not (os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(p)):
            targets.append((p, dst, long_side))
    stats = {"n_large": len(mapped), "n_created": 0, "n_errors": 0, "duration_s": 0.0,
             "long_side": long_side}
    if not mapped:
        return list(image_paths), stats
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    if targets:
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        done = 0
        with multiprocessing.Pool(processes=num_cores) as pool:
            for src, dst, err in pool.imap_unordered(prepare_resized_task, targets):
                done += 1
                if err is not None:
                    stats["n_errors"] += 1
                    mapped.pop(src, None)
                    print(f"縮小失敗（原寸で続行）: {os.path.basename(src)}: {err}")
                if progress:
                    progress(f"縮小版を準備中: {done}/{len(targets)}")
        if progress:
            progress("")
    stats["n_created"] = len(targets) - stats["n_errors"]
    stats["duration_s"] = round(time.time() - t0, 1)
    return [mapped.get(p, p) for p in image_paths], stats


def needs_resize_choice(image_paths):
    """縮小サイズの選択が必要か（最小選択肢より大きい画像があるか）。
    Returns: (n_over_min, max_dim)"""
    dims = scan_dimensions(image_paths)
    min_choice = min(CACHE_SIZE_CHOICES)
    n_over = sum(1 for _, d in dims if d > min_choice)
    max_dim = max((d for _, d in dims), default=0)
    return n_over, max_dim
