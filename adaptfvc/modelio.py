# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""モデルと教師点の保存・読込。

設計:
- 正は「教師点」。教師点は人が読める JSON（画像ファイル名 → 正規化座標の点列）で保存する。
  Random Forest は seed 固定で決定的なので、同じ教師点と同じ版からは常に同じモデルが再生成できる。
- モデル pickle は「バンドル」。model に加えて教師点・特徴量・学習器設定・環境の版を同梱し、
  読込時に照合する。版が合わなければ同梱の教師点から学習し直す
- v2 までの pickle（{'model','annotations'}。SVM の場合あり）は教師点だけ取り込む
"""

import json
import os
import pickle
import platform
import sys
from datetime import datetime

from adaptfvc import __version__
from adaptfvc.core import ALGO, FEATURE_MODE, RANDOM_STATE, RF_N_ESTIMATORS, IMAGE_EXTENSIONS

MODEL_FORMAT = "adaptfvc-model"
MODEL_FORMAT_VERSION = 3
ANNOTATIONS_FORMAT = "adaptfvc-annotations"
ANNOTATIONS_FORMAT_VERSION = 3


def environment_info():
    """再現性のために記録する実行環境"""
    import numpy as _np
    import cv2 as _cv2
    try:
        import sklearn as _sk
        sk_ver = getattr(_sk, "__version__", "N/A")
    except Exception:
        sk_ver = "N/A"
    return {
        "app_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": getattr(_np, "__version__", "N/A"),
        "opencv": getattr(_cv2, "__version__", "N/A"),
        "sklearn": sk_ver,
    }


def training_config():
    return {
        "feature_mode": FEATURE_MODE,
        "algo": ALGO,
        "n_estimators": RF_N_ESTIMATORS,
        "random_state": RANDOM_STATE,
    }


def original_name(path):
    """作業パス（縮小キャッシュ内なら 'IMG.JPG.png' など）から、元画像のファイル名を返す。
    教師点の保存キーに使う。フォルダや解像度が変わっても同じ画像と対応付けられる"""
    base = os.path.basename(path)
    low = base.lower()
    if low.endswith(".png"):
        stem = base[:-4]
        if stem.lower().endswith(IMAGE_EXTENSIONS):
            return stem
    return base


def annotations_to_portable(annotations):
    """{作業パス: 点列} → {元ファイル名: 点列}（座標とクラスだけに正規化）。

    座標は丸めない。学習に使う画素は int(nx * 幅) で決まるので、6桁に丸めただけでも
    境界で隣の画素に変わり、JSON から学習し直したモデルが元と一致しなくなる。
    Python の json は float を repr で書くため、丸めなければビット単位で往復する。

    実際にクリックした点には "click": true を付ける（v3.2 から。直線補間で足した点には付けない）。
    学習には使わず、クリック数の表示と、読み込み後の「クリック単位の取り消し」に使う。
    """
    out = {}
    for path, pts in annotations.items():
        if not pts:
            continue
        key = original_name(path)
        out.setdefault(key, [])
        for a in pts:
            p = {"nx": float(a["nx"]), "ny": float(a["ny"]),
                 "class": "plant" if a.get("class") == "plant" else "soil"}
            if a.get("click"):
                p["click"] = True
            out[key].append(p)
    return out


def portable_to_annotations(portable, image_paths):
    """{元ファイル名: 点列} → {作業パス: 点列}。現在の作業セットに無い画像の点は捨てる。

    Returns:
        (annotations, matched_names, unmatched_names)
    """
    by_name = {original_name(p): p for p in image_paths}
    annotations, matched, unmatched = {}, [], []
    for name, pts in portable.items():
        p = by_name.get(name)
        if p is None:
            unmatched.append(name)
            continue
        matched.append(name)
        annotations[p] = []
        for a in pts:
            q = {"nx": float(a["nx"]), "ny": float(a["ny"]), "class": a.get("class", "soil"), "size": 1}
            if a.get("click"):
                q["click"] = True
            annotations[p].append(q)
    return annotations, matched, unmatched


def count_points(portable):
    n_plant = sum(1 for pts in portable.values() for a in pts if a.get("class") == "plant")
    n_soil = sum(1 for pts in portable.values() for a in pts if a.get("class") != "plant")
    return n_plant, n_soil


def count_clicks(annotations):
    """クラスごとのクリック数。v3.1 以前の教師点（click の印が無い画像）は1点を1クリックと数える"""
    n_plant = n_soil = 0
    for pts in annotations.values():
        marked = any(a.get("click") for a in pts)
        for a in pts:
            if a.get("click") or not marked:
                if a.get("class") == "plant":
                    n_plant += 1
                else:
                    n_soil += 1
    return n_plant, n_soil


def click_groups(pts):
    """点列をクリック単位の点数の列に戻す（取り消しに使う）。
    1回のクリックで入る点列は、最後の点に click の印が付く。印の無い古い教師点は1点ずつ"""
    if not any(a.get("click") for a in pts):
        return [1] * len(pts)
    groups, run = [], 0
    for a in pts:
        run += 1
        if a.get("click"):
            groups.append(run)
            run = 0
    groups.extend([1] * run)
    return groups


def save_annotations_json(path, portable, extra=None):
    data = {
        "format": ANNOTATIONS_FORMAT,
        "format_version": ANNOTATIONS_FORMAT_VERSION,
        "app_version": __version__,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        **(extra or {}),
        "annotations": portable,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


def load_annotations_json(path):
    """教師点 JSON を読む。v2.5 の run 内 annotations_<日時>.json（素の辞書）も受け付ける"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get("format") == ANNOTATIONS_FORMAT:
        meta = {k: v for k, v in data.items() if k != "annotations"}
        portable = data.get("annotations", {})
    else:
        meta = {}
        portable = data
    portable = {original_name(k): v for k, v in portable.items()}
    return portable, meta


def save_model_bundle(path, model, portable, extra=None):
    """学習済みモデルを、教師点・設定・環境ごと pickle で保存する"""
    bundle = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "environment": environment_info(),
        "training": training_config(),
        **(extra or {}),
        "annotations": portable,
        "model": model,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(bundle, f)
    os.replace(tmp, path)
    return path


def load_model_file(path):
    """モデル/教師点ファイルを読み、統一した辞書で返す。

    Returns dict:
        kind: "bundle"（v3）| "legacy"（v2 の {'model','annotations'}）| "annotations"（JSON）| "model_only"
        model: 学習器 or None
        annotations: {元ファイル名: 点列}
        meta: 付随情報
        legacy_algo: v2 pickle の学習器名（参考）
    """
    if path.lower().endswith(".json"):
        portable, meta = load_annotations_json(path)
        return {"kind": "annotations", "model": None, "annotations": portable, "meta": meta}
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and obj.get("format") == MODEL_FORMAT:
        meta = {k: v for k, v in obj.items() if k not in ("model", "annotations")}
        return {"kind": "bundle", "model": obj.get("model"),
                "annotations": obj.get("annotations", {}), "meta": meta}
    if isinstance(obj, dict) and "model" in obj:
        raw = obj.get("annotations", {}) or {}
        portable = {}
        for k, pts in raw.items():
            name = original_name(k)
            portable.setdefault(name, [])
            for a in pts:
                portable[name].append({"nx": float(a["nx"]), "ny": float(a["ny"]),
                                       "class": "plant" if a.get("class") == "plant" else "soil"})
        model = obj.get("model")
        return {"kind": "legacy", "model": model, "annotations": portable,
                "meta": {"learning_mode": obj.get("learning_mode"), "timestamp": obj.get("timestamp")},
                "legacy_algo": type(model).__name__ if model is not None else None}
    if hasattr(obj, "predict"):
        return {"kind": "model_only", "model": obj, "annotations": {}, "meta": {}}
    raise ValueError("adaptFVC のモデル/教師点ファイルではありません。")


def model_is_usable(loaded):
    """読み込んだモデルをそのまま推論に使ってよいか。

    使えるのは v3 バンドルで、特徴量 Lab・学習器 RF・scikit-learn の版が今と同じもの。
    それ以外（v2 の SVM/RF、版違い）は教師点から学習し直す。
    Returns: (ok, reason)
    """
    model = loaded.get("model")
    if model is None or not hasattr(model, "predict"):
        return False, "モデル本体がありません"
    if loaded.get("kind") != "bundle":
        return False, "v2 以前の形式です（教師点から学習し直します）"
    meta = loaded.get("meta", {})
    tr = meta.get("training", {})
    if tr.get("feature_mode") != FEATURE_MODE or tr.get("algo") != ALGO:
        return False, f"学習設定が違います（{tr.get('feature_mode')}/{tr.get('algo')}）"
    env = meta.get("environment", {})
    cur = environment_info()
    if env.get("sklearn") != cur["sklearn"]:
        return False, f"scikit-learn の版が違います（保存時 {env.get('sklearn')} / 現在 {cur['sklearn']}）"
    n_feat = getattr(model, "n_features_in_", 3)
    if n_feat != 3:
        return False, f"特徴量の数が違います（{n_feat}）"
    return True, ""
