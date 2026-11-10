# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""挙動等価の検証ハーネス（v3）。

決定論的な合成画像（RGB 12枚 + アルファ付き 2枚）に対して計算コアの成果物を記録し、
リファクタリング後も同じ値が出ることを確認する。比較するのは成果物そのもの
（マスクの SHA-1・植被率・特徴量配列・ワーカー出力ファイル名・バンドルの往復）。

使い方:
    python tests/equivalence_harness.py record   # 現在のパッケージの出力を tests/baseline.json に記録
    python tests/equivalence_harness.py check    # 現在の出力と baseline.json を突き合わせ

baseline に無い新キーは「比較対象外」として表示する。baseline にあって今は計算しないキーは
REMOVED_KEYS（意図して廃止したもの）に含まれていれば無視し、含まれていなければ失敗にする。
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
BASELINE = os.path.join(HERE, "baseline.json")

# v3 で意図的に廃止したキー（SVM 経路と SVM を使うワーカー）
REMOVED_KEYS = {"svm_mask", "svm_cov", "worker", "worker_outputs"}


def make_images(tmpdir):
    """決定論的な合成画像12枚（緑の円 + 固定シードのノイズ）を書き出す"""
    rng = np.random.RandomState(0)
    paths = []
    for i in range(12):
        h, w = 96, 128
        img = np.zeros((h, w, 3), np.uint8)
        img[..., 0] = 40 + i * 5
        img[..., 1] = 60 + (i * 13) % 100
        img[..., 2] = 50
        cv2.circle(img, (w // 2, h // 2), 10 + i, (30, 200, 40), -1)
        img = cv2.add(img, rng.randint(0, 20, (h, w, 3)).astype(np.uint8))
        p = os.path.join(tmpdir, f"img_{i:02d}.png")
        cv2.imwrite(p, img)
        paths.append(p)
    return paths


def make_alpha_images(tmpdir):
    """区画が斜めに写った切り出し画像（四隅が透明）を2枚。1枚は TIFF"""
    rng = np.random.RandomState(1)
    paths = []
    for i, ext in enumerate(("png", "tif")):
        S = 96
        yy, xx = np.mgrid[0:S, 0:S]
        rgb = np.zeros((S, S, 3), np.uint8)
        rgb[..., 0] = 120 + rng.randint(-10, 10, (S, S))
        rgb[..., 1] = 90 + rng.randint(-10, 10, (S, S))
        rgb[..., 2] = 60 + rng.randint(-10, 10, (S, S))
        veg = (xx - 48) ** 2 + (yy - 48) ** 2 < (18 + 6 * i) ** 2
        rgb[veg] = (60, 160, 50)
        roi = (np.abs(xx - 48) + np.abs(yy - 48)) < 44          # ひし形の区画
        rgba = np.dstack([rgb, np.where(roi, 255, 0).astype(np.uint8)])
        rgba[~roi, :3] = 0
        p = os.path.join(tmpdir, f"alpha_{i}.{ext}")
        Image.fromarray(rgba).save(p)
        paths.append(p)
    return paths


def arr_sig(a):
    a = np.asarray(a)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest(),
        "sum": float(np.asarray(a, np.float64).sum()),
    }


def run():
    from adaptfvc import core, modelio
    from adaptfvc import workers as wk
    from adaptfvc.gui import AnnotationApp as App

    out = {}
    tmp = tempfile.mkdtemp(prefix="adaptfvc_eq_")
    try:
        paths = make_images(tmp)
        img = cv2.imread(paths[0])

        # --- 旧来のキー（v2 と同じ値になるべきもの） ---
        mask, thr = core.calculate_exg_mask(img)
        out["exg_mask"] = arr_sig(mask)
        out["exg_thr"] = float(thr)
        for mode in ["Lab", "HSV", "All"]:
            for exg in [False, True]:
                out[f"feat_{mode}_{exg}"] = arr_sig(core.get_pixel_features(img, mode, exg))
        out["coverage_exg"] = float(core.calculate_coverage(mask))
        out["overlay"] = arr_sig(np.array(core.create_overlay_from_mask(mask)))
        out["overlay_col"] = arr_sig(np.array(core.create_overlay_from_mask_colored(mask, (0, 255, 0, 120))))

        from sklearn.ensemble import RandomForestClassifier
        feats = core.get_pixel_features(img, "Lab", False)
        labels = (feats[:, 1].astype(int) < 128).astype(int)
        idx = np.linspace(0, len(feats) - 1, 400, dtype=int)
        rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(feats[idx], labels[idx])
        m_rf, _ = core.segment_image_with_method(img, "Lab", rf, "svm", False)
        out["rf_mask"] = arr_sig(m_rf)
        out["rf_cov"] = float(core.calculate_coverage(m_rf))
        m_exg2, info = core.segment_image_with_method(img, "Lab", None, "exg", False)
        out["exg2_mask"] = arr_sig(m_exg2)
        out["exg2_thr"] = float(info["otsu_threshold"])
        out["rep_count"] = [App.compute_representative_count(None, n) for n in [5, 10, 11, 100, 1000, 5000]]
        sel = App.select_representative_images(None, paths, 4)
        out["rep_sel"] = [os.path.basename(p) for p in sel]

        # --- v3 の新キー ---
        # 新API predict_mask は旧API(Lab, RF) と同一のマスクを返す
        out["v3_predict_mask_equals_old"] = bool(np.array_equal(core.predict_mask(img, rf), m_rf))

        # 教師点 → 学習行列（順序固定）→ RF → マスク
        ann = {}
        for p in paths[:3]:
            ann[p] = [{"nx": 0.5, "ny": 0.5, "class": "plant", "size": 1},
                      {"nx": 0.52, "ny": 0.5, "class": "plant", "size": 1},
                      {"nx": 0.1, "ny": 0.1, "class": "soil", "size": 1},
                      {"nx": 0.9, "ny": 0.9, "class": "soil", "size": 1}]
            for k in range(8):
                ann[p].append({"nx": 0.45 + 0.01 * k, "ny": 0.5, "class": "plant", "size": 1})
                ann[p].append({"nx": 0.05 + 0.1 * k, "ny": 0.05, "class": "soil", "size": 1})
        X, y, st = core.extract_training_pixels(ann)
        out["train_X"] = arr_sig(X)
        out["train_y"] = arr_sig(y)
        out["train_stats"] = st
        model = core.train_rf(X, y)
        m_v3 = core.predict_mask(img, model)
        out["rf_v3_mask"] = arr_sig(m_v3)
        out["rf_v3_cov"] = float(core.calculate_coverage(m_v3))
        # 同じ教師点からもう一度学習しても同じマスク（決定性）
        out["rf_v3_deterministic"] = bool(np.array_equal(core.predict_mask(img, core.train_rf(X, y)), m_v3))

        # アルファ付き画像: 透明画素は分母にも学習にも入らない
        apaths = make_alpha_images(tmp)
        for ap in apaths:
            key = os.path.splitext(os.path.basename(ap))[0] + "_" + os.path.splitext(ap)[1][1:]
            bgr, valid = core.load_image(ap)
            out[f"alpha_{key}_valid"] = arr_sig(valid)
            amask = core.segment_image(bgr, model, valid)
            out[f"alpha_{key}_cov"] = float(core.calculate_coverage(amask, valid))
            out[f"alpha_{key}_cov_no_valid"] = float(core.calculate_coverage(core.predict_mask(bgr, model)))
            out[f"alpha_{key}_vf"] = round(core.valid_fraction(valid), 4)
            out[f"alpha_{key}_result_img"] = arr_sig(np.array(core.compose_result_image(bgr, amask, valid)))
        # 透明画素に打った教師点は除外される
        ann_a = {apaths[0]: [{"nx": 0.02, "ny": 0.02, "class": "soil", "size": 1},
                             {"nx": 0.5, "ny": 0.5, "class": "plant", "size": 1}]}
        _, _, st_a = core.extract_training_pixels(ann_a)
        out["alpha_skipped_points"] = st_a["n_skipped_invalid"]

        # ワーカー（v3）: RGB と RGBA の両方
        outdir = os.path.join(tmp, "out")
        os.makedirs(outdir, exist_ok=True)
        wk._init_worker(model)
        res = [wk.process_image_v3(p, outdir) for p in paths[:3] + apaths]
        out["worker_v3"] = [[os.path.basename(p), None if c is None else round(float(c), 6), e, round(float(vf), 4)]
                            for p, c, e, vf in res]
        out["worker_v3_outputs"] = sorted(os.listdir(outdir))
        # 旧ワーカー（研究用）も RF で動く
        wk._init_worker(rf, "Lab", "svm", False)
        res_old = wk.process_image_task(paths[0], outdir)
        out["worker_old_cov"] = round(float(res_old[1]), 6)

        # 縮小キャッシュはアルファを保持する
        dst = os.path.join(tmp, "resized.png")
        wk.prepare_resized_task((apaths[0], dst, 48))
        with Image.open(dst) as im:
            out["resize_alpha_mode"] = im.mode
            out["resize_alpha_size"] = list(im.size)

        # バンドル往復
        portable = modelio.annotations_to_portable(ann)
        bpath = os.path.join(tmp, "model.pkl")
        modelio.save_model_bundle(bpath, model, portable, {"image_folder": tmp})
        loaded = modelio.load_model_file(bpath)
        ok, reason = modelio.model_is_usable(loaded)
        out["bundle_usable"] = bool(ok)
        out["bundle_annotations_roundtrip"] = loaded["annotations"] == portable
        out["bundle_predict_equal"] = bool(np.array_equal(core.predict_mask(img, loaded["model"]), m_v3))
        back, matched, unmatched = modelio.portable_to_annotations(portable, paths)
        out["portable_remap"] = [len(matched), len(unmatched)]
        # v2 形式 pickle の取り込み
        import pickle
        legacy = {"model": rf, "annotations": {p: ann[p] for p in paths[:2]}, "learning_mode": "Lab"}
        lpath = os.path.join(tmp, "legacy.pkl")
        with open(lpath, "wb") as f:
            pickle.dump(legacy, f)
        lloaded = modelio.load_model_file(lpath)
        out["legacy_kind"] = lloaded["kind"]
        out["legacy_usable"] = bool(modelio.model_is_usable(lloaded)[0])
        out["legacy_points"] = list(modelio.count_points(lloaded["annotations"]))
        # 縮小キャッシュ名 'IMG.JPG.png' → 'IMG.JPG'
        out["original_name"] = [modelio.original_name(r"C:\x\adaptFVC\cache\resized_1024\IMG_0001.JPG.png"),
                                modelio.original_name("plot_01.png"), modelio.original_name("a.tif")]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    data = run()
    if cmd == "record":
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)
        print(f"baseline recorded: {len(data)} keys -> {BASELINE}")
        return 0
    with open(BASELINE, "r", encoding="utf-8") as f:
        base = json.load(f)
    bad, new, removed = [], [], []
    for k in sorted(set(base) | set(data)):
        if k not in base:
            new.append(k)
        elif k not in data:
            removed.append(k)
            if k not in REMOVED_KEYS:
                bad.append(k)
                print(f"MISSING {k} (baseline has it, not computed now)")
        elif base[k] != data[k]:
            bad.append(k)
            print(f"MISMATCH {k}:\n  base={base[k]}\n  now ={data[k]}")
    if new:
        print(f"new keys (not in baseline, not compared): {len(new)}")
    if removed:
        print(f"removed keys (intentionally): {sorted(set(removed) & REMOVED_KEYS)}")
    n_cmp = len(set(base) & set(data))
    if bad:
        print(f"FAILED: {len(bad)} problems ({n_cmp} keys compared)")
        return 1
    print(f"OK: {n_cmp} compared keys identical; {len(new)} new keys")
    return 0


if __name__ == "__main__":
    sys.exit(main())
