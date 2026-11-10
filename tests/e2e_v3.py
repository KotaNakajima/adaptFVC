# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""v3 E2E: 実 Tk ウィンドウで、フォルダ読込 → 教示 → 学習(T) → 自動保存 → 推論表示(Space) →
自動推論 → 連打 → 取り消し → 再学習 → モデル読込（v3/v2） → プレビュー → 全画像の処理 →
再起動時の復元 → アルファ付きフォルダ（透明画素の除外・TIFF）→ CLI バッチ を検証する。
ダイアログはすべてモンキーパッチで自動応答する。test_smoke/ の画像を使う。

使い方: python tests/e2e_v3.py
"""
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SMOKE = os.path.join(ROOT, "test_smoke")
PY = sys.executable
sys.stdout.reconfigure(errors="replace")  # 絵文字を含む結果表示で cp932 コンソールが落ちないように

results = []
# 子プロセス（CLI バッチ）の出力は UTF-8 に固定して読む（親が UTF-8 モードでも cp932 でも同じに動くように）
CHILD_IO = dict(text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def check(name, ok, info=""):
    results.append((name, bool(ok), info))
    print(("PASS " if ok else "FAIL ") + name + ("  " + str(info) if info != "" else ""))


dialogs = []
answers = {"askyesno": True}


def _info(title, msg, **k): dialogs.append(("info", title, msg)); return "ok"
def _warn(title, msg, **k): dialogs.append(("warn", title, msg)); return "ok"
def _err(title, msg, **k):  dialogs.append(("error", title, msg)); return "ok"
def _yes(title, msg, **k):  dialogs.append(("yesno", title, msg)); return answers["askyesno"]


def pump(root, seconds):
    t_end = time.time() + seconds
    while time.time() < t_end:
        root.update()
        time.sleep(0.02)


def wait_idle(app, root, timeout=60):
    t_end = time.time() + timeout
    while time.time() < t_end:
        root.update()
        if app._infer_running is None and app._infer_pending is None:
            return True
        time.sleep(0.02)
    return False


def synth_points(app, paths, n=40):
    for p in paths:
        img = cv2.imread(p)
        b, g, r = cv2.split(img.astype(np.int32))
        exg = (2 * g - r - b)
        h, w = exg.shape
        order = np.argsort(exg.ravel())
        pts = []
        for idx in order[-n:]:
            y, x = divmod(int(idx), w)
            pts.append({'nx': x / w, 'ny': y / h, 'class': 'plant', 'size': 1})
        for idx in order[:n]:
            y, x = divmod(int(idx), w)
            pts.append({'nx': x / w, 'ny': y / h, 'class': 'soil', 'size': 1})
        app.annotations[p] = pts
        app.annotation_groups[p] = [1] * len(pts)


def make_alpha_folder():
    """区画が斜めに写った RGBA 切り出し画像 16 枚（PNG 15 + TIFF 1）のフォルダを作る（>12 枚で Pool 経路を通す）"""
    d = tempfile.mkdtemp(prefix="adaptfvc_alpha_")
    rng = np.random.RandomState(0)
    S = 144
    yy, xx = np.mgrid[0:S, 0:S]
    for i in range(16):
        rgb = np.zeros((S, S, 3), np.uint8)
        rgb[..., 0] = 120 + rng.randint(-15, 15, (S, S))
        rgb[..., 1] = 90 + rng.randint(-15, 15, (S, S))
        rgb[..., 2] = 60 + rng.randint(-15, 15, (S, S))
        veg = np.zeros((S, S), bool)
        for _ in range(20 + 3 * i):
            cx, cy, r = rng.randint(0, S), rng.randint(0, S), rng.randint(5, 12)
            veg |= (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        plant = np.stack([60 + rng.randint(-20, 20, (S, S)), 150 + rng.randint(-20, 20, (S, S)),
                          50 + rng.randint(-20, 20, (S, S))], -1).clip(0, 255).astype(np.uint8)
        rgb[veg] = plant[veg]
        roi = (np.abs(xx - 72) + np.abs(yy - 72)) < 70      # ひし形 = 回転した正方形
        rgba = np.dstack([rgb, np.where(roi, 255, 0).astype(np.uint8)])
        rgba[~roi, :3] = 0
        ext = "tif" if i == 15 else "png"
        Image.fromarray(rgba).save(os.path.join(d, f"plot_{i:02d}.{ext}"))
    return d


def main():
    filedialog.askdirectory = lambda **k: SMOKE
    messagebox.showinfo, messagebox.showwarning = _info, _warn
    messagebox.showerror, messagebox.askyesno = _err, _yes
    from adaptfvc import __version__, imagecache
    from adaptfvc.gui import AnnotationApp
    from adaptfvc.modelio import load_model_file, model_is_usable
    AnnotationApp._ask_resize_size = lambda self, n, m: 256

    # 前回のテスト出力を消して、クリーンな状態から
    shutil.rmtree(os.path.join(SMOKE, "adaptFVC"), ignore_errors=True)
    smoke_max_dim = max(Image.open(p).size[0] for p in glob.glob(os.path.join(SMOKE, "*.JPG")))

    root = tk.Tk()
    app = AnnotationApp(root)
    pump(root, 0.4)
    check("version 3.x", __version__.startswith("3."), __version__)
    check("images loaded", len(app.image_paths) == 10, len(app.image_paths))
    if smoke_max_dim > 256:
        check("cache used (256)", app.cache_long_side == 256 and "resized_256" in app.image_paths[0], app.image_paths[0])
    else:
        check("no cache needed (small images)", app.cache_long_side is None, smoke_max_dim)
    check("no restore prompt on fresh folder", not any(d[0] == "yesno" for d in dialogs))
    check("menu exists", isinstance(app.root.nametowidget(app.root.cget("menu")), tk.Menu))

    def walk(w, acc):
        if isinstance(w, (tk.Button, tk.Checkbutton, tk.Radiobutton)):
            acc.append(str(w.cget("takefocus")))
        for c in w.winfo_children():
            walk(c, acc)
    acc = []
    walk(root, acc)
    check("buttons takefocus=0", acc and all(v == "0" for v in acc), f"{len(acc)} widgets")

    # 教師点が足りないときの学習は警告
    n_warn = len([d for d in dialogs if d[0] == "warn"])
    app.train_model_ui(source="key")
    check("train without points -> warning", len([d for d in dialogs if d[0] == "warn"]) == n_warn + 1 and app.model is None)

    # クリック相当で教師点を打つ → 自動保存される
    class E:
        pass
    e = E()
    e.x, e.y = app.display_w // 2, app.display_h // 2
    app.on_image_click(e)
    e.x += 20
    app.on_image_click(e)
    check("click adds points with interpolation", len(app.annotations[app.current_image_path]) > 2,
          len(app.annotations[app.current_image_path]))
    check("seq counter", app.seq_progress_label.cget("text").startswith("連続注釈: 2/5"), app.seq_progress_label.cget("text"))
    n_px = len(app.annotations[app.current_image_path])
    check("count label shows clicks and pixels",
          app.plant_count_label.cget("text") == f"🌱 植物: 2 クリック（{n_px} 画素）"
          or app.soil_count_label.cget("text") == f"🟫 土壌: 2 クリック（{n_px} 画素）",
          app.plant_count_label.cget("text") + " / " + app.soil_count_label.cget("text"))
    from adaptfvc.modelio import annotations_to_portable, click_groups, portable_to_annotations
    port = annotations_to_portable({app.current_image_path: app.annotations[app.current_image_path]})
    back, _, _ = portable_to_annotations(port, [app.current_image_path])
    check("click marks survive portable round trip and rebuild undo groups",
          sum(1 for a in next(iter(port.values())) if a.get("click")) == 2
          and click_groups(back[app.current_image_path]) == app.annotation_groups[app.current_image_path],
          app.annotation_groups[app.current_image_path])
    pump(root, 1.2)
    ann_latest = os.path.join(SMOKE, "adaptFVC", "models", "annotations_latest.json")
    check("annotations autosaved after click", os.path.exists(ann_latest))
    app.undo_last_annotation(source="key")
    check("undo removes the whole click group", len(app.annotations[app.current_image_path]) == 1)
    app.undo_last_annotation(source="key")
    app.annotations, app.annotation_groups = {}, {}

    # 本物の教師点 → 学習 (T)
    synth_points(app, app.annotation_images[:2])
    app.update_annotation_counts()
    app.train_model_ui(source="key")
    check("trained", app.model is not None and app.model_info.get("source") == "train")
    check("show_inference auto ON", app.show_inference.get() is True)
    model_latest = os.path.join(SMOKE, "adaptFVC", "models", "model_latest.pkl")
    check("model autosaved", os.path.exists(model_latest) and app.model_info.get("saved") is True)
    check("annotation history saved", len(glob.glob(os.path.join(SMOKE, "adaptFVC", "models", "history", "*.json"))) >= 1)
    check("model status label", "学習" in app.model_status_label.cget("text"), app.model_status_label.cget("text"))
    ok = wait_idle(app, root)
    check("auto inference on current", ok and app.current_image_path in app.infer_cache and app.current_overlay_pil is not None)
    check("status shows coverage", "植被率" in app.infer_status_label.cget("text"), app.infer_status_label.cget("text"))

    # ナビ・連打・トグル
    app.navigate_next(source="key"); wait_idle(app, root)
    check("navigate auto-infers", app.current_image_path in app.infer_cache and app.current_overlay_pil is not None)
    before = set(app.infer_cache)
    for _ in range(3):
        app.navigate_next(source="key")
    wait_idle(app, root)
    check("rapid nav: last shown", app.current_image_path in app.infer_cache and app.current_overlay_pil is not None)
    check("rapid nav: skipped intermediates", len(set(app.infer_cache) - before) <= 2, len(set(app.infer_cache) - before))
    app.toggle_inference_display(source="key"); root.update()
    check("toggle OFF", app.current_overlay_pil is None and "OFF" in app.infer_status_label.cget("text"))
    app.toggle_inference_display(source="key"); root.update()
    check("toggle ON cached", app.current_overlay_pil is not None and app._infer_running is None)
    app.set_annotation_class("soil"); root.update()
    app.set_annotation_class("soil"); root.update()
    check("W fixes soil (idempotent)", app.annotation_class.get() == "soil"
          and "土壌" in app.canvas_class_banner.cget("text")
          and app.image_canvas.cget("highlightbackground") == "#7a5b40", app.canvas_class_banner.cget("text"))
    app.set_annotation_class("plant"); root.update()
    check("Q fixes plant + banner/frame color", app.annotation_class.get() == "plant"
          and "植物" in app.canvas_class_banner.cget("text")
          and app.image_canvas.cget("highlightbackground") == "#1e8a4a")
    app.toggle_seq_interp(); root.update()
    check("X toggles seq + label OFF", app.seq_interp_enabled.get() is False and "OFF" in app.seq_progress_label.cget("text"))
    app.toggle_seq_interp(); root.update()

    # 実キー
    root.deiconify(); root.lift(); app.image_canvas.focus_force(); root.update()
    st = app.show_inference.get(); root.event_generate("<space>"); pump(root, 0.3)
    check("real key Space", app.show_inference.get() != st)
    root.event_generate("<space>"); pump(root, 0.3)
    root.event_generate("<Key-w>"); pump(root, 0.2)
    ok_w = app.annotation_class.get() == "soil"
    root.event_generate("<Key-w>"); pump(root, 0.2)
    ok_w2 = app.annotation_class.get() == "soil"
    root.event_generate("<Key-q>"); pump(root, 0.2)
    check("real key w/w/q -> soil/soil/plant", ok_w and ok_w2 and app.annotation_class.get() == "plant")
    i0 = app.current_image_index; root.event_generate("<Key-d>"); wait_idle(app, root)
    check("real key d", app.current_image_index == (i0 + 1) % len(app.annotation_images))
    root.event_generate("<Key-a>"); wait_idle(app, root)
    check("real key a", app.current_image_index == i0)

    # 再学習 → キャッシュ破棄
    gen = app._model_generation
    app.train_model_ui(source="button"); wait_idle(app, root)
    check("retrain bumps generation & re-infers", app._model_generation > gen and app.current_image_path in app.infer_cache)

    # モデル読込（v3 バンドル: そのまま使える）
    loaded = load_model_file(model_latest)
    check("bundle usable", model_is_usable(loaded)[0] is True, model_is_usable(loaded)[1])
    filedialog.askopenfilename = lambda **k: model_latest
    answers["askyesno"] = True
    app.load_model_dialog(); wait_idle(app, root)
    check("load v3 bundle", app.model is not None and app.model_info.get("source") == "load")

    # v2 形式 pickle（SVM 相当）は教師点だけ取り込んで学習し直す
    legacy_path = os.path.join(tempfile.gettempdir(), "adaptfvc_legacy_v2.pkl")
    with open(legacy_path, "wb") as f:
        pickle.dump({"model": object(), "annotations": {p: app.annotations[p] for p in app.annotation_images[:2]},
                     "learning_mode": "Lab", "timestamp": "x"}, f)
    filedialog.askopenfilename = lambda **k: legacy_path
    app.load_model_dialog(); wait_idle(app, root)
    last_info = [d for d in dialogs if d[0] == "info"][-1][2]
    check("load v2 pickle -> retrained", app.model is not None and "学習し直しました" in last_info, last_info[:80])

    # プレビュー: 10枚以下のフォルダでは全画像が「教える画像」なので、3枚外してから
    removed = list(app.annotation_images[-3:])
    for p in removed:
        app.remove_from_annotation_images(p)
    root.update()
    # 「画像を追加」で戻した画像は一覧の一番下に合流する（既存の並びは不変）
    app.display_image(0)
    before = list(app.annotation_images)
    filedialog.askopenfilenames = lambda **k: [removed[1]]
    app.add_images_from_folder(); root.update()
    check("added image goes to the bottom", app.annotation_images[:-1] == before and app.annotation_images[-1] == removed[1])
    app.remove_from_annotation_images(removed[1]); root.update()
    # 追加先の順序は _append_annotation_images の規則（プレビューからの追加も同じ関数を使う）
    added = app._append_annotation_images([removed[2], removed[0]])
    check("append keeps folder order among new ones, after existing",
          app.annotation_images[-2:] == [removed[0], removed[2]] and app.annotation_images[:-2] == before, added)
    for p in (removed[0], removed[2]):
        app.remove_from_annotation_images(p)
    root.update()
    n_top = len([w for w in root.winfo_children() if isinstance(w, tk.Toplevel)])
    t0 = time.time()
    app.preview(); pump(root, 0.5)
    tops = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    n_prev = len(glob.glob(os.path.join(SMOKE, "adaptFVC", "previews", "preview_1", "*")))
    check("preview opens review window with 3 images", len(tops) == n_top + 1 and n_prev == 3, f"tops={len(tops)-n_top} files={n_prev}")
    check("preview is fast (sequential, <20s)", time.time() - t0 < 20, f"{time.time()-t0:.1f}s")
    pw = [w for w in tops if isinstance(w, tk.Toplevel)][-1]
    pw.update()
    sw, sh = pw.winfo_screenwidth(), pw.winfo_screenheight()
    check("preview window fits on screen",
          pw.winfo_y() >= 0 and pw.winfo_y() + pw.winfo_height() <= sh and pw.winfo_x() + pw.winfo_width() <= sw,
          (pw.winfo_x(), pw.winfo_y(), pw.winfo_width(), pw.winfo_height(), sw, sh))
    # 1枚選んで「教える画像に追加」→ サムネイル列の一番下に入り、列がそこまでスクロールする
    app.thumbnail_canvas.yview_moveto(0.0)
    chk = [w for w in pw.winfo_children()[1].winfo_children()[0].winfo_children()[0].winfo_children()]
    btns = [b for cell in chk for b in cell.winfo_children() if isinstance(b, tk.Checkbutton)]
    btns[0].invoke()
    add_btn = [b for b in pw.winfo_children()[0].winfo_children() if isinstance(b, tk.Button)][0]
    add_btn.invoke(); root.update()
    v0, v1 = app.thumbnail_canvas.yview()
    check("preview add -> appended at bottom & thumbnails scrolled down",
          app.annotation_images[-1] == app.current_image_path and v1 > 0.99, (round(v0, 2), round(v1, 2)))
    app.remove_from_annotation_images(app.annotation_images[-1])
    for w in [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]:
        w.destroy()
    root.update()

    # 全画像の処理（multiprocessing）
    app.process_all(); root.update()
    runs = sorted(glob.glob(os.path.join(SMOKE, "adaptFVC", "results", "run_*")))
    check("process_all run dir", len(runs) >= 1)
    if runs:
        rd = runs[-1]
        files = os.listdir(rd)
        check("run has xlsx/pkl/json/metadata/overlays",
              any(f.endswith(".xlsx") for f in files) and any(f.startswith("model_") for f in files)
              and any(f.startswith("annotations_") for f in files) and "run_metadata.json" in files
              and len(os.listdir(os.path.join(rd, "overlays"))) == 10, files)
        from openpyxl import load_workbook
        wb = load_workbook(glob.glob(os.path.join(rd, "*.xlsx"))[0])
        ws = wb["結果"]
        v = ws.cell(row=2, column=4).value
        check("xlsx coverage is numeric", isinstance(v, (int, float)), repr(v))
        check("xlsx valid fraction column", ws.cell(row=1, column=5).value == "有効画素率(%)" and ws.cell(row=2, column=5).value == 100)
        name = ws.cell(row=2, column=2).value
        check("xlsx file name is the original (not the cache name)", name.endswith(".JPG") and not name.endswith(".png"), name)
        meta = json.load(open(os.path.join(rd, "run_metadata.json"), encoding="utf-8"))
        check("metadata training config", meta["training"]["algo"] == "RF" and meta["training"]["feature_mode"] == "Lab")

    # 学習のあとに教師点を足して全画像を処理すると、学習し直すかを聞く。
    # 結果フォルダの教師点 JSON と、モデルに同梱された教師点と、いまの教師点が一致する（JSON から同じモデルを作り直せる）
    from adaptfvc.modelio import annotations_to_portable, load_model_file
    p0 = app.annotation_images[0]
    app.annotations[p0].append({'nx': 0.5, 'ny': 0.5, 'class': 'soil', 'size': 1, 'click': True})
    app.annotation_groups.setdefault(p0, []).append(1)
    n_yes = len([d for d in dialogs if d[0] == "yesno"])
    answers["askyesno"] = True
    app.process_all(); root.update()
    asked = len([d for d in dialogs if d[0] == "yesno"]) == n_yes + 1
    rd_b = sorted(glob.glob(os.path.join(SMOKE, "adaptFVC", "results", "run_*")))[-1]
    run_ann = load_model_file(glob.glob(os.path.join(rd_b, "annotations_*.json"))[0])["annotations"]
    run_mod = load_model_file(glob.glob(os.path.join(rd_b, "model_*.pkl"))[0])["annotations"]
    check("process_all after edits: asks, retrains, run json == model's points == current",
          asked and run_ann == run_mod == annotations_to_portable(app.annotations),
          f"asked={asked} json={sum(map(len, run_ann.values()))} model={sum(map(len, run_mod.values()))}")
    # 学習のあとにさらに1点足して閉じる。復元でその点も戻ること（以前は最後に学習した時点の教師点に戻っていた）
    app.annotations[p0].append({'nx': 0.52, 'ny': 0.5, 'class': 'soil', 'size': 1, 'click': True})
    app.annotation_groups[p0].append(1)
    n_before_close = sum(len(v) for v in app.annotations.values())

    # 終了 → 再起動で復元
    app._on_close()
    dialogs.clear()
    answers["askyesno"] = True
    root = tk.Tk()
    app = AnnotationApp(root)
    pump(root, 0.6)
    check("restore prompt shown", any(d[0] == "yesno" and "前回" in d[1] for d in dialogs))
    check("annotations restored incl. points added after the last training",
          sum(len(v) for v in app.annotations.values()) == n_before_close,
          f"{sum(len(v) for v in app.annotations.values())} vs {n_before_close}")
    check("model restored & usable", app.model is not None and app.model_info.get("source") == "restore")
    wait_idle(app, root)
    check("restored: inference shown", app.show_inference.get() and app.current_overlay_pil is not None)

    # アルファ付きフォルダ（透明画素の除外・TIFF）
    alpha_dir = make_alpha_folder()
    filedialog.askdirectory = lambda **k: alpha_dir
    app.change_folder(); pump(root, 0.4)
    check("alpha folder opened (png+tif)", len(app.image_paths) == 16 and any(p.lower().endswith(".tif") for p in app.image_paths))
    synth_points(app, app.image_paths[:3], n=30)
    # 透明な隅に打った点は学習から除外される
    app.annotations[app.image_paths[0]].append({'nx': 0.01, 'ny': 0.01, 'class': 'soil', 'size': 1})
    app.update_annotation_counts()
    app.train_model_ui(source="key"); wait_idle(app, root)
    check("alpha: trained & skipped invalid point", app.model is not None and app.model_info.get("n_skipped_invalid") == 1,
          app.model_info)
    cov = app.infer_cache.get(app.current_image_path, (None, None))[1]
    from adaptfvc.core import load_image, predict_mask, calculate_coverage
    bgr, valid = load_image(app.current_image_path)
    naive = calculate_coverage(predict_mask(bgr, app.model))
    check("alpha: coverage excludes transparent corners (> naive)", cov is not None and cov > naive * 1.3, f"{cov:.1f} vs naive {naive:.1f}")
    app.process_all(); root.update()
    runs = sorted(glob.glob(os.path.join(alpha_dir, "adaptFVC", "results", "run_*")))
    from openpyxl import load_workbook
    wb = load_workbook(glob.glob(os.path.join(runs[-1], "*.xlsx"))[0])
    ws = wb["結果"]
    vfs = [ws.cell(row=r, column=5).value for r in range(2, 18)]
    check("alpha: xlsx valid fraction < 100 for all", all(isinstance(v, (int, float)) and 40 < v < 60 for v in vfs), vfs)
    ov = Image.open(glob.glob(os.path.join(runs[-1], "overlays", "*.png"))[0])
    arr = np.array(ov.convert("RGB"))
    check("alpha: overlay paints ROI-outside gray", tuple(arr[0, 0]) == (128, 128, 128), tuple(arr[0, 0]))
    app._on_close()

    # CLI バッチ（GUI なし）: 自動保存されたモデルで再実行
    model_a = os.path.join(alpha_dir, "adaptFVC", "models", "model_latest.pkl")
    r = subprocess.run([PY, "-m", "adaptfvc", "--batch", alpha_dir, "--model", model_a],
                       capture_output=True, cwd=ROOT, timeout=300, **CHILD_IO)
    runs2 = sorted(glob.glob(os.path.join(alpha_dir, "adaptFVC", "results", "run_*")))
    check("cli batch exit 0", r.returncode == 0, r.stdout[-300:] + r.stderr[-300:])
    check("cli batch produced new run", len(runs2) == len(runs) + 1)
    # 教師点 JSON だけからでも再学習して処理できる（再現性の要）
    ann_a = os.path.join(alpha_dir, "adaptFVC", "models", "annotations_latest.json")
    r2 = subprocess.run([PY, "-m", "adaptfvc", "--batch", alpha_dir, "--model", ann_a],
                        capture_output=True, cwd=ROOT, timeout=300, **CHILD_IO)
    check("cli batch from annotations json", r2.returncode == 0 and "学習し直します" in r2.stdout, r2.stdout[-200:])
    if r.returncode == 0 and r2.returncode == 0:
        runs3 = sorted(glob.glob(os.path.join(alpha_dir, "adaptFVC", "results", "run_*")))
        x1 = load_workbook(glob.glob(os.path.join(runs3[-2], "*.xlsx"))[0])["結果"]
        x2 = load_workbook(glob.glob(os.path.join(runs3[-1], "*.xlsx"))[0])["結果"]
        same = all(x1.cell(row=r_, column=4).value == x2.cell(row=r_, column=4).value for r_ in range(2, 18))
        check("reproducible: pkl run == json-retrained run", same)

    shutil.rmtree(alpha_dir, ignore_errors=True)
    check("no error dialogs", not any(d[0] == "error" for d in dialogs), [d for d in dialogs if d[0] == "error"])
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    return n_fail


if __name__ == "__main__":
    sys.exit(main())
