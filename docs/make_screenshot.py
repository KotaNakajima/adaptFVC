# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""取扱説明書のメイン画面のスクリーンショット（docs/figures/screenshot_main.png）を撮る。Windows 専用。

test_smoke/ の画像（.JPG を 10 枚）を一時フォルダに写してアプリで開き、2 枚に教師点を打って学習し、
推論表示を ON にした状態のウィンドウを撮る。教師点はアプリのクリック処理（on_image_click）に
座標を渡して打つので、連続注釈の直線補間・クリック数の表示も実物どおりになる。

使い方: python docs/make_screenshot.py   （数秒だけアプリの画面が開く。表示倍率 100% の画面で撮ること）
"""
import ctypes
import glob
import os
import shutil
import sys
import tempfile
import time
import tkinter as tk
from ctypes import wintypes
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import ImageGrab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SMOKE = os.path.join(ROOT, "test_smoke")
OUT = os.path.join(ROOT, "docs", "figures", "screenshot_main.png")


def pump(root, seconds):
    t_end = time.time() + seconds
    while time.time() < t_end:
        root.update()
        time.sleep(0.02)


class _Ev:
    def __init__(self, x, y):
        self.x, self.y = x, y


def strokes(img_path, cls, n_strokes, rng):
    """クラスの「芯」（境界から離れた画素）の中で、端から端まで同じクラスに収まる短い線分を選ぶ"""
    bgr = cv2.imread(img_path)
    b, g, r = cv2.split(bgr.astype(np.int32))
    exg = cv2.normalize(2 * g - r - b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, plant = cv2.threshold(exg, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = plant if cls == "plant" else 1 - plant
    core_mask = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    h, w = core_mask.shape
    ys, xs = np.nonzero(core_mask)
    out, tries = [], 0
    while len(out) < n_strokes and tries < 5000:
        tries += 1
        i = rng.randint(len(ys))
        x0, y0 = xs[i], ys[i]
        ang = rng.uniform(0, np.pi)
        ln = rng.randint(10, 24)
        x1, y1 = int(x0 + np.cos(ang) * ln), int(y0 + np.sin(ang) * ln)
        if not (0 <= x1 < w and 0 <= y1 < h):
            continue
        t = np.linspace(0, 1, ln + 1)
        if core_mask[(y0 + (y1 - y0) * t).astype(int), (x0 + (x1 - x0) * t).astype(int)].all():
            out.append(((x0 / w, y0 / h), (x1 / w, y1 / h)))
    return out


def click(app, nx, ny):
    app.on_image_click(_Ev(nx * app.display_w * app.zoom - app.pan_x, ny * app.display_h * app.zoom - app.pan_y))


def window_bbox(root):
    """影を除いたウィンドウの外枠（タイトルバー込み）を画面座標で返す"""
    hwnd = int(root.wm_frame(), 16)
    rect = wintypes.RECT()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect))  # 9 = DWMWA_EXTENDED_FRAME_BOUNDS
    return rect.left, rect.top, rect.right, rect.bottom


def main():
    # 表示倍率が 100% でない画面では、DPI 非対応のままだと Windows が窓を拡大してぼやける。
    # DPI 対応を宣言し、Tk の倍率を 100% 相当（96/72）に固定して、等倍のくっきりした画面を撮る
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    jpgs = sorted(glob.glob(os.path.join(SMOKE, "*.JPG")))
    if len(jpgs) < 2:
        print("test_smoke/ に .JPG の画像を置いてください")
        return 2
    work = tempfile.mkdtemp(prefix="adaptfvc_shot_")
    for p in jpgs:
        shutil.copy2(p, work)
    filedialog.askdirectory = lambda **k: work
    messagebox.askyesno = lambda *a, **k: False
    messagebox.showinfo = messagebox.showwarning = lambda *a, **k: None

    from adaptfvc.gui import AnnotationApp
    AnnotationApp._ask_resize_size = lambda self, n, m: 1024  # 縮小サイズの選択ダイアログに自動で答える（推奨値）
    root = tk.Tk()
    root.tk.call("tk", "scaling", 96 / 72)
    app = AnnotationApp(root)
    root.geometry("1200x800+40+40")
    pump(root, 0.8)
    rng = np.random.RandomState(1)
    targets = app.annotation_images[:2]
    for path in targets:
        app.display_image(path)
        pump(root, 0.2)
        for cls, n in (("plant", 6), ("soil", 5)):
            app.set_annotation_class(cls)
            for (p0, p1) in strokes(path, cls, n, rng):
                click(app, *p0)
                click(app, *p1)
                app._reset_seq_state()
        pump(root, 0.2)
    app.train_model_ui(source="key")
    pump(root, 1.0)
    app.navigate_prev(source="key")  # 推論表示 ON のまま 1 枚目へ戻る（切り替えで自動推論される）
    pump(root, 0.3)
    if app.current_image_path != targets[0]:
        app.display_image(targets[0])
    app.set_annotation_class("plant")
    t_end = time.time() + 30
    while time.time() < t_end and (app._infer_running is not None or app._infer_pending is not None):
        root.update()
        time.sleep(0.02)
    root.attributes("-topmost", True)
    root.lift()
    pump(root, 1.5)
    bbox = window_bbox(root)
    ImageGrab.grab(bbox=bbox, all_screens=True).save(OUT)
    print(f"saved {OUT} {bbox}  {app.plant_count_label.cget('text')} / {app.soil_count_label.cget('text')}")
    app._on_close()
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
