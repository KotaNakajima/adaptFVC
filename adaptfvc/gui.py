# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""対話型セグメンテーションGUI（AnnotationApp）。

学習データの準備 → 学習 → 確認 → 全画像の処理 のループを1枚の画面で提供する（v3: タブなし）。
計算は adaptfvc.core / workers / batch に、保存は adaptfvc.modelio に委譲する。
"""

import glob
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from adaptfvc import __copyright__, __url__, __version__, imagecache
from adaptfvc.batch import run_batch, run_full
from adaptfvc.core import (
    calculate_coverage,
    create_overlay_from_mask,
    extract_training_pixels,
    imread_unicode,
    load_image,
    segment_image,
    train_rf,
)
from adaptfvc.modelio import (
    annotations_to_portable,
    click_groups,
    count_clicks,
    count_points,
    load_model_file,
    model_is_usable,
    portable_to_annotations,
    save_annotations_json,
    save_model_bundle,
)
from adaptfvc.session_log import NullSessionLogger, SessionLogger

# --- UI 配色（強調の階層を色で符号化する） ---
# 主要アクション（次に押すべきもの）= 緑 / 補助 = 無彩色（既定）/ 破壊的・注意 = 琥珀
# クラスは 植物=緑・土壌=茶 で一貫させる（アイコン・取説の図と同じ系統色）
UI_COLORS = {
    "accent": "#2e9e5b",
    "accent_active": "#26824c",
    "plant": "#1e8a4a",
    "plant_bg": "#e2f4e7",
    "soil": "#7a5b40",
    "soil_bg": "#efe6dc",
    "warn": "#b85e10",
    "warn_bg": "#fdf1e4",
    "note": "#5a6e61",
}

SHORTCUTS = [
    ("クリック", "表示中の画像に教師点を打つ（同じクラスで続けると直線補間）"),
    ("Q", "植物をアノテーションする（押すと必ず「植物」になる）"),
    ("W", "土壌をアノテーションする（押すと必ず「土壌」になる）"),
    ("Z", "直前のクリックを取り消す"),
    ("X", "連続クリックの直線補間を ON/OFF"),
    ("T", "学習（教師点からモデルを作る）"),
    ("Space", "推論結果の表示を ON/OFF（ON のあいだは画像を切り替えると自動で推論）"),
    ("A / D, ← / →", "前の画像 / 次の画像"),
    ("ホイール", "拡大・縮小（中ボタン or 右ボタンのドラッグで移動）"),
]


class AnnotationApp:
    # 「推論表示」の縮小規則: 長辺がこの値を超える画像は長辺をこの値に縮小してから推論する。
    # 全体処理は作業画像の解像度のまま
    INFER_MAX_LONG_SIDE = 1500
    AUTOSAVE_DELAY_MS = 800

    def __init__(self, root):
        self.root = root
        self.root.title(f"adaptFVC v{__version__} — 植被率 対話型セグメンテーション")
        self.root.geometry("1200x800")
        self.root.minsize(980, 620)

        # --- データ ---
        self.image_folder = ""
        self.image_paths = []            # 作業画像（縮小キャッシュ使用時はキャッシュのパス）
        self.annotation_images = []      # 教師点を打つ対象（代表画像 + 追加分）
        self.current_image_path = None
        self.current_image_index = -1
        self.annotations = {}            # {作業パス: [{'nx','ny','class','size'}]}
        self.annotation_groups = {}      # クリック単位の Undo 用 {path: [点数]}
        self.cache_long_side = None
        self.model = None
        # いまのモデルを学習した教師点（portable 形式）。モデルの保存と全画像の処理で、モデルと一緒に書き出す
        self.model_annotations = None
        # モデルを設定した時点の教師点（portable 形式）。その後に教師点が変わったかの判定に使う
        self.annotations_at_model = None
        self.model_info = {}
        self.loop_round = 0              # 学習の回数（学習→確認→教師点の追加 の周回）
        self.preview_count = 0
        self.batch_mode = False

        # --- 表示 ---
        self.display_w, self.display_h = 800, 600
        self.current_overlay_pil = None
        self._last_canvas_size = (0, 0)
        self.zoom, self.pan_x, self.pan_y = 1.0, 0, 0
        self._is_panning, self._pan_start = False, (0, 0)
        self._busy = False

        # --- 連続注釈（最大5クリック） ---
        self.seq_interp_enabled = tk.BooleanVar(value=True)
        self.seq_interp_enabled.trace_add("write", lambda *a: self._reset_seq_state())
        self.seq_count, self.seq_class, self.seq_last, self.seq_image_path = 0, None, None, None

        # --- 推論表示（Space） ---
        self.show_inference = tk.BooleanVar(value=False)
        self.infer_cache = {}            # {path: (overlay_pil, coverage, capped)}
        self._model_generation = 0
        self._infer_running = None
        self._infer_pending = None
        self._infer_result = None

        # --- 自動保存 ---
        self._autosave_after_id = None

        self.session_logger = NullSessionLogger()

        self.setup_ui()
        self.bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.prompt_for_folder()

    # ==========================================================
    # UI 構築
    # ==========================================================
    def _primary_button(self, parent, text, command, big=False):
        return tk.Button(parent, text=text, command=command,
                         bg=UI_COLORS["accent"], fg="white",
                         activebackground=UI_COLORS["accent_active"], activeforeground="white",
                         disabledforeground="#bcd9c6",
                         font=("Yu Gothic UI", 12 if big else 10, "bold"),
                         relief=tk.FLAT, padx=8, pady=12 if big else 6, cursor="hand2")

    def _note_label(self, parent, text):
        return tk.Label(parent, text=text, font=("Yu Gothic UI", 8),
                        fg=UI_COLORS["note"], wraplength=250, justify=tk.LEFT)

    def _small_button(self, parent, text, command):
        """補助操作用の小さなボタン（左パネルに横並びで置く）"""
        return tk.Button(parent, text=text, command=command, font=("Yu Gothic UI", 8), pady=1)

    def setup_ui(self):
        self.setup_menu()
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = tk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.setup_left_panel(left_panel)

        right_panel = tk.Frame(main_frame, width=200, relief=tk.RIDGE, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_panel.pack_propagate(False)
        tk.Label(right_panel, text="学習用の画像", font=("Yu Gothic UI", 10, "bold"),
                 fg=UI_COLORS["note"]).pack(pady=(4, 0))

        center_panel = tk.Frame(main_frame, relief=tk.RIDGE, borderwidth=2)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        nav_frame = tk.Frame(center_panel)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))
        # 画像のすぐ上に「いまどのクラスをアノテーション中か」を大きく出す。目線を画像から外さずに確認できる
        self.canvas_class_banner = tk.Label(center_panel, text="", font=("Yu Gothic UI", 13, "bold"),
                                            fg="white", pady=4)
        self.canvas_class_banner.pack(side=tk.TOP, fill=tk.X)
        # キャンバスの枠もクラス色にする（クリックする場所そのものが色で囲まれる）
        self.image_canvas = tk.Canvas(center_panel, bg="#d9d9d9", highlightthickness=4)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.prev_btn = tk.Button(nav_frame, text="◀ 前の画像 (A)", width=15, command=self.navigate_prev)
        self.prev_btn.pack(side=tk.LEFT, padx=4)
        self.nav_label = tk.Label(nav_frame, text="- / -", font=("Helvetica", 10))
        self.nav_label.pack(side=tk.LEFT, padx=6)
        self.infer_status_label = tk.Label(nav_frame, text="", font=("Yu Gothic UI", 9, "bold"),
                                           fg=UI_COLORS["note"])
        self.infer_status_label.pack(side=tk.LEFT, padx=8, expand=True)
        self.progress_label = tk.Label(nav_frame, text="", font=("Yu Gothic UI", 9, "bold"),
                                       fg=UI_COLORS["accent"])
        self.progress_label.pack(side=tk.LEFT, padx=8)
        self.zoom_label = tk.Label(nav_frame, text="100%", font=("Yu Gothic UI", 8), fg=UI_COLORS["note"])
        self.zoom_label.pack(side=tk.LEFT, padx=4)
        self.next_btn = tk.Button(nav_frame, text="次の画像 (D) ▶", width=15, command=self.navigate_next)
        self.next_btn.pack(side=tk.RIGHT, padx=4)

        self.image_canvas.bind("<Button-1>", self.on_image_click)
        self.image_canvas.bind("<Configure>", self.on_canvas_configure)
        self.image_canvas.bind("<MouseWheel>", self.on_mousewheel_zoom)
        self.image_canvas.bind("<Button-4>", self.on_mousewheel_zoom)
        self.image_canvas.bind("<Button-5>", self.on_mousewheel_zoom)
        for b in ("2", "3"):
            self.image_canvas.bind(f"<ButtonPress-{b}>", self.on_pan_start)
            self.image_canvas.bind(f"<B{b}-Motion>", self.on_pan_move)
            self.image_canvas.bind(f"<ButtonRelease-{b}>", self.on_pan_end)

        self.thumbnail_canvas = tk.Canvas(right_panel)
        sb = tk.Scrollbar(right_panel, orient="vertical", command=self.thumbnail_canvas.yview)
        self.thumbnail_frame = tk.Frame(self.thumbnail_canvas)
        self.thumbnail_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.thumbnail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.thumbnail_canvas.create_window((0, 0), window=self.thumbnail_frame, anchor="nw")
        self.thumbnail_frame.bind(
            "<Configure>", lambda e: self.thumbnail_canvas.configure(scrollregion=self.thumbnail_canvas.bbox("all")))

        # ボタン類にキーボードフォーカスを渡さない（押した直後の Space がボタンを再発火するのを防ぐ）
        self._disable_widget_focus(main_frame)
        self._update_class_banner()
        self._update_infer_status()
        self._update_model_status()

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="別のフォルダを開く...", command=self.change_folder)
        m_file.add_separator()
        m_file.add_command(label="モデル / 教師点を読み込む...", command=self.load_model_dialog)
        m_file.add_command(label="モデルを別名で保存...", command=self.save_model_as)
        m_file.add_command(label="教師点を打つ画像を追加...", command=self.add_images_from_folder)
        m_file.add_separator()
        m_file.add_command(label="出力フォルダを開く", command=self.open_output_folder)
        m_file.add_separator()
        m_file.add_command(label="終了", command=self._on_close)
        menubar.add_cascade(label="ファイル", menu=m_file)

        m_view = tk.Menu(menubar, tearoff=0)
        m_view.add_checkbutton(label="推論結果を表示 (Space)", variable=self.show_inference,
                               command=lambda: self._apply_inference_display(source="menu"))
        m_view.add_separator()
        m_view.add_command(label="拡大", command=lambda: self.on_mousewheel_zoom(self._fake_wheel_event(120)))
        m_view.add_command(label="縮小", command=lambda: self.on_mousewheel_zoom(self._fake_wheel_event(-120)))
        m_view.add_command(label="ズームをリセット", command=self.on_zoom_reset)
        menubar.add_cascade(label="表示", menu=m_view)

        m_set = tk.Menu(menubar, tearoff=0)
        m_set.add_command(label="縮小サイズを選び直す...", command=self.reset_resize_choice)
        m_set.add_command(label="リセット...", command=self.reset_dialog)
        menubar.add_cascade(label="設定", menu=m_set)

        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="ショートカット一覧", command=self.show_shortcuts)
        m_help.add_command(label="バージョン情報", command=self.show_about)
        menubar.add_cascade(label="ヘルプ", menu=m_help)
        self.root.config(menu=menubar)

    def setup_left_panel(self, parent):
        # ---- 1 学習データの準備 ----
        f1 = tk.LabelFrame(parent, text=" 1  学習データの準備 ", font=("Yu Gothic UI", 10, "bold"), padx=8, pady=6)
        f1.pack(fill=tk.X, pady=(0, 8))
        self.annotation_class = tk.StringVar(value="plant")
        self.annotation_class.trace_add("write", lambda *a: self._reset_seq_state())
        self.annotation_class.trace_add("write", lambda *a: self._update_class_banner())
        self.current_class_banner = tk.Label(f1, text="", font=("Yu Gothic UI", 11, "bold"), pady=6)
        self.current_class_banner.pack(fill=tk.X, pady=(2, 4))
        log_cls = lambda: self._slog("class_switch", cls=self.annotation_class.get(), source="button")
        tk.Radiobutton(f1, text="🌱 植物 (Plant)   Q", variable=self.annotation_class, value="plant",
                       command=log_cls, fg=UI_COLORS["plant"], font=("Yu Gothic UI", 10)).pack(anchor=tk.W)
        tk.Radiobutton(f1, text="🟫 土壌 (Soil)   W", variable=self.annotation_class, value="soil",
                       command=log_cls, fg=UI_COLORS["soil"], font=("Yu Gothic UI", 10)).pack(anchor=tk.W)
        self._update_class_banner()

        self.plant_count_label = tk.Label(f1, text="🌱 植物: 0 クリック（0 画素）", font=("Yu Gothic UI", 9))
        self.plant_count_label.pack(anchor=tk.W, pady=(8, 0))
        self.soil_count_label = tk.Label(f1, text="🟫 土壌: 0 クリック（0 画素）", font=("Yu Gothic UI", 9))
        self.soil_count_label.pack(anchor=tk.W)
        self.image_annot_stats_label = tk.Label(f1, text="教師点を打った画像: 0 枚", font=("Yu Gothic UI", 8),
                                                fg=UI_COLORS["note"])
        self.image_annot_stats_label.pack(anchor=tk.W)
        self.seq_progress_label = tk.Label(f1, text="連続注釈: 0/5", font=("Yu Gothic UI", 9))
        self.seq_progress_label.pack(anchor=tk.W, pady=(6, 0))
        tk.Checkbutton(f1, text="連続注釈の直線補間 (X)", variable=self.seq_interp_enabled,
                       command=lambda: self._slog("seq_toggle", on=bool(self.seq_interp_enabled.get()), source="button"),
                       font=("Yu Gothic UI", 8)).pack(anchor=tk.W)
        row1 = tk.Frame(f1)
        row1.pack(fill=tk.X, pady=(6, 2))
        tk.Button(row1, text="直前を取り消し (Z)", command=self.undo_last_annotation).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self._small_button(row1, "画像を追加...", self.add_images_from_folder).pack(side=tk.LEFT, fill=tk.Y)

        # ---- 2 学習と確認 ----
        f2 = tk.LabelFrame(parent, text=" 2  学習と確認 ", font=("Yu Gothic UI", 10, "bold"), padx=8, pady=6)
        f2.pack(fill=tk.X, pady=(0, 8))
        self.train_button = self._primary_button(f2, "学習 (T)", self.train_model_ui)
        self.train_button.pack(fill=tk.X, pady=(4, 2))
        self.model_status_label = tk.Label(f2, text="", font=("Yu Gothic UI", 8), fg=UI_COLORS["note"],
                                           wraplength=250, justify=tk.LEFT)
        self.model_status_label.pack(anchor=tk.W)
        # よく使うファイル操作はメニューに隠さず、ここに小さなボタンで置く
        row2 = tk.Frame(f2)
        row2.pack(fill=tk.X, pady=(2, 0))
        self._small_button(row2, "モデル/教師点を読み込む...", self.load_model_dialog).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self._small_button(row2, "別名で保存...", self.save_model_as).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Checkbutton(f2, text="推論結果を表示 (Space)", variable=self.show_inference,
                       command=lambda: self._apply_inference_display(source="button"),
                       font=("Yu Gothic UI", 9)).pack(anchor=tk.W, pady=(6, 0))
        # 注記に半角スペースを入れない: Tk はスペースで折り返すため「ON」「OFF」の前後で変な改行になる
        self._note_label(f2, "ONのあいだ画像を切り替えるたびに自動で推論して重ねます。"
                             "OFFで素の画像に戻り、教師点を足す場所を見比べられます。").pack(anchor=tk.W)
        tk.Button(f2, text="プレビュー（未注釈の画像 10 枚）", command=self.preview).pack(fill=tk.X, pady=(6, 2))
        self._note_label(f2, "学習に使っていない画像で汎化を確認。うまく分類できていない画像を選んで教師点を足し、再学習。").pack(anchor=tk.W)
        self.latest_coverage_label = tk.Label(f2, text="最新の植被率: -",
                                              font=("Yu Gothic UI", 12, "bold"), fg=UI_COLORS["accent"])
        self.latest_coverage_label.pack(anchor=tk.W, pady=(6, 0))

        # ---- 3 全画像の処理 ----
        f3 = tk.LabelFrame(parent, text=" 3  全画像の処理 ", font=("Yu Gothic UI", 10, "bold"), padx=8, pady=6)
        f3.pack(fill=tk.X)
        self._primary_button(f3, "全画像を処理", self.process_all, big=True).pack(fill=tk.X, pady=(4, 2))
        self._note_label(f3, "全画像を処理し、画像フォルダ内のadaptFVC/results/に判定画像・Excel・モデル・教師点を保存。").pack(anchor=tk.W)
        self._small_button(f3, "出力フォルダを開く", self.open_output_folder).pack(fill=tk.X, pady=(4, 0))

    def _disable_widget_focus(self, widget):
        try:
            if isinstance(widget, (tk.Button, tk.Checkbutton, tk.Radiobutton)):
                widget.config(takefocus=0)
            for child in widget.winfo_children():
                self._disable_widget_focus(child)
        except Exception:
            pass

    def _update_class_banner(self):
        plant = self.annotation_class.get() == "plant"
        if hasattr(self, "current_class_banner"):
            if plant:
                self.current_class_banner.config(text="アノテーション中のクラス: 🌱 植物",
                                                 bg=UI_COLORS["plant_bg"], fg=UI_COLORS["plant"])
            else:
                self.current_class_banner.config(text="アノテーション中のクラス: 🟫 土壌",
                                                 bg=UI_COLORS["soil_bg"], fg=UI_COLORS["soil"])
        color = UI_COLORS["plant"] if plant else UI_COLORS["soil"]
        if hasattr(self, "canvas_class_banner"):
            self.canvas_class_banner.config(
                text=("いま「🌱 植物」をアノテーション中" if plant else "いま「🟫 土壌」をアノテーション中")
                + "　　（Q = 植物 / W = 土壌）", bg=color)
        if hasattr(self, "image_canvas"):
            self.image_canvas.config(highlightbackground=color, highlightcolor=color)

    def bind_shortcuts(self):
        """アプリ全体のショートカット（左手配置: Q/A/D/X/Z/T/Space、右手はマウス）"""
        def is_text_input(w):
            try:
                return (w.winfo_class() if w else "") in ("Entry", "TEntry", "Spinbox", "TSpinbox", "Text", "TCombobox")
            except Exception:
                return False

        def guarded(fn):
            def handler(e):
                focus_w = self.root.focus_get()
                if is_text_input(focus_w):
                    return
                try:
                    if focus_w is not None and focus_w.winfo_toplevel() is not self.root:
                        return
                except Exception:
                    pass
                try:
                    fn()
                except Exception as ex:
                    print(f"[shortcut] error: {ex}")
            return handler

        binds = {
            ("<Key-z>", "<Key-Z>"): lambda: self.undo_last_annotation(source="key"),
            ("<Left>", "<Key-a>", "<Key-A>"): lambda: self.navigate_prev(source="key"),
            ("<Right>", "<Key-d>", "<Key-D>"): lambda: self.navigate_next(source="key"),
            ("<space>",): lambda: self.toggle_inference_display(source="key"),
            ("<Key-q>", "<Key-Q>"): lambda: self.set_annotation_class("plant", source="key"),
            ("<Key-w>", "<Key-W>"): lambda: self.set_annotation_class("soil", source="key"),
            ("<Key-x>", "<Key-X>"): lambda: self.toggle_seq_interp(source="key"),
            ("<Key-t>", "<Key-T>"): lambda: self.train_model_ui(source="key"),
        }
        for keys, fn in binds.items():
            for k in keys:
                self.root.unbind_all(k)
                self.root.bind_all(k, guarded(fn))

    # ==========================================================
    # フォルダ・画像の読み込み
    # ==========================================================
    def _app_dir(self, *parts, make=False):
        return imagecache.app_dir(self.image_folder, *parts, make=make)

    def _slog(self, event, **data):
        try:
            self.session_logger.event(event, data)
        except Exception:
            pass

    def _show_batch_progress(self, text):
        """画面下部の進捗ラベルを更新し、UIイベントを1回処理する（フリーズ対策）"""
        try:
            self.progress_label.config(text=text)
            self.root.update()
        except Exception:
            pass

    def prompt_for_folder(self):
        folder = filedialog.askdirectory(title="画像フォルダを選択してください")
        if not folder:
            messagebox.showwarning("警告", "フォルダが選択されませんでした。アプリケーションを終了します。")
            self.root.destroy()
            return
        self.open_folder(folder)

    def change_folder(self):
        folder = filedialog.askdirectory(title="画像フォルダを選択してください", initialdir=self.image_folder or None)
        if not folder or os.path.normpath(folder) == os.path.normpath(self.image_folder):
            return
        self._slog("session_end", reason="change_folder")
        self.session_logger.close()
        self.open_folder(folder)

    def open_folder(self, folder):
        """フォルダを開いて作業状態を初期化する（起動時・フォルダ変更時）"""
        paths = imagecache.discover_images(folder)
        if not paths:
            messagebox.showerror("エラー", "選択されたフォルダに画像 (jpg/png/tif) が見つかりません。")
            if not self.image_folder:
                self.prompt_for_folder()
            return
        self.image_folder = folder
        self.annotations, self.annotation_groups = {}, {}
        self.model, self.model_info = None, {}
        self.loop_round, self.preview_count = 0, 0
        self.show_inference.set(False)
        self._on_model_changed()
        self.cache_long_side = None
        self._cancel_autosave()

        imagecache.migrate_legacy_outputs(folder)
        resize_log = None
        try:
            paths, resize_log = self._prepare_resized_cache(paths)
        except Exception as e:
            print(f"縮小キャッシュの準備に失敗（原寸で続行）: {e}")
        self.image_paths = paths

        total = len(self.image_paths)
        if total <= 10:
            self.annotation_images = list(self.image_paths)
        else:
            num = self.compute_representative_count(total)
            self.annotation_images = self.select_representative_images(self.image_paths, num)
        self.display_thumbnails()
        self.display_image(0)
        self.update_annotation_counts()
        self._update_model_status()

        try:
            self.session_logger = SessionLogger(self._app_dir("logs"), __version__)
        except Exception:
            self.session_logger = NullSessionLogger()
        self._slog("images_loaded", n_images=total, n_representative=len(self.annotation_images),
                   folder=os.path.basename(folder))
        if resize_log:
            self._slog("resize_cache", **resize_log)
        self.root.after(200, self._offer_restore_autosave)

    def _prepare_resized_cache(self, paths):
        """長辺が大きい画像を縮小版に差し替える（選択はフォルダごとに記憶）"""
        n_over, max_dim = imagecache.needs_resize_choice(paths)
        if n_over == 0:
            return paths, None
        long_side = imagecache.load_resize_choice(self.image_folder)
        if long_side is None:
            long_side = self._ask_resize_size(n_over, max_dim)
            imagecache.save_resize_choice(self.image_folder, long_side)
        self.cache_long_side = long_side
        new_paths, stats = imagecache.prepare_resized_cache(
            self.image_folder, paths, long_side, progress=self._show_batch_progress)
        return new_paths, stats

    def _ask_resize_size(self, n_large, max_dim):
        """縮小後の長辺を 256/512/1024 から選ばせるモーダル。閉じたら既定値"""
        result = {"v": imagecache.CACHE_SIZE_DEFAULT}
        dlg = tk.Toplevel(self.root)
        dlg.title("縮小サイズの選択")
        dlg.resizable(False, False)
        tk.Label(dlg, justify=tk.LEFT, padx=14, pady=10, text=(
            f"長辺が大きい画像が {n_large} 枚あります（最大 {max_dim}px）。\n"
            "縮小後の長辺を選んでください。以後の表示・学習・植被率計算は\n"
            "すべてこのサイズの縮小版で行われます（元の画像は変更されません）。\n"
            "この選択はフォルダごとに記憶され、次回からは聞かれません。")).pack(anchor=tk.W)
        var = tk.IntVar(value=imagecache.CACHE_SIZE_DEFAULT)
        notes = {
            1024: "1024 px — 精度優先（推奨。植被率誤差の目安 +0.2pt）",
            512: "512 px — バランス（目安 +0.7pt）",
            256: "256 px — 速度優先（誤差1pt超になりうる。幼苗・細い葉では非推奨）",
        }
        for v in sorted(imagecache.CACHE_SIZE_CHOICES, reverse=True):
            tk.Radiobutton(dlg, text=notes[v], variable=var, value=v).pack(anchor=tk.W, padx=18)

        def ok():
            result["v"] = var.get()
            dlg.destroy()

        self._primary_button(dlg, "決定", ok).pack(pady=10)
        dlg.protocol("WM_DELETE_WINDOW", ok)
        dlg.transient(self.root)
        try:
            dlg.grab_set()
        except Exception:
            pass
        self.root.wait_window(dlg)
        return result["v"]

    def reset_resize_choice(self):
        try:
            imagecache.clear_resize_choice(self.image_folder)
            self._slog("resize_choice_reset")
            messagebox.showinfo(
                "縮小サイズ",
                "記憶した縮小サイズを消去しました。アプリを起動し直してこのフォルダを開くと、"
                "選択ダイアログが再表示されます。\n（解像度が変わると色の混ざり方が変わるため、"
                "教師点とモデルは解像度ごとに作り直すのが安全です）")
        except Exception as e:
            messagebox.showerror("エラー", f"消去に失敗しました: {e}")

    def compute_representative_count(self, total):
        """代表画像の枚数（総数の10%を切り上げ、最低10枚・最大50枚。総数10枚以下は総数）"""
        if total <= 10:
            return total
        return min(50, max(10, math.ceil(total * 0.10)))

    def select_representative_images(self, image_paths, num_images):
        """輝度の分布を等間隔にカバーするように代表画像を選ぶ"""
        pool_cap = max(num_images * 10, 200)
        if len(image_paths) > pool_cap:
            idx = np.linspace(0, len(image_paths) - 1, pool_cap, dtype=int)
            image_paths = [image_paths[i] for i in idx]
        metrics = []
        for path in image_paths:
            try:
                img = imread_unicode(path)
                if img is None:
                    continue
                metrics.append({'path': path, 'brightness': np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))})
            except Exception as e:
                print(f"Error processing {path}: {e}")
        if not metrics:
            return image_paths[:num_images]
        metrics.sort(key=lambda x: x['brightness'])
        indices = np.linspace(0, len(metrics) - 1, num_images, dtype=float)
        unique_indices = sorted(set(int(round(x)) for x in indices))
        return [metrics[i]['path'] for i in unique_indices]

    # ==========================================================
    # サムネイル
    # ==========================================================
    def display_thumbnails(self):
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()
        for i, path in enumerate(self.annotation_images):
            try:
                img = Image.open(path)
                img.thumbnail((150, 150))
                tk_img = ImageTk.PhotoImage(img)
            except Exception:
                continue
            w, h = tk_img.width(), tk_img.height()
            thumb = tk.Canvas(self.thumbnail_frame, width=w, height=h, highlightthickness=0, bd=0)
            thumb.create_image(0, 0, anchor=tk.NW, image=tk_img)
            thumb.image = tk_img
            thumb.path = path
            anns = self.annotations.get(path, [])
            has_p = any(a.get('class') == 'plant' for a in anns)
            has_s = any(a.get('class') != 'plant' for a in anns)

            def dashed_rect(cnv, box, color):
                x0, y0, x1, y1 = box
                for seg in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)):
                    cnv.create_line(*seg, fill=color, width=2, dash=(6, 4))
            if has_p:
                dashed_rect(thumb, (2, 2, w - 3, h - 3), "#00cc00")
            if has_s:
                inset = 6 if has_p else 2
                dashed_rect(thumb, (inset, inset, w - 1 - inset, h - 1 - inset), "#d68c46")
            text_id = thumb.create_text(w - 2, 2, text=os.path.basename(path), anchor='ne',
                                        fill='black', font=('TkDefaultFont', 8))
            bbox = thumb.bbox(text_id)
            if bbox:
                rect_id = thumb.create_rectangle(bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2,
                                                 fill='white', outline='')
                thumb.tag_lower(rect_id, text_id)
            thumb.bind("<Button-1>", lambda e, idx=i: self.display_image(idx))
            thumb.bind("<Button-3>", self.on_thumbnail_right_click)
            thumb.pack(pady=5)

    def on_thumbnail_right_click(self, event):
        widget = event.widget
        if hasattr(widget, 'path'):
            menu = tk.Menu(self.root, tearoff=0)
            menu.add_command(label="学習用の画像から外す",
                             command=lambda p=widget.path: self.remove_from_annotation_images(p))
            menu.post(event.x_root, event.y_root)

    def _scroll_thumbnails_to(self, path, only_if_hidden=False):
        """サムネイル列を path の画像が見える位置までスクロールする"""
        try:
            self.thumbnail_frame.update_idletasks()
            total = max(1, self.thumbnail_frame.winfo_height())
            for w in self.thumbnail_frame.winfo_children():
                if getattr(w, "path", None) != path:
                    continue
                top, bottom = w.winfo_y(), w.winfo_y() + w.winfo_height()
                v0, v1 = self.thumbnail_canvas.yview()
                if only_if_hidden and v0 * total <= top and bottom <= v1 * total:
                    return
                self.thumbnail_canvas.yview_moveto(max(0.0, (top - 8) / total))
                return
        except Exception:
            pass

    def _append_annotation_images(self, new_paths):
        """「学習用の画像」に追加する。既存の並びは変えず、新しい画像は一覧の一番下に合流させる
        （プレビューで見つけて追加した画像が、代表画像の間に紛れ込まないように）"""
        order = {p: i for i, p in enumerate(self.image_paths)}
        existing = set(self.annotation_images)
        added = [p for p in sorted(set(new_paths), key=lambda p: order.get(p, 10 ** 9)) if p not in existing]
        self.annotation_images = list(self.annotation_images) + added
        return added

    def remove_from_annotation_images(self, image_path):
        if image_path not in self.annotation_images:
            return
        self.annotation_images.remove(image_path)
        self.display_thumbnails()
        if self.current_image_path == image_path:
            if self.annotation_images:
                self.display_image(0)
            else:
                self.image_canvas.delete("all")
                self.current_image_path = None

    def add_images_from_folder(self):
        """フォルダ内の任意の画像を「学習用の画像」に追加する"""
        if not self.image_folder:
            return
        filepaths = filedialog.askopenfilenames(
            title="教師点を打つ画像を選択", initialdir=self.image_folder,
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff")])
        if not filepaths:
            return
        # 原本を選んでも、作業セット（縮小版）側のパスに対応付ける
        by_name = {os.path.basename(p): p for p in self.image_paths}
        from adaptfvc.modelio import original_name
        by_orig = {original_name(p): p for p in self.image_paths}
        selected = []
        for fp in filepaths:
            p = by_name.get(os.path.basename(fp)) or by_orig.get(os.path.basename(fp))
            if p and p not in selected:
                selected.append(p)
        if not selected:
            messagebox.showinfo("情報", "追加できる画像がありませんでした（このフォルダの画像だけ追加できます）。")
            return
        self._append_annotation_images(selected)
        self.display_thumbnails()
        self.display_image(selected[-1])
        self._scroll_thumbnails_to(selected[-1])
        self._slog("add_images", n=len(selected))

    # ==========================================================
    # 画像表示・ズーム・移動
    # ==========================================================
    def display_image(self, index_or_path, overlay_pil_image=None):
        """中央パネルに画像を表示する。推論表示 ON ならキャッシュを重ねるか裏で推論を要求する"""
        if isinstance(index_or_path, int):
            if not self.annotation_images:
                return
            self.current_image_index = index_or_path
            self.current_image_path = self.annotation_images[index_or_path]
        else:
            self.current_image_path = index_or_path
            try:
                self.current_image_index = self.annotation_images.index(index_or_path)
            except ValueError:
                self.current_image_index = -1
        self._update_nav_label()
        self._reset_seq_state()
        self.seq_image_path = self.current_image_path

        auto_infer_path = None
        if overlay_pil_image is None and self.show_inference.get() and self._model_ready():
            cached = self.infer_cache.get(self.current_image_path)
            if cached is not None:
                overlay_pil_image = cached[0]
                self.update_latest_coverage(cached[1], capped=cached[2])
            else:
                auto_infer_path = self.current_image_path
        if overlay_pil_image is None:
            self.update_latest_coverage(None)
        self.current_overlay_pil = overlay_pil_image

        path = self.current_image_path
        self.pil_image = Image.open(path).convert("RGBA")
        w, h = self.pil_image.size
        canvas_w, canvas_h = self.image_canvas.winfo_width(), self.image_canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50:
            canvas_w, canvas_h = self.display_w, self.display_h
        ratio = min(canvas_w / w, canvas_h / h)
        self.display_w, self.display_h = max(1, int(w * ratio)), max(1, int(h * ratio))
        base_img = self.pil_image.resize((self.display_w, self.display_h), Image.Resampling.LANCZOS)
        if overlay_pil_image is not None:
            try:
                ov = overlay_pil_image.resize((self.display_w, self.display_h), Image.Resampling.NEAREST)
                base_img = Image.alpha_composite(base_img, ov)
            except Exception as e:
                print(f"オーバーレイ表示エラー: {e}")

        if self.zoom != 1.0 or self.pan_x or self.pan_y:
            zw, zh = max(1, int(self.display_w * self.zoom)), max(1, int(self.display_h * self.zoom))
            zoomed = base_img.resize((zw, zh), Image.Resampling.LANCZOS)
            self._clamp_pan()
            visible = zoomed.crop((self.pan_x, self.pan_y, self.pan_x + self.display_w, self.pan_y + self.display_h))
        else:
            visible = base_img

        # 画像右上にファイル名
        try:
            draw = ImageDraw.Draw(visible)
            fname = os.path.basename(path)
            size = int(max(14, min(visible.width, visible.height) * 0.04))
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except Exception:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), fname, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = max(4, size // 6)
            x1, y0 = visible.width - pad, pad
            x0, y1 = max(0, x1 - tw - 2 * pad), y0 + th + pad
            draw.rectangle([x0, y0, x1, y1], fill="white")
            draw.text((x0 + pad, y0 + pad // 2), fname, fill="black", font=font)
        except Exception:
            pass

        self.display_pil_image = visible
        self.tk_image = ImageTk.PhotoImage(visible)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.redraw_markers()
        try:
            self.image_canvas.focus_set()
        except Exception:
            pass
        if auto_infer_path is not None:
            self._request_inference(auto_infer_path, auto=True)

    def _update_nav_label(self):
        total = len(self.annotation_images)
        if self.current_image_index >= 0 and total > 0:
            self.nav_label.config(text=f"{self.current_image_index + 1} / {total}")
        else:
            self.nav_label.config(text=f"- / {total}" if total else "- / -")

    def navigate_prev(self, source="button"):
        if not self.annotation_images:
            return
        idx = len(self.annotation_images) - 1 if self.current_image_index <= 0 else self.current_image_index - 1
        self._slog("nav", dir="prev", source=source, auto_infer=bool(self.show_inference.get()))
        self.display_image(idx)
        self._scroll_thumbnails_to(self.current_image_path, only_if_hidden=True)

    def navigate_next(self, source="button"):
        if not self.annotation_images:
            return
        idx = 0 if self.current_image_index >= len(self.annotation_images) - 1 else self.current_image_index + 1
        self._slog("nav", dir="next", source=source, auto_infer=bool(self.show_inference.get()))
        self.display_image(idx)
        self._scroll_thumbnails_to(self.current_image_path, only_if_hidden=True)

    def on_canvas_configure(self, event):
        new_size = (event.width, event.height)
        if new_size == self._last_canvas_size:
            return
        self._last_canvas_size = new_size
        if self.current_image_path:
            self.display_image(self.current_image_path, overlay_pil_image=self.current_overlay_pil)

    def _clamp_pan(self):
        zw, zh = max(1, int(self.display_w * self.zoom)), max(1, int(self.display_h * self.zoom))
        self.pan_x = min(max(0, int(self.pan_x)), max(0, zw - self.display_w))
        self.pan_y = min(max(0, int(self.pan_y)), max(0, zh - self.display_h))

    def _update_zoom_label(self):
        self.zoom_label.config(text=f"{int(self.zoom * 100)}%")

    def on_mousewheel_zoom(self, event):
        delta = getattr(event, "delta", 0)
        if delta == 0 and hasattr(event, "num"):
            delta = 120 if event.num == 4 else -120
        factor = 1.1 if delta > 0 else 1 / 1.1
        old_zoom = self.zoom
        new_zoom = max(1.0, min(4.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        cx, cy = self.pan_x + event.x, self.pan_y + event.y
        self.zoom = new_zoom
        self.pan_x = cx / old_zoom * new_zoom - event.x
        self.pan_y = cy / old_zoom * new_zoom - event.y
        self._clamp_pan()
        self._update_zoom_label()
        if self.current_image_path:
            self.display_image(self.current_image_path, overlay_pil_image=self.current_overlay_pil)

    def _fake_wheel_event(self, delta):
        class E:
            pass
        e = E()
        e.delta = delta
        e.x = max(1, self.image_canvas.winfo_width() // 2)
        e.y = max(1, self.image_canvas.winfo_height() // 2)
        return e

    def on_zoom_reset(self):
        self.zoom, self.pan_x, self.pan_y = 1.0, 0, 0
        self._update_zoom_label()
        if self.current_image_path:
            self.display_image(self.current_image_path, overlay_pil_image=self.current_overlay_pil)

    def on_pan_start(self, event):
        self._is_panning, self._pan_start = True, (event.x, event.y)

    def on_pan_move(self, event):
        if not self._is_panning:
            return
        sx, sy = self._pan_start
        self._pan_start = (event.x, event.y)
        self.pan_x += sx - event.x
        self.pan_y += sy - event.y
        self._clamp_pan()
        if self.current_image_path:
            self.display_image(self.current_image_path, overlay_pil_image=self.current_overlay_pil)

    def on_pan_end(self, event):
        self._is_panning = False

    def redraw_markers(self):
        for ann in self.annotations.get(self.current_image_path, []):
            sx = ann['nx'] * self.display_w * self.zoom - self.pan_x
            sy = ann['ny'] * self.display_h * self.zoom - self.pan_y
            if sx < 0 or sy < 0 or sx > self.display_w or sy > self.display_h:
                continue
            color = "green" if ann['class'] == "plant" else "brown"
            self.image_canvas.create_rectangle(sx - 1, sy - 1, sx + 1, sy + 1, fill=color, outline=color)

    # ==========================================================
    # 教師点
    # ==========================================================
    def _reset_seq_state(self):
        self.seq_count, self.seq_class, self.seq_last, self.seq_image_path = 0, None, None, None
        self._update_seq_progress_label()

    def _update_seq_progress_label(self):
        if not hasattr(self, "seq_progress_label"):
            return
        if self.seq_interp_enabled.get():
            self.seq_progress_label.config(text=f"連続注釈: {self.seq_count}/5")
        else:
            self.seq_progress_label.config(text="連続注釈: OFF（1クリック=1画素）")

    def set_annotation_class(self, cls, source="key"):
        """アノテーションするクラスを固定で指定する（Q=植物 / W=土壌）。トグルにすると今どちらか分からなくなるため"""
        if self.annotation_class.get() != cls:
            self.annotation_class.set(cls)
        self._slog("class_switch", cls=cls, source=source)

    def toggle_seq_interp(self, source="key"):
        self.seq_interp_enabled.set(not self.seq_interp_enabled.get())
        self._slog("seq_toggle", on=bool(self.seq_interp_enabled.get()), source=source)

    def _iter_line_points_norm(self, nx0, ny0, nx1, ny1, w, h):
        """2点（正規化座標）を結ぶ線分上の全ピクセルを列挙し、正規化座標で返す"""
        x0, y0 = int(round(nx0 * max(1, w - 1))), int(round(ny0 * max(1, h - 1)))
        x1, y1 = int(round(nx1 * max(1, w - 1))), int(round(ny1 * max(1, h - 1)))
        dx, dy = x1 - x0, y1 - y0
        steps = max(abs(dx), abs(dy))
        seen, uniq = set(), []
        for i in range(steps + 1):
            t = i / steps if steps else 0.0
            key = (int(round(x0 + dx * t)), int(round(y0 + dy * t)))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((key[0] / float(w), key[1] / float(h)))
        return uniq

    def _append_annotation_points(self, path, cls, points):
        self.annotations.setdefault(path, [])
        self.annotation_groups.setdefault(path, [])
        for i, (nx, ny) in enumerate(points):
            a = {'nx': nx, 'ny': ny, 'class': cls, 'size': 1}
            if i == len(points) - 1:
                a['click'] = True  # 実際にクリックした点（直線補間で足した点と区別し、クリック数を数える）
            self.annotations[path].append(a)
        self.annotation_groups[path].append(max(1, len(points)))
        self.update_annotation_counts()
        self.redraw_markers()
        self.display_thumbnails()
        self._schedule_autosave()
        return len(points)

    def on_image_click(self, event):
        """画像クリック時のアノテーション処理（1ピクセル＋直線自動追加、最大5クリック）"""
        selected_class = self.annotation_class.get()
        path = self.current_image_path
        if not path:
            return
        if self.seq_image_path != path:
            self._reset_seq_state()
            self.seq_image_path = path
        scaled_x, scaled_y = self.pan_x + event.x, self.pan_y + event.y
        nx = max(0.0, min(1.0, (scaled_x / max(1e-6, self.zoom)) / self.display_w))
        ny = max(0.0, min(1.0, (scaled_y / max(1e-6, self.zoom)) / self.display_h))

        if (self.seq_interp_enabled.get() and self.seq_class == selected_class
                and self.seq_count > 0 and self.seq_last is not None):
            ow, oh = self.pil_image.size
            points = self._iter_line_points_norm(self.seq_last[0], self.seq_last[1], nx, ny, ow, oh)
        else:
            points = [(nx, ny)]
            self.seq_class = selected_class

        added = self._append_annotation_points(path, selected_class, points)
        self._slog("annotate", cls=selected_class, n_points=added, interp=len(points) > 1,
                   round=self.loop_round)
        self.seq_count = min(5, self.seq_count + 1)
        self.seq_last = (nx, ny)
        self.seq_image_path = path
        self._update_seq_progress_label()
        if self.seq_count >= 5:
            self._reset_seq_state()
            self.seq_image_path = path

    def update_annotation_counts(self):
        n_plant = sum(1 for anns in self.annotations.values() for a in anns if a.get('class') == 'plant')
        n_soil = sum(1 for anns in self.annotations.values() for a in anns if a.get('class') != 'plant')
        c_plant, c_soil = count_clicks(self.annotations)
        n_imgs = sum(1 for anns in self.annotations.values() if anns)
        self.plant_count_label.config(text=f"🌱 植物: {c_plant} クリック（{n_plant} 画素）")
        self.soil_count_label.config(text=f"🟫 土壌: {c_soil} クリック（{n_soil} 画素）")
        self.image_annot_stats_label.config(text=f"教師点を打った画像: {n_imgs} 枚")

    def undo_last_annotation(self, source="button"):
        path = self.current_image_path
        if not path or not self.annotations.get(path):
            return
        groups = self.annotation_groups.get(path, [])
        n = max(1, int(groups.pop())) if groups else 1
        ann_list = self.annotations[path]
        removed = 0
        for _ in range(min(n, len(ann_list))):
            ann_list.pop()
            removed += 1
        self._slog("undo", n_points=removed, source=source)
        self.update_annotation_counts()
        self.display_thumbnails()
        self._schedule_autosave()
        self.display_image(path)

    # ==========================================================
    # 推論表示（Space トグル・自動推論）
    # ==========================================================
    def _model_ready(self):
        return self.model is not None

    def _on_model_changed(self):
        self.infer_cache = {}
        self._model_generation += 1
        self._infer_pending = None

    def toggle_inference_display(self, source="key"):
        self.show_inference.set(not self.show_inference.get())
        self._apply_inference_display(source=source)

    def _apply_inference_display(self, source="button"):
        on = bool(self.show_inference.get())
        self._slog("inference_display", on=on, source=source)
        if not self.current_image_path:
            self._update_infer_status()
            return
        self.display_image(self.current_image_path)

    def update_latest_coverage(self, coverage, capped=False):
        if coverage is None:
            text = "最新の植被率: -"
        else:
            text = f"最新の植被率: {coverage:.2f}%" + ("（縮小推定）" if capped else "")
        self.latest_coverage_label.config(text=text)
        self._update_infer_status(coverage, capped)

    def _update_infer_status(self, coverage=None, capped=False, busy=False):
        lbl = getattr(self, "infer_status_label", None)
        if lbl is None:
            return
        if busy:
            lbl.config(text="推論中...", fg=UI_COLORS["accent"])
        elif not self.show_inference.get():
            lbl.config(text="推論表示: OFF (Space)", fg=UI_COLORS["note"])
        elif coverage is None:
            lbl.config(text="推論表示: ON (Space)" + ("" if self._model_ready() else "  モデル未学習"),
                       fg=UI_COLORS["accent"])
        else:
            lbl.config(text=f"推論表示: ON  植被率 {coverage:.1f}%" + ("（縮小推定）" if capped else ""),
                       fg=UI_COLORS["accent"])

    def _update_model_status(self):
        lbl = getattr(self, "model_status_label", None)
        if lbl is None:
            return
        if self.model is None:
            lbl.config(text="モデル: 未学習（教師点を各クラス 10 点以上打って T）")
        else:
            mi = self.model_info
            src = {"train": "学習", "restore": "前回分を復元", "load": "読込"}.get(mi.get("source"), "")
            when = mi.get("at", "")
            saved = "（自動保存済み）" if mi.get("saved") else ""
            lbl.config(text=f"モデル: {src} {when}  植物 {mi.get('n_plant', '?')} / 土壌 {mi.get('n_soil', '?')} 画素 {saved}")

    def _generate_overlay(self, image_path, max_long_side=None, model=None):
        """1枚を推論してオーバーレイ画像と植被率を返す（別スレッドから呼ばれる。tk には触らない）"""
        model = model if model is not None else self.model
        img_bgr, valid = load_image(image_path)
        h0, w0 = img_bgr.shape[:2]
        capped = False
        if max_long_side is not None and max(h0, w0) > max_long_side:
            scale = max_long_side / float(max(h0, w0))
            size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
            img_bgr = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
            if valid is not None:
                valid = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
            capped = True
        mask = segment_image(img_bgr, model, valid)
        coverage = calculate_coverage(mask, valid)
        if coverage is None:
            raise ValueError("有効画素がありません（全面が透明）")
        if capped:
            mask = cv2.resize(mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
        return create_overlay_from_mask(mask), coverage, capped

    def _request_inference(self, path, auto=False, force=False):
        """推論を要求する。推論中なら「最新の要求」だけ覚えて終わり次第まわす"""
        if not force:
            cached = self.infer_cache.get(path)
            if cached is not None:
                if path == self.current_image_path:
                    self.display_image(path, overlay_pil_image=cached[0])
                    self.update_latest_coverage(cached[1], capped=cached[2])
                return
        if self._infer_running is not None:
            self._infer_pending = path
            return
        self._start_inference(path, auto=auto)

    def _start_inference(self, path, auto=False):
        self._infer_running = path
        self._infer_result = None
        generation = self._model_generation
        model = self.model
        self._update_infer_status(busy=True)

        def worker():
            t0 = time.time()
            try:
                overlay_img, coverage, capped = self._generate_overlay(
                    path, max_long_side=self.INFER_MAX_LONG_SIDE, model=model)
                err = None
            except Exception as e:
                overlay_img, coverage, capped, err = None, None, False, e
            self._infer_result = (overlay_img, coverage, capped, err, time.time() - t0)

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            if self._infer_result is None:
                self.root.after(100, poll)
                return
            overlay_img, coverage, capped, err, dt = self._infer_result
            self._infer_result = None
            self._infer_running = None
            stale = (generation != self._model_generation)
            if err is not None:
                self.update_latest_coverage(None)
                self.infer_status_label.config(text=f"推論エラー: {err}", fg=UI_COLORS["warn"])
                print(f"推論エラー ({os.path.basename(path)}): {err}")
            elif not stale:
                self.infer_cache[path] = (overlay_img, coverage, capped)
                if path == self.current_image_path and self.show_inference.get():
                    self.display_image(path, overlay_pil_image=overlay_img)
                    self.update_latest_coverage(coverage, capped=capped)
                else:
                    self._update_infer_status()
                self._slog("infer", image=os.path.basename(path), duration_s=round(dt, 2),
                           coverage=round(coverage, 2), capped=capped, auto=auto, round=self.loop_round)
            pending, self._infer_pending = self._infer_pending, None
            if (pending is not None and self.show_inference.get() and pending not in self.infer_cache
                    and pending == self.current_image_path):
                self._start_inference(pending, auto=True)

        self.root.after(100, poll)

    # ==========================================================
    # 学習・自動保存・モデル読込
    # ==========================================================
    def train_model_ui(self, source="button"):
        if getattr(self, "_training", False):
            return
        self._training = True
        try:
            t0 = time.time()
            if self.train_model():
                self._slog("train", duration_s=round(time.time() - t0, 2), source=source,
                           round=self.loop_round, **{k: v for k, v in self.model_info.items()
                                                     if k in ("n_plant", "n_soil", "n_images", "n_skipped_invalid")})
                self.show_inference.set(True)
                self._apply_inference_display(source="train")
        finally:
            self._training = False

    def train_model(self):
        """教師点から Random Forest を学習する（Lab 固定・seed 固定）"""
        if not self.annotations:
            messagebox.showwarning("警告", "教師点がありません。画像をクリックして、植物と土壌の教師点を打ってください。")
            return False
        try:
            self.root.config(cursor="watch")
            self.root.update()
        except Exception:
            pass
        try:
            X, y, st = extract_training_pixels(self.annotations)
            if st["n_plant"] < 10 or st["n_soil"] < 10:
                messagebox.showwarning("警告", f"各クラスに 10 画素以上の教師点が必要です。\n"
                                               f"植物: {st['n_plant']} 画素, 土壌: {st['n_soil']} 画素"
                                       + (f"\n（透明な画素に打たれた {st['n_skipped_invalid']} 画素は除外）"
                                          if st["n_skipped_invalid"] else ""))
                return False
            self.model = train_rf(X, y)
        except Exception as e:
            messagebox.showerror("エラー", f"学習に失敗しました:\n{e}")
            return False
        finally:
            try:
                self.root.config(cursor="")
            except Exception:
                pass
        self.loop_round += 1
        self.model_annotations = annotations_to_portable(self.annotations)
        self.annotations_at_model = self.model_annotations
        self.model_info = {"source": "train", "at": datetime.now().strftime("%H:%M:%S"),
                           "n_plant": st["n_plant"], "n_soil": st["n_soil"], "n_images": st["n_images"],
                           "n_skipped_invalid": st["n_skipped_invalid"], "saved": False}
        self._on_model_changed()
        self._autosave_model()
        self._update_model_status()
        return True

    def _models_dir(self):
        return self._app_dir("models", make=True)

    def _autosave_extra(self):
        return {"image_folder": self.image_folder, "resize_long_side": self.cache_long_side,
                "n_images": len(self.image_paths), "n_annotation_images": len(self.annotation_images)}

    def _schedule_autosave(self):
        """教師点の自動保存（クリックごと。連打に備えて少し待ってからまとめて書く）"""
        self._cancel_autosave()
        try:
            self._autosave_after_id = self.root.after(self.AUTOSAVE_DELAY_MS, self._autosave_annotations_now)
        except Exception:
            pass

    def _cancel_autosave(self):
        if self._autosave_after_id is not None:
            try:
                self.root.after_cancel(self._autosave_after_id)
            except Exception:
                pass
            self._autosave_after_id = None

    def _autosave_annotations_now(self):
        self._autosave_after_id = None
        if not self.image_folder:
            return
        try:
            portable = annotations_to_portable(self.annotations)
            save_annotations_json(os.path.join(self._models_dir(), "annotations_latest.json"),
                                  portable, self._autosave_extra())
        except Exception as e:
            print(f"教師点の自動保存に失敗: {e}")

    def _autosave_model(self):
        """学習成功の瞬間にモデルと教師点を保存する（保存ボタンの押し忘れを無くす）"""
        try:
            portable = annotations_to_portable(self.annotations)
            trained = self.model_annotations if self.model_annotations is not None else portable
            mdir = self._models_dir()
            save_model_bundle(os.path.join(mdir, "model_latest.pkl"), self.model, trained, self._autosave_extra())
            save_annotations_json(os.path.join(mdir, "annotations_latest.json"), portable, self._autosave_extra())
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_annotations_json(os.path.join(mdir, "history", f"annotations_{ts}.json"), trained,
                                  self._autosave_extra())
            self.model_info["saved"] = True
            self._slog("autosave", what="model+annotations")
        except Exception as e:
            print(f"モデルの自動保存に失敗: {e}")

    def _offer_restore_autosave(self):
        """起動時: このフォルダに前回の教師点があれば復元を提案する"""
        ann_path = os.path.join(self._app_dir("models"), "annotations_latest.json")
        if not os.path.exists(ann_path):
            return
        try:
            loaded = load_model_file(ann_path)
        except Exception:
            return
        portable = loaded["annotations"]
        n_plant, n_soil = count_points(portable)
        if n_plant + n_soil == 0:
            return
        if not messagebox.askyesno("前回の続き",
                                   f"このフォルダには前回の教師点が保存されています。\n"
                                   f"植物 {n_plant} 画素 / 土壌 {n_soil} 画素（{len(portable)} 枚）\n\n"
                                   "復元しますか？（モデルも保存されていれば一緒に読み込みます）"):
            self._slog("autosave_restore", accepted=False)
            return
        # 教師点は常に自動保存の JSON（最後にクリックした時点）から戻す。保存済みモデルは、
        # 同梱の教師点が JSON と同じとき（＝学習のあとに教師点を変えていないとき）だけ使い、
        # 違えば JSON の教師点から学習し直す。以前は学習後に打った教師点が復元で失われていた
        model_path = os.path.join(self._app_dir("models"), "model_latest.pkl")
        if os.path.exists(model_path):
            try:
                bundle = load_model_file(model_path)
                if bundle.get("annotations") == portable:
                    loaded = bundle
                else:
                    print("学習のあとに教師点が変わっているため、教師点から学習し直します")
            except Exception as e:
                print(f"model_latest.pkl を読めません（教師点から学習し直します）: {e}")
        self._apply_loaded(loaded, source="restore", replace=True, quiet=True)
        self._slog("autosave_restore", accepted=True, n_plant=n_plant, n_soil=n_soil)

    def load_model_dialog(self):
        path = filedialog.askopenfilename(
            title="モデル / 教師点を読み込む", initialdir=self._app_dir("models") if self.image_folder else None,
            filetypes=[("adaptFVC モデル・教師点", "*.pkl;*.json"), ("すべて", "*.*")])
        if not path:
            return
        try:
            loaded = load_model_file(path)
        except Exception as e:
            messagebox.showerror("エラー", f"読み込みに失敗しました:\n{e}")
            return
        replace = True
        if any(self.annotations.values()):
            n = sum(len(v) for v in self.annotations.values())
            replace = messagebox.askyesno("教師点の扱い",
                                          f"現在の教師点（{n} 画素）があります。\n"
                                          "「はい」= 読み込んだ教師点で置き換える\n"
                                          "「いいえ」= 現在の教師点に追加する")
        self._apply_loaded(loaded, source="load", replace=replace, quiet=False, file=path)

    def _apply_loaded(self, loaded, source, replace=True, quiet=False, file=None):
        """読み込んだモデル/教師点を現在のフォルダに適用する。
        教師点はファイル名で現フォルダの画像に対応付け、モデルがこの環境で使えなければ学習し直す"""
        portable = loaded.get("annotations", {})
        annotations, matched, unmatched = portable_to_annotations(portable, self.image_paths)
        if replace:
            self.annotations, self.annotation_groups = {}, {}
        for p, pts in annotations.items():
            self.annotations.setdefault(p, []).extend(pts)
            self.annotation_groups.setdefault(p, []).extend(click_groups(pts))
        self._append_annotation_images(annotations.keys())

        ok, reason = model_is_usable(loaded)
        notes = []
        if unmatched:
            notes.append(f"このフォルダに無い画像の教師点 {len(unmatched)} 枚分は読み飛ばしました。")
        if ok:
            self.model = loaded["model"]
            self.model_annotations = portable
            self.annotations_at_model = annotations_to_portable(self.annotations)
            n_plant, n_soil = count_points(portable)
            self.model_info = {"source": source, "at": datetime.now().strftime("%H:%M:%S"),
                               "n_plant": n_plant, "n_soil": n_soil, "saved": source == "restore"}
            self._on_model_changed()
            if source == "load":
                self._autosave_model()
        else:
            self.model = None
            self.model_annotations = self.annotations_at_model = None
            self._on_model_changed()
            X, y, st = extract_training_pixels(self.annotations)
            if st["n_plant"] >= 10 and st["n_soil"] >= 10:
                try:
                    self.root.config(cursor="watch")
                    self.root.update()
                    self.model = train_rf(X, y)
                finally:
                    self.root.config(cursor="")
                self.loop_round += 1
                self.model_annotations = annotations_to_portable(self.annotations)
                self.annotations_at_model = self.model_annotations
                self.model_info = {"source": source, "at": datetime.now().strftime("%H:%M:%S"),
                                   "n_plant": st["n_plant"], "n_soil": st["n_soil"], "n_images": st["n_images"],
                                   "n_skipped_invalid": st["n_skipped_invalid"], "saved": False}
                self._on_model_changed()
                self._autosave_model()
                if loaded.get("model") is not None:
                    notes.append(f"モデル本体はそのまま使えないため（{reason}）、教師点から学習し直しました。")
            else:
                notes.append(f"教師点を取り込みましたが、学習には各クラス 10 画素以上必要です"
                             f"（植物 {st['n_plant']} / 土壌 {st['n_soil']}）。")
        self._slog("model_load", source=source, kind=loaded.get("kind"), usable=ok,
                   n_matched=len(matched), n_unmatched=len(unmatched), file=os.path.basename(file) if file else None)
        self.update_annotation_counts()
        self.display_thumbnails()
        self._update_model_status()
        self._schedule_autosave()
        if self.model is not None:
            self.show_inference.set(True)
        if self.current_image_path:
            self.display_image(self.current_image_path)
        if not quiet:
            n_plant, n_soil = count_points(portable)
            messagebox.showinfo("読み込み完了",
                                f"教師点 植物 {n_plant} / 土壌 {n_soil} 画素（{len(matched)} 枚）を読み込みました。\n"
                                + "\n".join(notes))

    def save_model_as(self):
        if self.model is None:
            messagebox.showwarning("警告", "まだモデルが学習されていません。")
            return
        path = filedialog.asksaveasfilename(
            title="モデルを別名で保存", initialdir=self._models_dir(), defaultextension=".pkl",
            initialfile=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            filetypes=[("adaptFVC モデル", "*.pkl")])
        if not path:
            return
        try:
            portable = annotations_to_portable(self.annotations)
            save_model_bundle(path, self.model, portable, self._autosave_extra())
            save_annotations_json(os.path.splitext(path)[0] + "_annotations.json", portable, self._autosave_extra())
            self._slog("model_save", file=os.path.basename(path))
            messagebox.showinfo("保存", f"保存しました:\n{path}\n（教師点の JSON も隣に保存）")
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました:\n{e}")

    # ==========================================================
    # プレビュー・全体処理
    # ==========================================================
    def preview(self):
        if self.model is None:
            messagebox.showwarning("警告", "モデルが学習されていません。先に「学習」を実行してください。")
            return
        if self._busy:
            return
        annotated = set(self.annotation_images)
        targets = [p for p in self.image_paths if p not in annotated]
        if not targets:
            messagebox.showinfo("情報", "学習に使っていない画像がありません。")
            return
        if len(targets) > 10:
            targets = random.sample(targets, 10)
        self.preview_count += 1
        output_folder = self._app_dir("previews", f"preview_{self.preview_count}")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        self._busy = True
        t0 = time.time()
        try:
            results = run_batch(targets, self.model, output_folder, progress=self._show_batch_progress,
                                title="プレビュー")
        except Exception as e:
            messagebox.showerror("エラー", f"プレビューに失敗しました:\n{e}")
            return
        finally:
            self._busy = False
            self._show_batch_progress("")
        self._slog("preview_round", round=self.loop_round, n_images=len(targets),
                   duration_s=round(time.time() - t0, 1))
        self.review_results(output_folder, results)

    def review_results(self, result_folder, results=None):
        """プレビュー結果を並べ、選んだ画像を「学習用の画像」に追加する"""
        result_paths = sorted(glob.glob(os.path.join(result_folder, "*")))
        if not result_paths:
            messagebox.showinfo("情報", "レビュー対象の画像がありません。")
            return
        win = tk.Toplevel(self.root)
        win.title("プレビュー（うまく分類できていない画像を選んで「学習用の画像に追加」）")
        # 画面の大きさに合わせて窓を決め、必ず画面内に置く（下のボタンが画面外に出ないように）
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        ww, wh = min(1000, sw - 40), min(750, sh - 120)
        win.geometry(f"{ww}x{wh}+{max(0, (sw - ww) // 2)}+{max(0, min(40, sh - wh - 80))}")
        win.minsize(500, 300)
        # ボタン列を先に pack して高さを確保する（後から pack すると窓が小さいとき潰れる）
        bottom = tk.Frame(win)
        bottom.pack(fill="x", side="bottom", pady=8)
        top = tk.Frame(win)
        top.pack(fill="both", expand=True, padx=5, pady=5)
        canvas = tk.Canvas(top)
        sb = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        win.bind("<MouseWheel>", lambda e: canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        selected = {}
        cov_by_path = {r[0]: r[1] for r in (results or [])}

        def add_selected():
            if not selected:
                messagebox.showinfo("情報", "画像が選択されていません。")
                return
            # 追加した画像はサムネイル列の一番下に合流する（既存の並びは変えない）
            self._append_annotation_images(selected)
            self.display_thumbnails()
            self.show_inference.set(True)
            self.display_image(list(selected)[-1])
            self._slog("preview_add", n=len(selected))
            win.destroy()
            # 右のサムネイル列を、追加した画像（一番下）までスクロールする
            self._scroll_thumbnails_to(list(selected)[0])

        tk.Button(bottom, text="学習用の画像に追加", command=add_selected, font=("Helvetica", 12)).pack(side="left", expand=True, padx=20)
        tk.Button(bottom, text="閉じる", command=win.destroy, font=("Helvetica", 12)).pack(side="right", expand=True, padx=20)

        by_basename = {os.path.basename(p): p for p in self.image_paths}
        for i, res_path in enumerate(result_paths):
            original = by_basename.get(os.path.basename(res_path))
            if original is None or not os.path.exists(original):
                continue
            try:
                img = Image.open(res_path)
                img.thumbnail((180, 180))
                tk_img = ImageTk.PhotoImage(img)
            except Exception:
                continue
            var = tk.BooleanVar()

            def on_check(p=original, v=var):
                if v.get():
                    selected[p] = True
                else:
                    selected.pop(p, None)
            cell = tk.Frame(frame)
            chk = tk.Checkbutton(cell, image=tk_img, variable=var, command=on_check,
                                 indicatoron=False, selectcolor="yellow", bd=3)
            chk.image = tk_img
            chk.pack()
            cov = cov_by_path.get(original)
            tk.Label(cell, text=f"{os.path.basename(original)}  " + (f"{cov:.1f}%" if cov is not None else "エラー"),
                     font=("Yu Gothic UI", 8)).pack()
            cell.grid(row=i // 5, column=i % 5, padx=5, pady=5)

    def process_all(self):
        if self.model is None:
            messagebox.showwarning("警告", "モデルが学習されていません。先に「学習」を実行してください。")
            return
        if self._busy:
            return
        current = annotations_to_portable(self.annotations)
        if self.annotations_at_model is not None and current != self.annotations_at_model:
            # 古いモデルのまま処理すると、結果フォルダの教師点とモデルが食い違う（教師点から作り直せなくなる）
            if not messagebox.askyesno(
                    "教師点が変わっています",
                    "最後に学習したあとで、教師点が追加または削除されています。\n"
                    "いまの教師点で学習し直してから、全画像を処理しますか？\n\n"
                    "「はい」= 学習し直して処理する\n「いいえ」= 処理をやめる"):
                return
            self.train_model_ui(source="process_all")
            if self.annotations_at_model != current:
                return  # 学習できなかった（教師点の不足など。警告は train_model が出す）
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._app_dir("results", f"run_{ts}", make=True)
        self._busy = True
        try:
            trained = self.model_annotations if self.model_annotations is not None else current
            out = run_full(self.image_folder, self.image_paths, self.model,
                           trained, run_dir,
                           progress=self._show_batch_progress, extra_meta=self._autosave_extra())
        except Exception as e:
            messagebox.showerror("致命的なエラー", f"全画像の処理中に予期せぬエラーが発生しました:\n{e}")
            return
        finally:
            self._busy = False
            self._show_batch_progress("")
        results = out["results"]
        errors = [r[2] for r in results if r[2] is not None]
        self._slog("batch", n_images=len(self.image_paths), n_errors=len(errors),
                   duration_s=out["duration_s"], round=self.loop_round)
        if errors:
            messagebox.showwarning("警告", f"{len(errors)} 件のエラーがありました。Excel の「エラー」列を確認してください。\n"
                                          + "\n".join(errors[:5]))
        if not self.batch_mode:
            messagebox.showinfo("完了",
                                f"全画像の処理が完了しました（{len(results) - len(errors)}/{len(results)} 枚, {out['duration_s']}s）。\n\n"
                                f"結果フォルダ: {run_dir}\n"
                                f"Excel: {os.path.basename(out['excel'])}\n"
                                f"モデル: {os.path.basename(out['model'])}")

    # ==========================================================
    # その他
    # ==========================================================
    def open_output_folder(self):
        d = self._app_dir(make=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as e:
            messagebox.showinfo("出力フォルダ", f"{d}\n({e})")

    def reset_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("リセットの選択")
        dlg.grab_set()
        tk.Label(dlg, text="何をリセットしますか？", font=("Helvetica", 12, "bold")).pack(padx=10, pady=10)
        v_all = tk.BooleanVar(value=True)
        v_cur = tk.BooleanVar(value=False)
        v_model = tk.BooleanVar(value=False)
        v_results = tk.BooleanVar(value=False)
        tk.Checkbutton(dlg, text="教師点をすべて削除（このフォルダ）", variable=v_all).pack(anchor=tk.W, padx=15)
        tk.Checkbutton(dlg, text="この画像の教師点だけ削除", variable=v_cur).pack(anchor=tk.W, padx=15)
        tk.Checkbutton(dlg, text="学習済みモデルを削除", variable=v_model).pack(anchor=tk.W, padx=15)
        tk.Checkbutton(dlg, text="解析結果（adaptFVC/results・previews）を削除", variable=v_results).pack(anchor=tk.W, padx=15)
        self._note_label(dlg, "教師点を削除すると、自動保存の教師点（models/annotations_latest.json）も空になります。"
                         "学習ごとの履歴（models/history/）と縮小キャッシュは残ります。").pack(anchor=tk.W, padx=15, pady=(4, 0))
        btns = tk.Frame(dlg)
        btns.pack(fill=tk.X, pady=10)

        def do_reset():
            if v_all.get():
                self.annotations, self.annotation_groups = {}, {}
            elif v_cur.get() and self.current_image_path:
                self.annotations.pop(self.current_image_path, None)
                self.annotation_groups.pop(self.current_image_path, None)
            if v_model.get():
                self.model, self.model_info = None, {}
                self.model_annotations = self.annotations_at_model = None
                self._on_model_changed()
                self.show_inference.set(False)
            if v_results.get() and self.image_folder:
                for sub in ("results", "runs", "previews"):  # runs は v3.1 以前の名前
                    d = self._app_dir(sub)
                    if os.path.exists(d):
                        shutil.rmtree(d, ignore_errors=True)
            self._slog("reset", annotations=bool(v_all.get() or v_cur.get()), model=bool(v_model.get()),
                       results=bool(v_results.get()))
            self.update_annotation_counts()
            self.display_thumbnails()
            self._update_model_status()
            self._schedule_autosave()
            if self.current_image_path:
                self.display_image(self.current_image_path)
            dlg.destroy()

        tk.Button(btns, text="実行", command=do_reset).pack(side=tk.LEFT, expand=True, padx=10)
        tk.Button(btns, text="キャンセル", command=dlg.destroy).pack(side=tk.RIGHT, expand=True, padx=10)

    def show_shortcuts(self):
        text = "\n".join(f"{k:<16} {v}" for k, v in SHORTCUTS)
        messagebox.showinfo("ショートカット一覧", text)

    def show_about(self):
        from adaptfvc.modelio import environment_info
        env = environment_info()
        messagebox.showinfo("adaptFVC",
                            f"adaptFVC v{__version__}\n植被率（FVC）対話型セグメンテーションツール\n\n"
                            f"学習器: Random Forest（Lab 色空間、seed 固定）\n"
                            f"Python {env['python']} / scikit-learn {env['sklearn']} / numpy {env['numpy']} / OpenCV {env['opencv']}\n\n"
                            "このアプリは通信を行いません。出力はすべて画像フォルダ内の adaptFVC/ に保存されます。\n\n"
                            f"{__copyright__}\n"
                            "GNU General Public License v3.0 以降の条件で再配布・改変できます。無保証です。\n"
                            f"ソースコードとライセンス全文: {__url__}")

    def _on_close(self):
        self._cancel_autosave()
        try:
            if self.image_folder:
                self._autosave_annotations_now()
            self._slog("session_end")
            self.session_logger.close()
        except Exception:
            pass
        self.root.destroy()
