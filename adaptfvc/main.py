# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""アプリのエントリポイント。

PyInstaller で凍結した場合と `python -m adaptfvc` の両方から呼ばれる。
multiprocessing を使うため、freeze_support() を必ず最初に呼ぶこと。

使い方:
    adaptFVC                                  GUI を起動
    adaptFVC --batch <画像フォルダ> --model <model.pkl | annotations.json>
                                              GUI なしで全画像の処理（配布版の動作確認・自動化用）
"""

import argparse
import multiprocessing
import os
import sys


def _icon_path():
    """凍結時は _MEIPASS 配下、開発時はリポジトリの assets/ から探す"""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    path = os.path.join(base, "assets", "app.ico")
    return path if os.path.exists(path) else None


def run_batch_cli(image_folder, model_file, out_dir=None):
    """GUI を使わずに全画像の処理を行う。モデルが今の環境で使えなければ同梱の教師点から学習し直す"""
    from datetime import datetime
    from adaptfvc import imagecache
    from adaptfvc.batch import run_full
    from adaptfvc.core import extract_training_pixels, train_rf
    from adaptfvc.modelio import load_model_file, model_is_usable, portable_to_annotations

    image_folder = os.path.abspath(image_folder)
    paths = imagecache.discover_images(image_folder)
    if not paths:
        print(f"画像が見つかりません: {image_folder}")
        return 2
    imagecache.migrate_legacy_outputs(image_folder)
    long_side = imagecache.load_resize_choice(image_folder)
    if long_side is not None:
        paths, stats = imagecache.prepare_resized_cache(
            image_folder, paths, long_side, progress=lambda s: print(s) if s else None)
        print(f"縮小キャッシュ: 長辺{long_side}px, 対象{stats['n_large']}枚 (新規{stats['n_created']})")

    loaded = load_model_file(model_file)
    portable = loaded["annotations"]
    ok, reason = model_is_usable(loaded)
    if ok:
        model = loaded["model"]
        print(f"モデルをそのまま使用: {model_file}")
    else:
        if not portable:
            print(f"モデルが使えず、教師点もありません: {reason}")
            return 2
        annotations, matched, unmatched = portable_to_annotations(portable, paths)
        print(f"教師点から学習し直します（{reason}）: 対応 {len(matched)} 枚 / 未対応 {len(unmatched)} 枚")
        X, y, st = extract_training_pixels(annotations)
        if st["n_plant"] < 10 or st["n_soil"] < 10:
            print(f"教師点が足りません: 植物 {st['n_plant']} / 土壌 {st['n_soil']}")
            return 2
        model = train_rf(X, y)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir or imagecache.app_dir(image_folder, "results", f"run_{ts}")
    out = run_full(image_folder, paths, model, portable, run_dir,
                   progress=lambda s: print(s, end="\r") if s else print(),
                   extra_meta={"resize_long_side": long_side, "source_model_file": model_file})
    n_ok = sum(1 for r in out["results"] if r[2] is None)
    print(f"完了: {n_ok}/{len(paths)} 枚 ({out['duration_s']}s)\n結果: {run_dir}")
    # windowed（console=False）でビルドした exe にはコンソールが無く、print は捨てられる。
    # その場合は同じ内容を結果フォルダの batch_log.txt に残す
    import io
    if isinstance(sys.stdout, io.StringIO):
        try:
            with open(os.path.join(run_dir, "batch_log.txt"), "w", encoding="utf-8") as f:
                f.write(sys.stdout.getvalue())
        except Exception:
            pass
    return 0 if n_ok == len(paths) else 1


def run(argv=None):
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="adaptFVC - 植被率 対話型セグメンテーション")
    parser.add_argument("--batch", metavar="FOLDER", help="GUI なしで FOLDER 内の全画像を処理する")
    parser.add_argument("--model", metavar="FILE", help="--batch で使うモデル (.pkl) または教師点 (.json)")
    parser.add_argument("--out", metavar="DIR", help="--batch の出力先（省略時は FOLDER/adaptFVC/results/run_<日時>）")
    args = parser.parse_args(argv)

    if args.batch:
        if not args.model:
            parser.error("--batch には --model が必要です")
        return run_batch_cli(args.batch, args.model, args.out)

    import tkinter as tk
    from adaptfvc.gui import AnnotationApp

    root = tk.Tk()
    icon = _icon_path()
    if icon:
        try:
            root.iconbitmap(icon)
        except Exception:
            pass
    AnnotationApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(run())
