# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""全画像の処理（バッチ）: 並列推論・Excel レポート・実行メタデータ。GUI と CLI の両方が使う。

実行1回 = 1フォルダ (adaptFVC/results/run_<日時>/):
    overlays/                      判定を重ねた確認用画像
    vegetation_coverage_<日時>.xlsx 植被率一覧（シート「結果」「モデル情報」）
    model_<日時>.pkl               使用モデル（教師点・設定・環境を同梱したバンドル）
    annotations_<日時>.json        教師点（人が読める形）
    run_metadata.json              解析条件・環境・教師点概要
"""

import json
import multiprocessing
import os
import time
from datetime import datetime
from functools import partial

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from adaptfvc import __version__
from adaptfvc.core import ALGO, FEATURE_MODE
from adaptfvc.modelio import (
    count_points,
    environment_info,
    original_name,
    save_annotations_json,
    save_model_bundle,
    training_config,
)
from adaptfvc.workers import _init_worker, process_image_v3


SEQUENTIAL_MAX = 12   # これ以下の枚数はプロセスを起動せず、その場で順に処理する


def run_batch(image_paths, model, output_folder, progress=None, title="全画像の処理"):
    """画像リストを推論し、判定画像を output_folder に保存する。

    枚数が少ないときは単一プロセスで順に処理する。Windows の spawn 方式ではワーカーごとに
    numpy / OpenCV / scikit-learn の import からやり直すため、プロセス起動だけで数十秒かかり、
    数枚のプレビューではその場で回したほうがはるかに速い。

    Returns:
        [(image_path, coverage, error, valid_fraction_percent), ...]
    """
    os.makedirs(output_folder, exist_ok=True)
    num_images = len(image_paths)
    if num_images == 0:
        return []
    results = []
    if num_images <= SEQUENTIAL_MAX:
        _init_worker(model)
        for p in image_paths:
            results.append(process_image_v3(p, output_folder))
            if progress:
                progress(f"{title}: {len(results)}/{num_images}")
        if progress:
            progress("")
        return results
    num_cores = max(1, min(multiprocessing.cpu_count() - 1, num_images))
    chunksize = max(1, num_images // (num_cores * 4)) if num_images >= 20 else 1
    task = partial(process_image_v3, output_folder=output_folder)
    with multiprocessing.Pool(processes=num_cores, initializer=_init_worker, initargs=(model,)) as pool:
        it = (pool.imap_unordered(task, image_paths, chunksize=chunksize)
              if num_images >= 20 else pool.imap_unordered(task, image_paths))
        for result in it:
            results.append(result)
            if progress:
                progress(f"{title}: {len(results)}/{num_images}")
    if progress:
        progress("")
    # imap_unordered は完了順に返すので、入力順（ファイル名順）に並べ直す。
    # Excel の行順が実行ごとに変わらず、結果の突き合わせがしやすい
    order = {p: i for i, p in enumerate(image_paths)}
    results.sort(key=lambda r: order.get(r[0], len(order)))
    return results


def _annotation_summary(portable):
    n_plant, n_soil = count_points(portable)
    imgs_p = sum(1 for pts in portable.values() if any(a.get("class") == "plant" for a in pts))
    imgs_s = sum(1 for pts in portable.values() if any(a.get("class") != "plant" for a in pts))
    return {
        "plant_points": n_plant, "soil_points": n_soil, "total_points": n_plant + n_soil,
        "images_with_plant": imgs_p, "images_with_soil": imgs_s, "images_with_any": len(portable),
    }


def write_excel_report(results, excel_path, info_rows):
    """Excelレポート。シート1=結果（数値は数値セル）、シート2=モデル情報"""
    wb = Workbook()
    ws = wb.active
    ws.title = "結果"
    headers = ["No.", "画像ファイル名", "処理に使った画像のパス", "植被率(%)", "有効画素率(%)", "ステータス", "エラー"]
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.fill = header_fill
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for idx, (image_path, coverage, err, vf) in enumerate(results, 1):
        ws.cell(row=idx + 1, column=1, value=idx)
        ws.cell(row=idx + 1, column=2, value=original_name(image_path))
        ws.cell(row=idx + 1, column=3, value=image_path)
        ws.cell(row=idx + 1, column=4, value=None if coverage is None else round(float(coverage), 2))
        ws.cell(row=idx + 1, column=5, value=round(float(vf), 2))
        ws.cell(row=idx + 1, column=6, value="成功" if err is None else "エラー")
        ws.cell(row=idx + 1, column=7, value=err or "")
    for col, w in zip("ABCDEFG", (6, 30, 60, 12, 14, 10, 40)):
        ws.column_dimensions[col].width = w
    ok = [float(c) for _, c, e, _ in results if e is None and c is not None]
    if ok:
        r = len(results) + 3
        ws.cell(row=r, column=1, value="統計情報").font = Font(bold=True)
        for i, (label, val) in enumerate([("平均植被率(%)", sum(ok) / len(ok)),
                                          ("最大植被率(%)", max(ok)), ("最小植被率(%)", min(ok)),
                                          ("成功数", len(ok)), ("エラー数", len(results) - len(ok))], 1):
            ws.cell(row=r + i, column=1, value=label)
            ws.cell(row=r + i, column=2, value=round(val, 2) if isinstance(val, float) else val)

    ws2 = wb.create_sheet(title="モデル情報")
    ws2.cell(row=1, column=1, value="項目").font = Font(bold=True)
    ws2.cell(row=1, column=2, value="内容").font = Font(bold=True)
    for i, (k, v) in enumerate(info_rows, 2):
        ws2.cell(row=i, column=1, value=k)
        ws2.cell(row=i, column=2, value=v)
    ws2.column_dimensions["A"].width = 30
    ws2.column_dimensions["B"].width = 60
    wb.save(excel_path)
    return excel_path


def run_full(image_folder, image_paths, model, portable_annotations, run_dir,
             progress=None, extra_meta=None, title="全画像の処理"):
    """全画像の処理の一式を run_dir に書き出す。

    Returns:
        dict(results=..., excel=..., model=..., annotations=..., metadata=..., overlays=...)
    """
    timestamp = os.path.basename(run_dir).replace("run_", "") or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(run_dir, exist_ok=True)
    overlays = os.path.join(run_dir, "overlays")
    t0 = time.time()
    results = run_batch(image_paths, model, overlays, progress=progress, title=title)
    duration = round(time.time() - t0, 1)

    extra = dict(extra_meta or {})
    extra.update({"image_folder": image_folder, "timestamp": timestamp})
    model_path = os.path.join(run_dir, f"model_{timestamp}.pkl")
    save_model_bundle(model_path, model, portable_annotations, extra)
    ann_path = os.path.join(run_dir, f"annotations_{timestamp}.json")
    save_annotations_json(ann_path, portable_annotations, extra)

    summary = _annotation_summary(portable_annotations)
    n_ok = sum(1 for r in results if r[2] is None)
    env = environment_info()
    info_rows = [
        ("処理実行日時", timestamp),
        ("アプリ", f"adaptFVC v{__version__}"),
        ("解析手法", f"教師点に基づく機械学習（{ALGO}、色空間 {FEATURE_MODE}）"),
        ("学習器設定", json.dumps(training_config(), ensure_ascii=False)),
        ("モデルファイル", os.path.basename(model_path)),
        ("教師点ファイル", os.path.basename(ann_path)),
        ("縮小サイズ(長辺px)", extra.get("resize_long_side") or "原寸"),
        ("総画像数", len(image_paths)),
        ("成功数", n_ok),
        ("エラー数", len(results) - n_ok),
        ("植物の教師点数", summary["plant_points"]),
        ("土壌の教師点数", summary["soil_points"]),
        ("教師点を打った画像数", summary["images_with_any"]),
        ("透明画素の扱い", "アルファ<128 の画素は分母・学習から除外"),
        ("scikit-learn", env["sklearn"]),
        ("numpy / OpenCV", f"{env['numpy']} / {env['opencv']}"),
        ("Python / OS", f"{env['python']} / {env['platform']}"),
    ]
    excel_path = os.path.join(run_dir, f"vegetation_coverage_{timestamp}.xlsx")
    write_excel_report(results, excel_path, info_rows)

    meta = {
        "environment": env,
        "training": training_config(),
        "data": {
            "image_folder": image_folder,
            "total_images": len(image_paths),
            "resize_long_side": extra.get("resize_long_side"),
            "n_annotation_images": extra.get("n_annotation_images"),
        },
        "processing": {
            "timestamp": timestamp,
            "duration_s": duration,
            "n_success": n_ok,
            "n_errors": len(results) - n_ok,
            "model_path": model_path,
            "annotations_path": ann_path,
            "excel_path": excel_path,
            "alpha_handling": "alpha<128 excluded from denominator and training",
        },
        "annotations_summary": summary,
    }
    meta_path = os.path.join(run_dir, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"results": results, "excel": excel_path, "model": model_path,
            "annotations": ann_path, "metadata": meta_path, "overlays": overlays,
            "duration_s": duration}
