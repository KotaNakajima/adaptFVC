# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""配布物を作る（Windows）。

1. docs/manual.html を Edge のヘッドレス印刷で docs/adaptFVC_manual_ja.pdf にする
2. 同梱ライブラリのライセンス文を、ビルドに使った Python 環境から集めて THIRD_PARTY_LICENSES.txt にする
3. dist/adaptFVC.exe・取扱説明書・はじめに.txt・LICENSE・THIRD_PARTY_LICENSES を
   release/adaptFVC_v<版>_win64/ に集め、zip にする

使い方: venv\\Scripts\\python.exe tools\\make_release.py   （exe をビルドしたのと同じ venv で実行する）
前提: pyinstaller で dist/adaptFVC.exe をビルド済み（BUILD.md）
"""
import importlib.metadata as md
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from adaptfvc import __version__  # noqa: E402

REPO_URL = "https://github.com/KotaNakajima/adaptFVC"
MANUAL_PDF_NAME = "adaptFVC_manual_ja.pdf"

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

# exe に同梱される実行時ライブラリ（requirements-app-lock.txt のうちビルド専用のものを除く）
BUNDLED_PACKAGES = [
    "numpy", "opencv-python", "scikit-learn", "scipy", "joblib", "threadpoolctl",
    "pillow", "openpyxl", "et_xmlfile", "tqdm", "colorama",
]

README_TXT = """adaptFVC v{version}  —  画像から植被率（FVC）を測る対話型セグメンテーションツール
{date}

【同梱物】
  adaptFVC.exe                  本体（インストール不要。ダブルクリックで起動）
  {manual}        使い方と原理（先にお読みください）
  LICENSE.txt                   adaptFVC のライセンス（GNU GPL v3.0 以降）
  THIRD_PARTY_LICENSES.txt      本体に同梱しているライブラリのライセンス

【はじめて起動するとき】
  Windows SmartScreen の「WindowsによってPCが保護されました」が出たら、
  「詳細情報」→「実行」を押してください（署名の無い配布アプリのため）。
  起動後、解析したい画像（JPG/PNG/TIFF）が入ったフォルダを選びます。

【出力先】
  結果はすべて「画像フォルダ\\adaptFVC\\」の中に保存されます。元の画像は変更しません。
  学習したモデルと教師点は自動保存され、次回同じフォルダを開くと復元できます。

【このアプリは通信を行いません】
  インターネットへの接続・送信は一切ありません。ログも画像フォルダ内にだけ残ります。

【困ったとき】
  取扱説明書 第7章「トラブルシューティング」を参照してください。
  不具合の報告は GitHub の Issues へ。「画像フォルダ\\adaptFVC\\logs\\」の作業ログを添えてください。

【ソースコードとライセンス】
  {repo}
  adaptFVC はフリーソフトウェアです。GNU General Public License v3.0 以降の条件で
  再配布・改変できます。無保証です（詳細は LICENSE.txt）。
"""


def find_edge():
    for p in EDGE_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Microsoft Edge が見つかりません（PDF 化に使います）")


def html_to_pdf(html_path, pdf_path):
    edge = find_edge()
    url = "file:///" + os.path.abspath(html_path).replace("\\", "/")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)  # 古い PDF が残っていると失敗を見逃す
    # 普段使いの Edge が起動中だと、既定プロファイルへの要求はそちらに吸われて PDF が書かれない。
    # 使い捨てのプロファイルを渡して独立に起動させる
    profile = tempfile.mkdtemp(prefix="adaptfvc_edge_")
    cmd = [edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--user-data-dir={profile}", f"--print-to-pdf={os.path.abspath(pdf_path)}", url]
    try:
        subprocess.run(cmd, check=True, timeout=180, capture_output=True)
        # msedge.exe は印刷を子プロセスに渡してすぐ戻ることがある。PDF が書き終わる（サイズが止まる）まで待つ
        last, deadline = -1, time.time() + 180
        while time.time() < deadline:
            size = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else -1
            if size > 0 and size == last:
                break
            last = size
            time.sleep(1)
    finally:
        time.sleep(1)
        shutil.rmtree(profile, ignore_errors=True)
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 10000:
        raise RuntimeError(f"PDF 化に失敗: {pdf_path}")
    print(f"pdf: {pdf_path} ({os.path.getsize(pdf_path) // 1024} KB)")


def _read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def _section(title, body):
    bar = "=" * 78
    return f"{bar}\n{title}\n{bar}\n\n{body}\n\n"


def third_party_licenses():
    """同梱ライブラリのライセンス文を、いま動いている Python 環境の dist-info から集める"""
    parts = [
        "adaptFVC の Windows 実行ファイルは PyInstaller で作られており、以下のソフトウェアを同梱しています。\n"
        "The Windows executable of adaptFVC is built with PyInstaller and bundles the software below.\n\n"
    ]
    for name in BUNDLED_PACKAGES:
        dist = md.distribution(name)
        meta = dist.metadata
        lic = meta.get("License-Expression") or ((meta.get("License") or "").splitlines() or [""])[0]
        files = []
        for f in dist.files or []:
            parts_f = str(f).replace("\\", "/").split("/")
            in_dist_info = parts_f[0].endswith(".dist-info")
            name_l = parts_f[-1].lower()
            if in_dist_info and (name_l.startswith(("licen", "copying", "notice"))):
                files.append(f)
        if not files:
            raise RuntimeError(f"{name}: dist-info にライセンス文が見つかりません")
        body = f"Version: {dist.version}\nLicense: {lic}\n"
        for f in sorted(files, key=str):
            body += f"\n----- {os.path.basename(str(f))} -----\n\n" + _read_text(dist.locate_file(f)) + "\n"
        parts.append(_section(f"{meta['Name']} {dist.version}", body))

    base = sys.base_prefix
    py_lic = os.path.join(base, "LICENSE.txt")
    if os.path.exists(py_lic):
        parts.append(_section(f"Python {sys.version.split()[0]}", _read_text(py_lic)))
    # python.org 版は tk8.6 にだけ license.terms を置く（Tcl と Tk は同じ条項）
    for rel in [("tcl", "tk8.6", "license.terms"), ("tcl", "tcl8.6", "license.terms")]:
        p = os.path.join(base, *rel)
        if os.path.exists(p):
            parts.append(_section("Tcl/Tk", _read_text(p)))
            break
    parts.append(_section(
        "PyInstaller bootloader",
        "The executable contains the PyInstaller bootloader, which is distributed under the\n"
        "GNU General Public License v2 or later with a special exception that allows it to be\n"
        "used in executables of any license. See https://pyinstaller.org/en/stable/license.html"))
    return "".join(parts)


def main():
    # 既定は dist/。アプリ起動中で dist の exe を置き換えられないときは --exe で別の場所を指定する
    exe = sys.argv[sys.argv.index("--exe") + 1] if "--exe" in sys.argv else os.path.join(ROOT, "dist", "adaptFVC.exe")
    if not os.path.exists(exe):
        print("dist/adaptFVC.exe がありません。先に pyinstaller でビルドしてください（BUILD.md）")
        return 2
    docs = os.path.join(ROOT, "docs")
    manual_pdf = os.path.join(docs, MANUAL_PDF_NAME)
    html_to_pdf(os.path.join(docs, "manual.html"), manual_pdf)

    name = f"adaptFVC_v{__version__}_win64"
    rel_dir = os.path.join(ROOT, "release", name)
    shutil.rmtree(rel_dir, ignore_errors=True)
    os.makedirs(rel_dir)
    shutil.copy2(exe, os.path.join(rel_dir, "adaptFVC.exe"))
    shutil.copy2(manual_pdf, os.path.join(rel_dir, MANUAL_PDF_NAME))
    shutil.copy2(os.path.join(ROOT, "LICENSE"), os.path.join(rel_dir, "LICENSE.txt"))
    with open(os.path.join(rel_dir, "THIRD_PARTY_LICENSES.txt"), "w", encoding="utf-8") as f:
        f.write(third_party_licenses())
    with open(os.path.join(rel_dir, "README_はじめに.txt"), "w", encoding="utf-8-sig") as f:
        f.write(README_TXT.format(version=__version__, date=date.today().isoformat(),
                                  manual=MANUAL_PDF_NAME, repo=REPO_URL))

    zip_path = os.path.join(ROOT, "release", name + ".zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(rel_dir)):
            z.write(os.path.join(rel_dir, fn), os.path.join(name, fn))
    print(f"release: {zip_path} ({os.path.getsize(zip_path) // (1024 * 1024)} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
