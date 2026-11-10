#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
# adaptFVC を macOS 上でビルドして dist/adaptFVC.app と配布用 zip を作る。
# 前提: macOS 12 以降、python.org 版の Python 3.11〜3.13（Tk 同梱）。
# Apple Silicon と Intel でそれぞれ実行すると、その CPU 向けの .app ができる。
#
#   chmod +x build_mac.sh && ./build_mac.sh
#
# 署名・公証（Apple Developer ID）が無い場合、配布先では
# 「開発元を確認できないため開けません」と出る。右クリック → 開く で回避できる（BUILD.md）。
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
VENV=.venv-mac
if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements-app.txt pyinstaller==6.19.0

# tkinter が使えることを先に確認（Homebrew の python では別途 python-tk が必要）
python -c "import tkinter; tkinter.Tk().destroy()" || {
  echo "tkinter が使えません。python.org の Python を使うか、brew install python-tk を実行してください" >&2
  exit 1
}

rm -rf build dist
pyinstaller build.spec --noconfirm --clean

# 配布先で隔離属性を外しやすいよう、zip は ditto で作る（Finder と同じ形式）
VERSION=$(python -c "import adaptfvc; print(adaptfvc.__version__)")
ARCH=$(uname -m)
mkdir -p release
OUT="release/adaptFVC_v${VERSION}_mac_${ARCH}.zip"
rm -f "$OUT"
ditto -c -k --sequesterRsrc --keepParent dist/adaptFVC.app "$OUT"
echo "built: dist/adaptFVC.app"
echo "zip:   $OUT"
