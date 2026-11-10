# -*- mode: python ; coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
# adaptFVC アプリのビルド定義（Windows / macOS 共通）
#   Windows: venv\Scripts\pyinstaller.exe build.spec --noconfirm --clean  → dist/adaptFVC.exe
#   macOS:   ./build_mac.sh                                                → dist/adaptFVC.app
# PyInstaller はクロスビルドできないので、配布したい OS の上で実行すること（BUILD.md）
import os
import sys

IS_MAC = sys.platform == "darwin"

# conda 由来の Python では tcl86t.dll / tk86t.dll / ffi.dll などが
# <base>\Library\bin に置かれ、PyInstaller の既定の探索範囲から外れる。
# 見つからないまま固めると _tkinter が読み込めず配布先で画面が開かない。
# python.org 版など Library\bin を持たない環境では何も起きない。
_conda_dll_dir = os.path.join(sys.base_prefix, 'Library', 'bin')
if os.path.isdir(_conda_dll_dir):
    os.environ['PATH'] = _conda_dll_dir + os.pathsep + os.environ.get('PATH', '')

a = Analysis(
    ['adaptFVC.py'],
    pathex=[],
    binaries=[],
    datas=[('assets/app.ico', 'assets')],
    hiddenimports=[
        'sklearn.ensemble',
        'sklearn.tree',
        'sklearn.utils._typedefs',
        'openpyxl.styles',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 研究用の依存は同梱しない（アプリは使わない）
    excludes=['matplotlib', 'pandas', 'google', 'IPython', 'pytest', 'scipy.spatial.cKDTree'],
    noarchive=False,
)
pyz = PYZ(a.pure)

if IS_MAC:
    # macOS: onedir + .app バンドル（onefile は起動が遅く Gatekeeper とも相性が悪い）
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='adaptFVC',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon='assets/app.icns',
    )
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='adaptFVC')
    app = BUNDLE(
        coll,
        name='adaptFVC.app',
        icon='assets/app.icns',
        bundle_identifier='io.github.kotanakajima.adaptfvc',
        info_plist={
            'CFBundleShortVersionString': '3.2.0',
            'CFBundleVersion': '3.2.0',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '12.0',
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='adaptFVC',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon='assets/app.ico',
    )
