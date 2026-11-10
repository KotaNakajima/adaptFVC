# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""PyInstaller 用エントリスクリプト。"""

import io
import multiprocessing
import sys

# windowed (console=False) ビルドでは stdout/stderr が None になり、
# tqdm や print が AttributeError で落ちるため受け皿を差す
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

from adaptfvc.main import run

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run()
