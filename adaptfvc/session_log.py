# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""セッションログ（作業ログ。画像フォルダ内の adaptFVC/logs/ にだけ書き、通信はしない）。

アプリの通常使用時に、改善判断の材料になるイベントを JSONL で記録する:
どの操作に何秒かかったか・1回に何点アノテーションしたか・どんな設定で学習したか・
学習→確認のループを何周したか。常時ON・軽量・失敗しても絶対に作業を止めない。
"""
import json
import os
import time
from datetime import datetime


class SessionLogger:
    """1セッション = 1 JSONL ファイル（<画像フォルダ>/adaptFVC/logs/）"""

    def __init__(self, log_dir, app_version):
        self._f = None
        self._t0 = time.time()
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(
            log_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        self._f = open(path, "a", encoding="utf-8")
        self.path = path
        self.event("session_start", {"app_version": app_version})

    def event(self, name, data=None):
        if self._f is None:
            return
        try:
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": round(time.time() - self._t0, 1),
                "event": name,
                **(data or {}),
            }
            self._f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._f.flush()
        except Exception:
            pass

    def close(self):
        try:
            if self._f is not None:
                self._f.close()
                self._f = None
        except Exception:
            pass


class NullSessionLogger:
    """フォルダ未選択時などのダミー"""

    path = None

    def event(self, name, data=None):
        pass

    def close(self):
        pass
