# adaptFVC の開発者向け情報（ビルド・テスト・配布）

この文書は、adaptFVC のコードを直す人・実行ファイルを作る人向けです。使い方は取扱説明書（`docs/adaptFVC_manual_ja.pdf`）を参照してください。

PyInstaller はクロスビルドできない。**Windows 用は Windows 上で、macOS 用は macOS 上で**ビルドする。
どちらも `build.spec` を共有する（OS を見て自動で切り替わる）。

## 共通の前提

- Python 3.12（**python.org 版を推奨**。Tk が同梱される。3.12.6 で確認）
- ビルド用の環境は、解析など他の用途と分けて作る。exe に入るのは、その環境に入っているライブラリだけになる
- 版は `requirements-app-lock.txt` に固定してある（実行時のライブラリ＋PyInstaller）。`requirements-app.txt` は実行時のライブラリだけ
- 版を上げるときは、公開から7日以上たった版を選び、`pip-audit` で既知の脆弱性が無いことを確かめる
- ビルド前にアプリが起動していないことを確認する（起動中だとビルドが固まる）

## Windows

```bat
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --no-deps --only-binary :all: -r requirements-app-lock.txt
.venv\Scripts\python -m pip check
.venv\Scripts\pyinstaller.exe build.spec --noconfirm --clean
```

- 成果物: `dist\adaptFVC.exe`（onefile、約 107 MB）
- **Miniconda 由来の Python でビルドする場合**: tcl86t.dll / tk86t.dll などが `<base>\Library\bin` にあり
  PyInstaller の探索範囲外。`build.spec` 冒頭で PATH に前置して対処済みだが、ビルドログに
  `Library not found` 警告が**ゼロ**であることを毎回確認する（警告が出てもビルドは成功扱いで完走してしまう）
- `Hidden import "scipy.special._cdflib" not found` の警告は無害（scipy 1.17 でそのモジュールが無くなった。アプリは使わない）
- 動作確認:
  1. `dist\adaptFVC.exe` をダブルクリックして画面が開くこと（PATH から Python を外した状態でも）
  2. 画面を開かない処理（下の節）で、12 枚を超えるフォルダの全画像を処理できること（frozen + multiprocessing の確認）
- **コード署名はしていない**（2026-11 の判断。費用に見合う効果が無い）。署名しても、新しい版は SmartScreen の評判がゼロから始まるので、
  「WindowsによってPCが保護されました」の警告はすぐには消えない（EV 証明書も 2024 年以降は同じ）。
  利用者は「詳細情報」→「実行」で起動できる。将来署名する場合は `signtool sign /fd SHA256 /tr <タイムスタンプ URL> /td SHA256 /a dist\adaptFVC.exe`

## macOS

```bash
chmod +x build_mac.sh
./build_mac.sh
```

- 成果物: `dist/adaptFVC.app` と `release/adaptFVC_v<版>_mac_<arm64|x86_64>.zip`（Mac での動作は未確認）
- Apple Silicon（arm64）と Intel（x86_64）は別々の Mac でビルドする（または universal2 の Python を使う）
- Homebrew の Python は Tk が別パッケージ（`brew install python-tk`）。python.org 版が確実
- **Gatekeeper**: Developer ID で署名・公証していない .app は、配布先で
  「開発元を確認できないため開けません」と表示される。初回のみ **右クリック → 開く** で起動できる。
  zip を展開した .app に隔離属性が付いて開けない場合は `xattr -dr com.apple.quarantine adaptFVC.app`
- 署名・公証する場合（Apple Developer Program が必要）:
  ```bash
  codesign --deep --force --options runtime --sign "Developer ID Application: <名前>" dist/adaptFVC.app
  ditto -c -k --keepParent dist/adaptFVC.app adaptFVC.zip
  xcrun notarytool submit adaptFVC.zip --keychain-profile <profile> --wait
  xcrun stapler staple dist/adaptFVC.app
  ```
- 動作確認は Windows と同じ（`dist/adaptFVC.app/Contents/MacOS/adaptFVC --batch ...` で画面を開かない処理も使える）

## 画面を開かない処理（`--batch`）

保存したモデルか教師点のファイルを使って、画面を開かずにフォルダ内の全画像を処理する。
配布用の実行ファイルの動作確認と自動化のための機能で、取扱説明書には載せていない。

```bat
adaptFVC.exe --batch <画像フォルダ> --model <model.pkl または annotations.json> [--out <出力先>]
```

ソースから動かす場合は `python -m adaptfvc --batch ...`。

- 結果は既定で `<画像フォルダ>\adaptFVC\results\run_<日時>\` に保存される（`--out` で変えられる）
- exe はコンソールの無い windowed 版。コマンドプロンプトやショートカットから起動すると画面には何も出ず、経過と結果は結果フォルダの `batch_log.txt` に書かれる。
  Git Bash のように出力を受け取るシェルから起動すると、メッセージはそのシェルに出て `batch_log.txt` は作られない（2026-11-10 に確認）。
  結果フォルダを作る前に失敗した場合（画像が無い、教師点が足りない、モデルが読めない）は何も残らない
- 縮小は、そのフォルダを画面で開いて縮小サイズを選んである場合だけ行われる。選んでいなければ原寸で処理する
  （縮小版で学習したモデルを原寸に当てることになるので、確認には画面で一度開いたフォルダを使う）
- モデルがそのまま使えない場合（scikit-learn の版が違うなど）は、同梱の教師点から学習し直す

## テスト

```bat
python tests\equivalence_harness.py check
python tests\e2e_v3.py
```

- `equivalence_harness.py`: 合成画像を使って、計算コアの結果（マスク・植被率・特徴量・ワーカーの出力・モデルの保存と読込）を
  `tests/baseline.json` と突き合わせる。リファクタリングのときは必ず回す。意図して結果を変えたときだけ `record` で基準を作り直す
- `e2e_v3.py`: 実際のアプリの画面を動かして、フォルダの読み込み → 教師点 → 学習 → 自動保存 → 推論表示 → 読込 → プレビュー →
  全画像の処理 → 復元 → 透明画素 → 画面を開かない処理までを確かめる。
  `test_smoke/` に植物と土の写った圃場画像を拡張子 `.JPG` で**ちょうど 10 枚**置いて実行する（画像はリポジトリに含めない）

## 取扱説明書の図

- 図1・図3〜5: `python docs/make_figures.py`（合成画像をアプリと同じ計算コードで処理して描く。図中の数値はすべて実計算。
  matplotlib が要る。アプリの依存には入れない）
- 図2（スクリーンショット）: `python docs/make_screenshot.py`（Windows。`test_smoke/` の画像でアプリを開き、教師点を打って学習した画面を撮る）
- 図の数値（植被率・票数など）を変えたら、`docs/manual.html` の図の説明文も合わせて直す

## 配布物の作り方（Windows）

```bat
.venv\Scripts\python.exe tools\make_release.py
```

`release/adaptFVC_v<版>_win64.zip` に次が入る: `adaptFVC.exe`、`adaptFVC_manual_ja.pdf`、
`README_はじめに.txt`、`LICENSE.txt`、`THIRD_PARTY_LICENSES.txt`。

- 取扱説明書の PDF は `docs/manual.html` から Edge のヘッドレス印刷で作り、`docs/adaptFVC_manual_ja.pdf` も更新する
- `THIRD_PARTY_LICENSES.txt` は exe に同梱されるライブラリのライセンス文を、実行中の Python 環境の dist-info から集める。
  **exe をビルドしたのと同じ venv で実行すること**（別の環境だと版がずれる）

## 配布前チェックリスト

- [ ] `pip-audit -r requirements-app-lock.txt --no-deps --disable-pip` で既知の脆弱性が無い
- [ ] `python tests\equivalence_harness.py check` が OK
- [ ] `python tests\e2e_v3.py` が全 PASS
- [ ] ビルドログに `Library not found` 警告が無い
- [ ] 実行ファイルを**ビルド環境の無い PC**（または PATH から Python を外した状態）で起動できる
- [ ] `--batch` で 12 枚を超えるフォルダの全画像の処理が通る（frozen + multiprocessing）
- [ ] 取扱説明書の版・日付・スクリーンショットが実物と一致する
- [ ] リリースのページに、zip と exe の SHA-256 を載せる（`certutil -hashfile <ファイル> SHA256`）
