# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""adaptFVC アプリアイコン生成スクリプト。

デザイン: 深緑のグラデーション角丸スクエアの上に稲の葉。葉の先端側が
ピクセル（正方形）に分解していく = 「画像をピクセル単位で分類して植被率を出す」
というアプリの原理をそのまま図案化したもの。

出力: リポジトリ直下の icon.png (1024x1024) と assets/app.ico (16-256 のマルチサイズ)
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
S = 4                    # スーパーサンプリング係数（エッジのアンチエイリアス用）
N = 1024 * S


def lerp(c1, c2, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def bezier(p0, p1, p2, t):
    t = np.asarray(t, dtype=float)
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return np.stack([x, y], axis=-1)


def build():
    # ---- 背景: 対角グラデーション（深いティール→ほぼ黒の緑）----
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    t_bg = (xx + yy) / (2 * N)
    c_tl = np.array([18, 62, 46], np.float32)     # 左上
    c_br = np.array([5, 20, 15], np.float32)      # 右下
    bg = (c_tl[None, None] * (1 - t_bg[..., None]) + c_br[None, None] * t_bg[..., None])

    # 中央上寄りの淡いグロー（葉の後光）
    gx, gy, gr = 0.44 * N, 0.38 * N, 0.62 * N
    d = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2) / gr
    glow = np.clip(1 - d, 0, 1) ** 2
    bg += glow[..., None] * np.array([12, 52, 30], np.float32)

    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    # ---- 葉のセンターライン（2次ベジェ）と幅プロファイル ----
    p0 = (0.25 * N, 0.88 * N)    # 基部（左下）
    p1 = (0.30 * N, 0.26 * N)    # 制御点
    p2 = (0.82 * N, 0.13 * N)    # 先端（右上）
    ts = np.linspace(0, 1, 400)
    center = bezier(p0, p1, p2, ts)
    tangent = np.gradient(center, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    wmax = 0.082 * N
    width = wmax * np.sin(np.pi * np.clip(ts, 0, 1) ** 0.72) + 0.004 * N

    left = center + normal * width[:, None]
    right = center - normal * width[:, None]
    poly = [tuple(p) for p in np.concatenate([left, right[::-1]], axis=0)]

    leaf_mask = Image.new("L", (N, N), 0)
    ImageDraw.Draw(leaf_mask).polygon(poly, fill=255)
    leaf_np = np.array(leaf_mask) > 0

    # ---- 葉のグラデーション（基部の深緑→先端の明るい黄緑）----
    # 各ピクセルの「葉に沿った位置」はセンターラインへの最近傍 t で近似
    step = 8  # 距離計算はダウンサンプルして補間（速度対策）
    ys, xs = np.mgrid[0:N:step, 0:N:step].astype(np.float32)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    d2 = ((pts[:, None, :] - center[None, ::8, :]) ** 2).sum(-1)
    tt = ts[::8][np.argmin(d2, axis=1)].reshape(ys.shape)
    t_leaf = np.array(Image.fromarray((tt * 255).astype(np.uint8)).resize((N, N), Image.BILINEAR),
                      np.float32) / 255.0
    c_base = np.array([27, 122, 62], np.float32)
    c_tip = np.array([158, 240, 96], np.float32)
    t_grad = np.clip(t_leaf / 0.68, 0, 1)[..., None]
    leaf_rgb = c_base[None, None] * (1 - t_grad) + c_tip[None, None] * t_grad

    # ---- 先端側をピクセルに分解 ----
    t_cut = 0.68
    cut_pt = bezier(p0, p1, p2, np.array([t_cut]))[0]
    i_cut = int(t_cut * (len(ts) - 1))
    u = tangent[i_cut]                       # 進行方向
    proj = (xx - cut_pt[0]) * u[0] + (yy - cut_pt[1]) * u[1]
    intact = leaf_np & (proj <= 0)
    shatter_zone = leaf_np & (proj > 0)

    # 無傷部分を描画
    leaf_layer = np.zeros((N, N, 4), np.uint8)
    leaf_layer[..., :3] = np.clip(leaf_rgb, 0, 255).astype(np.uint8)
    leaf_layer[..., 3] = np.where(intact, 255, 0).astype(np.uint8)
    img = Image.alpha_composite(img, Image.fromarray(leaf_layer, "RGBA"))

    # 中央脈（無傷部分のみ、基部→カット位置）
    vein = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vein)
    vein_pts = [tuple(p) for p in center[30:i_cut - 10]]
    vd.line(vein_pts, fill=(14, 84, 44, 150), width=int(0.006 * N), joint="curve")
    vein_np = np.array(vein)
    vein_np[..., 3] = np.where(intact, vein_np[..., 3], 0)
    img = Image.alpha_composite(img, Image.fromarray(vein_np, "RGBA"))

    hl = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hl)
    hl_pts = [tuple(p) for p in left[40:i_cut - 6]]
    hd.line(hl_pts, fill=(210, 255, 170, 90), width=int(0.005 * N), joint="curve")
    hl_np = np.array(hl)
    hl_np[..., 3] = np.where(intact, hl_np[..., 3], 0)
    img = Image.alpha_composite(img, Image.fromarray(hl_np, "RGBA"))

    # ピクセル片: グリッドに整列した正方形で、カットから離れるほど薄く・まばらに
    g = int(0.062 * N)                       # グリッドセル
    rng = np.random.RandomState(7)
    sq = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sq)
    fade_len = 0.34 * N
    for cy in range(0, N, g):
        for cx in range(0, N, g):
            cell = shatter_zone[cy:cy + g, cx:cx + g]
            if cell.size == 0:
                continue
            cov = cell.mean()
            ccx, ccy = cx + g / 2, cy + g / 2
            dproj = (ccx - cut_pt[0]) * u[0] + (ccy - cut_pt[1]) * u[1]
            if cov < 0.06 or dproj < -g:
                continue
            fade = float(np.clip(1.0 - dproj / fade_len, 0.10, 1.0))
            if fade < 0.55 and rng.rand() > 0.20 + 0.80 * fade:   # 遠いセルほど間引く
                continue
            tc = float(np.clip(t_leaf[int(ccy), int(ccx)], 0, 1))
            col = lerp(tuple(c_base.astype(int)), tuple(c_tip.astype(int)), tc)
            alpha = int(255 * (0.30 + 0.70 * fade) * min(1.0, cov * 2.0))
            inset = g * (0.10 + 0.16 * (1 - fade))
            sd.rectangle([cx + inset, cy + inset, cx + g - inset, cy + g - inset],
                         fill=col + (alpha,))
    # さらに先へ飛んでいく小片
    drift = bezier(p0, p1, p2, np.linspace(0.84, 1.18, 7))
    for k, (dx, dy) in enumerate(drift):
        side = 1 if k % 2 == 0 else -1
        ox = dx + side * rng.uniform(0.01, 0.05) * N
        oy = dy + rng.uniform(-0.03, 0.03) * N
        size = g * rng.uniform(0.30, 0.62)
        a = int(150 * max(0.15, 1 - k / 7))
        col = lerp(tuple(c_base.astype(int)), tuple(c_tip.astype(int)), min(1.0, 0.75 + k * 0.04))
        sd.rectangle([ox, oy, ox + size, oy + size], fill=col + (a,))
    img = Image.alpha_composite(img, sq)

    # ---- 角丸マスクと微妙な縁のハイライト ----
    radius = int(0.185 * N)
    mask = Image.new("L", (N, N), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, N - 1, N - 1], radius=radius, fill=255)
    out = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)

    edge = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    ImageDraw.Draw(edge).rounded_rectangle(
        [S, S, N - 1 - S, N - 1 - S], radius=radius, outline=(255, 255, 255, 26), width=3 * S)
    out = Image.alpha_composite(out, edge)

    final = out.resize((1024, 1024), Image.LANCZOS)
    return final


def main():
    final = build()
    png_path = ROOT / "icon.png"
    final.save(png_path)
    ico_path = ROOT / "assets" / "app.ico"
    final.save(ico_path, sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    print(f"Wrote {png_path} and {ico_path}")


if __name__ == "__main__":
    main()
