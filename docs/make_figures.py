# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""取扱説明書の図1・図3〜5を作る（図2はスクリーンショットで docs/make_screenshot.py が撮る）。

合成した圃場画像に対して、アプリと同じ計算コード（adaptfvc.core の Lab 特徴量・Random Forest・
ExG + Otsu）で実際に計算し、その結果を描く。図中の数値（植被率・しきい値・票数）はすべて実計算。

使い方: python docs/make_figures.py   （matplotlib が要る。アプリの実行には不要）
出力:   docs/figures/fig_pipeline.png, fig_random_forest.png, fig_decision_boundary.png,
        fig_adaptive_vs_fixed.png
"""
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from adaptfvc import core  # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 教師点の色はアプリの表示に合わせる（植物=緑、土壌=茶）。赤はアプリでは「植物と判定」の重ね表示の色なので使わない
C_PLANT, C_SOIL = "#1f9d55", "#7a5b40"
MASK_PLANT = np.array([20, 90, 40], np.uint8)  # 二値マスク表示の植物色（濃い緑）
MASK_OTHER = np.array([247, 251, 246], np.uint8)
H, W = 240, 320


# ---------------------------------------------------------------- 合成画像
def soil_texture(rng):
    base = np.array([128, 104, 78], np.float32)
    low = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 18)
    low = low / (np.abs(low).max() + 1e-6) * 22  # 乾いた土・湿った土のむら
    img = base[None, None, :] + low[..., None] + rng.normal(0, 6, (H, W, 3))
    return img


def seedling_field(seed=7):
    """苗の並んだ圃場（RGB, uint8）と、真の植物マスク・植え穴マスク"""
    rng = np.random.RandomState(seed)
    img = soil_texture(rng)
    plant = np.zeros((H, W), np.uint8)
    hole = np.zeros((H, W), np.uint8)
    for j in range(4):
        for i in range(8):
            cx = int(22 + i * 40 + rng.randint(-3, 4))
            cy = int(34 + j * 57 + rng.randint(-3, 4))
            cv2.ellipse(hole, (cx + 2, cy + 6), (15, 10), 0, 0, 360, 1, -1)
            if rng.rand() < 0.06:  # 欠株
                continue
            n_leaf = rng.randint(4, 7)
            for k in range(n_leaf):
                ang = rng.uniform(0, 360)
                length = rng.randint(10, 17)
                width = rng.randint(4, 7)
                t = np.deg2rad(ang)
                ex, ey = int(cx + np.cos(t) * length * 0.7), int(cy + np.sin(t) * length * 0.7)
                leaf = np.zeros((H, W), np.uint8)
                cv2.ellipse(leaf, (ex, ey), (length, width), ang, 0, 360, 1, -1)
                shade = rng.uniform(0.5, 1.25)  # 0.5 付近は影になった葉
                col = np.array([62, 168, 46]) * shade + rng.normal(0, 4, 3)
                m = leaf.astype(bool)
                img[m] = col + rng.normal(0, 5, (m.sum(), 3))
                plant |= leaf
    hm = hole.astype(bool) & ~plant.astype(bool)
    img[hm] = img[hm] * 0.42
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)  # カメラのにじみ＝境界の混合画素
    return img, plant.astype(bool), hm


def bare_field(seed=11):
    """植物がほぼ無いプロット（播種前・欠株）。暗い筋（轍・影）がある"""
    rng = np.random.RandomState(seed)
    img = soil_texture(rng)
    for (y0, y1, wd) in [(22, 30, 9), (118, 108, 7), (205, 214, 6), (226, 228, 5)]:
        band = np.zeros((H, W), np.uint8)
        cv2.line(band, (0, y0), (W, y1), 1, wd)
        b = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.5)
        img = img * (1 - 0.25 * b[..., None])
    return cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)


def sample_points(mask, n, rng, erode=3, margin=8):
    m = cv2.erode(mask.astype(np.uint8), np.ones((erode, erode), np.uint8)).astype(bool)
    m[:margin, :] = m[-margin:, :] = False
    m[:, :margin] = m[:, -margin:] = False
    ys, xs = np.nonzero(m)
    idx = rng.choice(len(ys), n, replace=False)
    return list(zip(xs[idx], ys[idx]))


def to_bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def lab_std(v_cv):
    """OpenCV の 8bit Lab（L*×2.55, a*+128, b*+128）を CIELAB の単位に戻す"""
    v = np.asarray(v_cv, np.float64)
    return np.stack([v[..., 0] / 2.55, v[..., 1] - 128.0, v[..., 2] - 128.0], -1)


def mask_rgb(mask):
    out = np.empty((*mask.shape, 3), np.uint8)
    out[:] = MASK_OTHER
    out[mask] = MASK_PLANT
    return out


# ---------------------------------------------------------------- 計算
def build():
    rng = np.random.RandomState(3)
    field, true_plant, holes = seedling_field()
    bare = bare_field()
    soil_area = ~true_plant
    plant_pts = sample_points(true_plant, 12, rng)
    soil_pts = sample_points(soil_area & ~holes, 9, rng, erode=7) + sample_points(holes, 3, rng)
    bgr = to_bgr(field)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    X = np.array([lab[y, x] for (x, y) in plant_pts + soil_pts], np.uint8)
    y = np.array([1] * len(plant_pts) + [0] * len(soil_pts))
    model = core.train_rf(X, y)
    return dict(field=field, bare=bare, true_plant=true_plant, plant_pts=plant_pts, soil_pts=soil_pts,
                X=X, y=y, model=model, lab=lab)


# ---------------------------------------------------------------- 図1 処理の流れ
def fig_pipeline(d):
    bgr = to_bgr(d["field"])
    mask = core.predict_mask(bgr, d["model"])
    fvc = core.calculate_coverage(mask)
    result = np.asarray(core.compose_result_image(bgr, mask))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].imshow(d["field"])
    axes[0].set_title("① 元の RGB 画像", fontsize=12)
    axes[1].imshow(d["field"])
    px, py = zip(*d["plant_pts"])
    sx, sy = zip(*d["soil_pts"])
    axes[1].scatter(px, py, s=60, c=C_PLANT, edgecolors="black", linewidths=1.6, label="教師点「植物」")
    axes[1].scatter(sx, sy, s=60, c=C_SOIL, edgecolors="white", linewidths=1.8, marker="s", label="教師点「土壌」")
    axes[1].set_title("② 植物と土壌に教師点を打つ（アノテーション）", fontsize=12)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=10, frameon=False)
    axes[2].imshow(result)
    axes[2].set_title(f"③ 全画素を分類（植被率 {fvc:.1f}%）", fontsize=12)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=150)
    plt.close(fig)
    return fvc


# ---------------------------------------------------------------- 図2 Random Forest の判定図
FEAT = ["L*", "a*", "b*"]


def _q(tree, node):
    f = tree.feature[node]
    t = tree.threshold[node]
    t_std = t / 2.55 if f == 0 else t - 128.0
    return f"{FEAT[f]} ≦ {t_std:.1f} ？", f, t


def _leaf_label(tree, node):
    v = tree.value[node][0]
    return "植物" if v[1] >= v[0] else "土壌"


def _box(ax, x, y, text, fc, ec, fs=10.5, w=0.2, h=0.085, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def _draw_tree(ax, tree, node, x, y, dx, depth, max_depth, path):
    on_path = node in path
    if tree.children_left[node] == -1:
        lab = _leaf_label(tree, node)
        _box(ax, x, y, lab, "#e3f4e8" if lab == "植物" else "#f6e7df", "#c0392b" if on_path else "#9aa5a0",
             w=0.1, bold=on_path)
        return
    if depth >= max_depth:
        _box(ax, x, y, "…", "#f2f4f3", "#c0392b" if on_path else "#9aa5a0", w=0.06)
        return
    text, _, _ = _q(tree, node)
    _box(ax, x, y, text, "#ffffff", "#c0392b" if on_path else "#56635c", w=0.17, bold=on_path)
    for child, sgn, lab in [(tree.children_left[node], -1, "はい"), (tree.children_right[node], 1, "いいえ")]:
        cx, cy = x + sgn * dx, y - 0.21
        hot = on_path and child in path
        ax.plot([x, cx], [y - 0.045, cy + 0.045], color="#c0392b" if hot else "#8a958f", lw=2.2 if hot else 1.1,
                zorder=0)
        ax.text((x + cx) / 2 + sgn * 0.012, (y + cy) / 2, lab, ha="right" if sgn < 0 else "left", va="center",
                fontsize=9, color="#c0392b" if hot else "#56635c")
        _draw_tree(ax, tree, child, cx, cy, dx / 2, depth + 1, max_depth, path)


def fig_random_forest(d):
    model = d["model"]
    bgr = to_bgr(d["field"])
    lab = d["lab"].reshape(-1, 3)
    # 境界付近の画素（葉の縁の混合画素）で、木によって答えが割れるものを1つ選ぶ
    votes_all = np.stack([est.predict(lab.astype(np.float64)) for est in model.estimators_], 1)
    frac = votes_all.mean(1)
    cand = np.nonzero((frac > 0.6) & (frac < 0.85))[0]
    pick = None
    for i in cand:
        v3 = votes_all[i, :3]
        if v3.min() != v3.max():
            pick = i
            break
    if pick is None:
        pick = cand[0]
    x_cv = lab[pick].astype(np.float64)
    x_std = lab_std(x_cv)
    v = votes_all[pick]
    n_plant, n_soil = int(v.sum()), int(len(v) - v.sum())

    fig = plt.figure(figsize=(11, 5.6))
    ax = fig.add_axes([0.0, 0.0, 0.64, 1.0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # 枝分かれが見えるように、200 本のうち質問（分岐）の数がいちばん多い木を描く
    # 3.2節で「緑さ（a*）が保たれる」と説明するので、根で a* を問う木（なければ上3段のどこかで a* を問う木）を選ぶ
    def uses_a(t, node=0, depth=0):
        if t.children_left[node] == -1 or depth >= 3:
            return False
        return (t.feature[node] == 1 or uses_a(t, t.children_left[node], depth + 1)
                or uses_a(t, t.children_right[node], depth + 1))
    ests = model.estimators_
    cands = [k for k, e in enumerate(ests) if e.tree_.feature[0] == 1 and e.tree_.node_count > 3]
    cands = cands or [k for k, e in enumerate(ests) if uses_a(e.tree_)] or list(range(len(ests)))
    k_show = max(cands, key=lambda k: (ests[k].tree_.node_count, -k))
    est = model.estimators_[k_show]
    tree = est.tree_
    path = set(est.decision_path(x_cv[None, :]).indices)
    ax.text(0.5, 0.95, f"決定木の1本（200 本のうち {k_show + 1} 本目。上から3段まで）", ha="center", fontsize=12,
            fontweight="bold")
    _draw_tree(ax, tree, 0, 0.5, 0.82, 0.24, 0, 3, path)
    ax.text(0.5, 0.06, f"赤い線＝右の画素（L* {x_std[0]:.0f}, a* {x_std[1]:.0f}, b* {x_std[2]:.0f}）がたどる道筋",
            ha="center", fontsize=10, color="#c0392b")

    ax2 = fig.add_axes([0.65, 0.0, 0.35, 1.0])
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.text(0.5, 0.95, "200 本の木の多数決", ha="center", fontsize=12, fontweight="bold")
    sw = np.full((10, 10, 3), d["lab"].reshape(-1, 3)[pick], np.uint8)
    sw_rgb = cv2.cvtColor(cv2.cvtColor(sw, cv2.COLOR_Lab2BGR), cv2.COLOR_BGR2RGB)
    axi = fig.add_axes([0.765, 0.74, 0.12, 0.13])
    axi.imshow(sw_rgb); axi.set_xticks([]); axi.set_yticks([])
    axi.set_title("判定したい画素の色", fontsize=9.5)
    ys = [0.62, 0.53, 0.44]
    for k, yy in enumerate(ys):
        lab_k = "植物" if v[k] == 1 else "土壌"
        _box(ax2, 0.5, yy, f"木 {k + 1}：{lab_k}", "#e3f4e8" if v[k] == 1 else "#f6e7df", "#9aa5a0", w=0.5, h=0.065)
    ax2.text(0.5, 0.37, "⋮", ha="center", va="center", fontsize=16, color="#56635c")
    lab_200 = "植物" if v[199] == 1 else "土壌"
    _box(ax2, 0.5, 0.29, f"木 200：{lab_200}", "#e3f4e8" if v[199] == 1 else "#f6e7df", "#9aa5a0", w=0.5, h=0.065)
    final = "植物" if n_plant > n_soil else "土壌"
    _box(ax2, 0.5, 0.14, f"植物 {n_plant} 票・土壌 {n_soil} 票\n→ この画素は「{final}」", "#fff8e1", "#f2b705",
         fs=11, w=0.78, h=0.12, bold=True)
    fig.savefig(os.path.join(OUT, "fig_random_forest.png"), dpi=150)
    plt.close(fig)
    return dict(x_std=x_std, n_plant=n_plant, n_soil=n_soil, root_q=_q(tree, 0)[0], k_show=k_show,
                depth=est.get_depth())


# ---------------------------------------------------------------- 図3 色空間の中の境界
def fig_decision_boundary(d):
    model = d["model"]
    lab = d["lab"].reshape(-1, 3)
    rng = np.random.RandomState(0)
    sub = lab[rng.choice(len(lab), 4000, replace=False)]
    s_std = lab_std(sub)
    L_med = float(np.median(lab[:, 0]))
    a = np.linspace(-68, 25, 400)
    b = np.linspace(-5, 70, 400)
    A, B = np.meshgrid(a, b)
    grid = np.stack([np.full(A.size, L_med), A.ravel() + 128, B.ravel() + 128], 1)
    Z = model.predict(grid).reshape(A.shape)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.contourf(A, B, Z, levels=[-0.5, 0.5, 1.5], colors=["#f1e4d6", "#dcf0dc"])
    ax.contour(A, B, Z, levels=[0.5], colors=["#1f7a3d"], linewidths=2)
    ax.scatter(s_std[:, 1], s_std[:, 2], s=3, c="#8c8c8c", alpha=0.35, label="画像の画素（4,000 画素を表示）")
    tp = lab_std(d["X"][d["y"] == 1]); ts = lab_std(d["X"][d["y"] == 0])
    ax.scatter(tp[:, 1], tp[:, 2], s=70, c=C_PLANT, edgecolors="black", linewidths=0.8, label="教師点「植物」")
    ax.scatter(ts[:, 1], ts[:, 2], s=60, c=C_SOIL, edgecolors="black", linewidths=0.8, marker="s",
               label="教師点「土壌」")
    ax.axvline(0, color="#777", lw=0.8, ls="--")
    ax.text(1, 66, "a* = 0（無彩色）", fontsize=8.5, color="#555")
    ax.set_xlabel("a*（左へ行くほど緑）")
    ax.set_ylabel("b*（上へ行くほど黄）")
    ax.set_title(f"植物と土壌の境界（L* = {L_med / 2.55:.0f} の断面）", fontsize=12)
    ax.set_xlim(-68, 25); ax.set_ylim(-5, 70)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.text(-58, 60, "植物と判定", color="#1f7a3d", fontsize=10, fontweight="bold")
    ax.text(5, 8, "土壌と判定", color="#8a5a2b", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_decision_boundary.png"), dpi=150)
    plt.close(fig)
    return L_med / 2.55


# ---------------------------------------------------------------- 図4 全自動のしきい値法が破綻する例
def fig_adaptive_vs_fixed(d):
    rows = [("苗が生えているプロット", d["field"]), ("植物がほぼ無いプロット", d["bare"])]
    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.99, top=0.9, bottom=0.05, wspace=0.05, hspace=0.22)
    out = []
    for r, (name, rgb) in enumerate(rows):
        bgr = to_bgr(rgb)
        m_exg, _ = core.calculate_exg_mask(bgr)
        m_rf = core.predict_mask(bgr, d["model"])
        f_exg, f_rf = core.calculate_coverage(m_exg), core.calculate_coverage(m_rf)
        out.append((f_exg, f_rf))
        panels = [(rgb, None), (mask_rgb(m_exg), f_exg), (mask_rgb(m_rf), f_rf)]
        for c, (im, f) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#333")
            if r == 0:
                ax.set_title(["撮影画像", "全自動のしきい値（ExG + Otsu）", "adaptFVC（教師点から学習）"][c], fontsize=12)
            if f is not None:
                bad = (r == 1 and c == 1)
                ax.text(0.5, -0.07, f"植被率 = {f:.1f}%", transform=ax.transAxes, ha="center", va="center",
                        fontsize=12 if bad else 11, color="#c0392b" if bad else "#222",
                        fontweight="bold" if bad else "normal")
            if c == 0:
                ax.text(-0.06, 0.5, name, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=11.5)
    fig.suptitle("自動しきい値は画像を必ず2つに割る。植物が無くても植物を作り出してしまう", fontsize=13)
    fig.savefig(os.path.join(OUT, "fig_adaptive_vs_fixed.png"), dpi=150)
    plt.close(fig)
    return out


def main():
    d = build()
    f1 = fig_pipeline(d)
    rf = fig_random_forest(d)
    lmed = fig_decision_boundary(d)
    f4 = fig_adaptive_vs_fixed(d)
    print(f"fig1 FVC = {f1:.1f}%")
    print(f"fig2 tree #{rf['k_show'] + 1} depth {rf['depth']} root question = {rf['root_q']}  pixel Lab = {np.round(rf['x_std'], 1)}  "
          f"votes plant {rf['n_plant']} / soil {rf['n_soil']}")
    print(f"fig3 L* slice = {lmed:.1f}")
    print(f"fig4 seedling: ExG+Otsu {f4[0][0]:.1f}% / RF {f4[0][1]:.1f}%   bare: ExG+Otsu {f4[1][0]:.1f}% / RF {f4[1][1]:.1f}%")


if __name__ == "__main__":
    main()
