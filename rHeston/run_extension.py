#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thí nghiệm mở rộng cho tiểu luận (đặt cùng thư mục với run_experiments.py và gen_data.py):

  1. Đường cong học: huấn luyện lại với 12.5%, 25%, 50%, 100% tập huấn luyện
  2. Nhiều hạt giống ngẫu nhiên (mặc định 5): báo cáo trung bình +- độ lệch chuẩn
  3. Khoảng tin cậy bootstrap (trên tập kiểm tra) cho chênh lệch RMSE giữa MLP (MSE) và trung bình posterior mNRE

Với mỗi cấu hình, script đo: sai số hồi phục (RMSE / khoảng prior), độ phủ HPD tại 50/80/90%,
sai số hiệu chuẩn trung bình (trung bình |độ phủ - mức danh nghĩa| trên các mức 0.1..0.9) và mức co hẹp posterior.

Cách chạy:
  python run_extension.py --data data\\rh_v1.npz --out results_ext
  python run_extension.py --data data\\rh_v1.npz --out results_ext --seeds 3 --fractions 0.25 1.0   (chạy nhanh hơn)
"""
import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_experiments as R

NOMINALS_ALL = np.arange(0.1, 0.95, 0.1)


def fit_and_evaluate(theta, x, split, frac, seed, hidden, max_epochs, patience, grid):
    """Huấn luyện MLP điểm + 6 mNRE trên một phần của tập huấn luyện, đánh giá trên tập kiểm tra cố định."""
    rng = np.random.default_rng(1000 * seed + 7)
    tr_idx = np.where(split == 0)[0]
    perm = np.random.default_rng(0).permutation(len(tr_idx))  # thứ tự cố định -> các tập con lồng nhau
    n_sub = max(100, int(round(frac * len(tr_idx))))
    sub = tr_idx[perm[:n_sub]]
    va, te = split == 1, split == 2

    pre = R.Preproc(x[sub])
    z_tr, z_va, z_te = pre(x[sub]), pre(x[va]), pre(x[te])
    u = R.to_u(theta)
    u_tr, u_va, u_te = u[sub], u[va], u[te]
    targs = argparse.Namespace(max_epochs=max_epochs, patience=patience)

    point, _ = R.train_point(z_tr, u_tr, z_va, u_va, hidden, targs, rng)
    clfs = [R.train_mnre(i, z_tr, u_tr, z_va, u_va, hidden, targs, rng)[0] for i in range(R.M)]

    se_mlp = (point.predict(z_te) - u_te) ** 2           # (n_te, 6), trong thang u
    se_nre = np.empty_like(se_mlp)
    levels = np.empty_like(se_mlp)
    sds = np.empty_like(se_mlp)
    for i in range(R.M):
        r = R.grid_ratio(clfs[i], z_te, grid)
        dens = R.posterior_density(r, grid)
        mean, _, sd, mass = R.summarize_posterior(dens, grid)
        se_nre[:, i] = (mean - u_te[:, i]) ** 2
        levels[:, i] = R.hpd_levels(mass, dens, grid, u_te[:, i])
        sds[:, i] = sd
    return n_sub, se_mlp, se_nre, levels, sds


def make_figures(rows, fracs, seeds, out):
    """Vẽ hai hình từ danh sách các dòng kết quả (dùng lại được từ learning_curve.csv bằng --replot)."""
    arr = {}
    for r in rows:
        arr.setdefault((r[0], r[3]), []).append(r)
    # ---------------- Hình đường cong học ----------------
    n_by_frac = {fr: next(r[1] for r in rows if r[0] == fr) for fr in fracs}
    xs = [n_by_frac[fr] for fr in fracs]

    def curve(name, col):
        m, s = [], []
        for fr in fracs:
            v = np.array([r[col] for r in arr[(fr, name)]], dtype=float)
            m.append(v.mean())
            s.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        return np.array(m), np.array(s)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        for col, lab, c in ((4, "MLP (MSE)", "tab:orange"), (5, "mNRE (trung bình)", "tab:blue")):
            m, s = curve(R.NAMES[i], col)
            ax.errorbar(xs, m, yerr=s, marker="o", capsize=3, label=lab, color=c)
        ax.set_xscale("log")
        ax.set_xlim(min(xs) * 0.8, max(xs) * 1.25)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in xs], fontsize=7)
        ax.minorticks_off()
        ax.set_title(R.LABELS[i])
        ax.set_ylabel("RMSE / khoảng prior")
        ax.set_xlabel("Số mẫu huấn luyện")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Đường cong học (trung bình ± độ lệch chuẩn trên {seeds} hạt giống)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_learning_curve.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(R.M):
        m, s = curve(R.NAMES[i], 9)
        axes[0].errorbar(xs, m, yerr=s, marker="o", capsize=3, label=R.LABELS[i])
        m, s = curve(R.NAMES[i], 10)
        axes[1].errorbar(xs, m, yerr=s, marker="o", capsize=3, label=R.LABELS[i])
    axes[0].set_ylabel("Sai số hiệu chuẩn trung bình")
    axes[1].set_ylabel("sd posterior / sd prior")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(min(xs) * 0.8, max(xs) * 1.25)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in xs], fontsize=7)
        ax.minorticks_off()
        ax.set_xlabel("Số mẫu huấn luyện")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_calibration_vs_size.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Đường cong học, nhiều hạt giống, bootstrap")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_ext")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--grid", type=int, default=1025)
    ap.add_argument("--boot", type=int, default=2000, help="Số lần lấy mẫu lại bootstrap")
    ap.add_argument("--noise-bp", type=float, default=0.0,
                    help="Cộng nhiễu Gauss độc lập (độ lệch chuẩn tính bằng điểm cơ bản của IV) vào mọi bề mặt")
    ap.add_argument("--replot", action="store_true", help="Chỉ vẽ lại hình từ learning_curve.csv, không huấn luyện lại")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.replot:
        import csv
        with open(os.path.join(args.out, "learning_curve.csv"), newline="", encoding="utf-8") as f:
            rd = list(csv.reader(f))[1:]
        rows = [[float(r[0]), int(float(r[1])), int(float(r[2])), r[3]] + [float(v) for v in r[4:]] for r in rd]
        make_figures(rows, sorted(set(r[0] for r in rows)), max(r[2] for r in rows) + 1, args.out)
        print("Đã vẽ lại hình trong", os.path.abspath(args.out))
        return
    d = np.load(args.data)
    theta, x, split = d["theta"], d["x"], d["split"]
    if args.noise_bp > 0:
        x = x + np.random.default_rng(99).normal(0.0, args.noise_bp * 1e-4, size=x.shape)
    grid = np.linspace(-1, 1, args.grid)
    hidden = tuple(args.hidden)
    n_te = int((split == 2).sum())
    print(f"Dữ liệu: {len(theta)} mẫu | train {(split == 0).sum()} | val {(split == 1).sum()} | test {n_te}", flush=True)

    quiet = lambda *a, **k: None
    real_log = R.log
    R.log = quiet  # tắt log chi tiết của từng lần huấn luyện

    rows = []
    store = {}   # (frac, seed) -> (se_mlp, se_nre)
    total = len(args.fractions) * args.seeds
    done, t0 = 0, time.time()
    for frac in args.fractions:
        for seed in range(args.seeds):
            n_sub, se_mlp, se_nre, levels, sds = fit_and_evaluate(
                theta, x, split, frac, seed, hidden, args.max_epochs, args.patience, grid)
            store[(frac, seed)] = (se_mlp, se_nre)
            for i in range(R.M):
                cov = lambda a: float((levels[:, i] <= a).mean())
                calib_err = float(np.mean([abs(cov(a) - a) for a in NOMINALS_ALL]))
                rows.append([frac, n_sub, seed, R.NAMES[i],
                             np.sqrt(se_mlp[:, i].mean()) / 2, np.sqrt(se_nre[:, i].mean()) / 2,
                             cov(0.5), cov(0.8), cov(0.9), calib_err,
                             float((sds[:, i] / np.sqrt(1 / 3)).mean())])
            done += 1
            el = time.time() - t0
            print(f"[{done}/{total}] tỉ lệ={frac} hạt giống={seed} n_train={n_sub} | {el / 60:.1f} phút, "
                  f"còn khoảng {el / done * (total - done) / 60:.1f} phút", flush=True)
    R.log = real_log

    header = ["fraction", "n_train", "seed", "param", "rmse_rel_mlp", "rmse_rel_nre",
              "cov50", "cov80", "cov90", "calib_err", "sd_ratio"]
    with open(os.path.join(args.out, "learning_curve.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.5g}" if isinstance(v, float) else str(v) for v in r) + "\n")

    # ---------------- Tổng hợp trung bình +- độ lệch chuẩn qua các hạt giống ----------------
    arr = {}
    for r in rows:
        arr.setdefault((r[0], r[3]), []).append(r)
    summ = []
    for (frac, name), rs in arr.items():
        a = np.array([[v for v in r[4:]] for r in rs], dtype=float)  # (seeds, 7)
        m, s = a.mean(axis=0), a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1])
        summ.append([frac, rs[0][1], name] + [f"{m[k]:.4f}" for k in range(a.shape[1])] +
                    [f"{s[k]:.4f}" for k in range(a.shape[1])])
    sh = ["fraction", "n_train", "param"] + [f"{h}_mean" for h in header[4:]] + [f"{h}_sd" for h in header[4:]]
    with open(os.path.join(args.out, "learning_curve_summary.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(sh) + "\n")
        for r in summ:
            f.write(",".join(str(v) for v in r) + "\n")

    make_figures(rows, sorted(set(args.fractions)), args.seeds, args.out)
    fracs = sorted(set(args.fractions))

    # ---------------- Bootstrap trên tập kiểm tra (kích thước dữ liệu đầy đủ, gộp các hạt giống) ----------------
    full = max(fracs)
    se_m = np.mean([store[(full, s)][0] for s in range(args.seeds)], axis=0)   # (n_te, 6)
    se_n = np.mean([store[(full, s)][1] for s in range(args.seeds)], axis=0)
    rng = np.random.default_rng(2024)
    idx = rng.integers(0, n_te, size=(args.boot, n_te))
    brow = []
    for i in range(R.M):
        rm = np.sqrt(se_m[idx, i].mean(axis=1)) / 2
        rn = np.sqrt(se_n[idx, i].mean(axis=1)) / 2
        diff = rm - rn
        lo, hi = np.percentile(diff, [2.5, 97.5])
        brow.append([R.NAMES[i], f"{np.sqrt(se_m[:, i].mean()) / 2:.4f}", f"{np.sqrt(se_n[:, i].mean()) / 2:.4f}",
                     f"{np.sqrt(se_m[:, i].mean()) / 2 - np.sqrt(se_n[:, i].mean()) / 2:.4f}",
                     f"{lo:.4f}", f"{hi:.4f}", f"{(diff > 0).mean():.3f}"])
    with open(os.path.join(args.out, "bootstrap_rmse_diff.csv"), "w", encoding="utf-8") as f:
        f.write("param,rmse_rel_mlp,rmse_rel_nre,diff_mlp_minus_nre,ci95_lo,ci95_hi,prob_nre_better\n")
        for r in brow:
            f.write(",".join(r) + "\n")

    print("\nBootstrap (tập kiểm tra, quy mô dữ liệu đầy đủ): chênh lệch RMSE/khoảng prior = MLP - mNRE", flush=True)
    for r in brow:
        sig = "có ý nghĩa" if (float(r[4]) > 0 or float(r[5]) < 0) else "chưa phân biệt được với 0"
        print(f"  {r[0]:10s} chênh lệch={r[3]} | KTC 95% [{r[4]}, {r[5]}] | P(mNRE tốt hơn)={r[6]} -> {sig}", flush=True)
    print(f"\nXong. Kết quả nằm trong: {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()