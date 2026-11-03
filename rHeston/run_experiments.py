#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thực nghiệm thu nhỏ cho bài báo Brigo - Huser - Leonte (arXiv 2609.31570), dùng file .npz do gen_data.py sinh ra.

Gồm 4 phần (đều chỉ dùng scikit-learn, không cần PyTorch):
  A. Ước lượng điểm (mạng MLP, hàm mất mát MSE)               -> mốc so sánh
  B. Ước lượng tỉ số nơ-ron MARGINAL (mNRE): 6 bộ phân loại, mỗi bộ cho một tham số
     p(theta_i | x) = r_i(x, theta_i) * p(theta_i)            -> posterior biên của từng tham số
  C. Đánh giá: sai số hồi phục, độ phủ vùng HPD (calibration), mức co hẹp posterior
  D. Hellinger-SHAP: Kernel SHAP với nền cục bộ (100 bề mặt gần nhất) áp lên hai hàm thông tin
     g_H(x) = H^2(p_hat(theta_i|x), p(theta_i)) và g_KL(x) = KL(p_hat(theta_i|x) || p(theta_i))

Khác bài gốc (cần ghi vào phần hạn chế của tiểu luận):
  - Dùng mNRE (không dùng TRE), nên không có posterior đồng thời / lấy mẫu chung cho cả 6 tham số.
  - Mỗi tham số một bộ phân loại riêng (không dùng encoder dùng chung), mạng nhỏ hơn, dữ liệu ít hơn nhiều.
  - Không có mạng MAE, không có surrogate định giá exotic, không hiệu chỉnh sau huấn luyện.
  - Tích phân 1 chiều bằng quy tắc hình thang trên lưới đều (bài dùng Clenshaw-Curtis).
  - Tỉ số r_i được chuẩn hóa để posterior tích phân bằng 1.

Cách chạy:
  python run_experiments.py --data data/rh_v1.npz --out results
  python run_experiments.py --data data/rh_v1.npz --out results --reuse --n-shap 100   (dùng lại mô hình đã huấn luyện)
"""
import argparse
import copy
import itertools
import os
import pickle
import time
from math import comb

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier, MLPRegressor

# ----------------------------------------------------------------------------------
# Hằng số (khớp với gen_data.py và bảng Table 1 của bài báo)
# ----------------------------------------------------------------------------------
# Cảnh báo vô hại của sklearn khi khởi tạo bằng lô nhỏ (batch_size lớn hơn số mẫu khởi tạo)
warnings.filterwarnings("ignore", message=".*batch_size.*")

NAMES = ["kappa", "theta_bar", "nu", "rho", "V0", "H"]
LABELS = [r"$\kappa$", r"$\bar\theta$", r"$\nu$", r"$\rho$", r"$V_0$", r"$H$"]
LO = np.array([0.10, 0.01, 0.05, -0.95, 0.01, 0.05])
HI = np.array([2.00, 0.25, 0.45, -0.10, 0.25, 0.45])
T_LABELS = ["5/252", "10/252", "1/12", "1/4", "1/2", "1", "2"]
Z_VALUES = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
M = 6
N_PC = 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_u(theta):
    """Chuẩn hóa theta về [-1, 1] theo khoảng prior."""
    return 2.0 * (theta - LO) / (HI - LO) - 1.0


def from_u(u):
    return LO + (np.asarray(u) + 1.0) / 2.0 * (HI - LO)


# ----------------------------------------------------------------------------------
# Tiền xử lý: PCA 20 thành phần (chỉ khớp trên tập huấn luyện) rồi chuẩn hóa phương sai đơn vị
# ----------------------------------------------------------------------------------
class Preproc:
    def __init__(self, x_train):
        self.pca = PCA(n_components=N_PC).fit(x_train)
        self.std = self.pca.transform(x_train).std(axis=0)
        self.explained = float(self.pca.explained_variance_ratio_.sum())

    def __call__(self, x):
        return self.pca.transform(x) / self.std


# ----------------------------------------------------------------------------------
# Huấn luyện với early stopping thủ công (dùng tập kiểm định riêng, không rò rỉ)
# ----------------------------------------------------------------------------------
def train_loop(model, epoch_fn, val_fn, max_epochs, patience, tag):
    best, best_model, bad = np.inf, None, 0
    hist = []
    for ep in range(1, max_epochs + 1):
        epoch_fn(model)
        v = val_fn(model)
        hist.append(v)
        if v < best - 1e-6:
            best, best_model, bad = v, copy.deepcopy(model), 0
        else:
            bad += 1
            if bad >= patience:
                break
    log(f"  {tag}: dừng ở epoch {ep}, val loss tốt nhất = {best:.4f}")
    return best_model, hist


def logit_of(proba1):
    p = np.clip(proba1, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def train_point(z_tr, u_tr, z_va, u_va, hidden, args, rng):
    reg = MLPRegressor(hidden_layer_sizes=hidden, alpha=1e-4, batch_size=128, learning_rate_init=1e-3,
                       random_state=int(rng.integers(1 << 30)))
    reg.partial_fit(z_tr[:8], u_tr[:8])  # khởi tạo trọng số

    def epoch_fn(m):
        m.partial_fit(z_tr, u_tr)

    def val_fn(m):
        return float(np.mean((m.predict(z_va) - u_va) ** 2))

    return train_loop(reg, epoch_fn, val_fn, args.max_epochs, args.patience, "MLP điểm (MSE)")


def train_mnre(i, z_tr, u_tr, z_va, u_va, hidden, args, rng):
    """Bộ phân loại cho tham số i: phân biệt (x, theta_i) thật với (x, theta_i ~ U[-1,1])."""
    clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                        random_state=int(rng.integers(1 << 30)))
    n_tr, n_va = len(z_tr), len(z_va)
    ones = np.ones(n_tr)
    zeros = np.zeros(n_tr)
    # Mẫu âm cố định cho tập kiểm định
    neg_va = np.random.default_rng(12345 + i).uniform(-1, 1, size=n_va)
    xv = np.vstack([np.column_stack([z_va, u_va[:, i]]), np.column_stack([z_va, neg_va])])
    yv = np.concatenate([np.ones(n_va), np.zeros(n_va)])
    # Khởi tạo trọng số bằng một bước nhỏ (sklearn cần gọi partial_fit lần đầu với danh sách lớp)
    x0 = np.vstack([np.column_stack([z_tr[:8], u_tr[:8, i]]),
                    np.column_stack([z_tr[:8], rng.uniform(-1, 1, size=len(z_tr[:8]))])])
    clf.partial_fit(x0, np.array([1] * len(z_tr[:8]) + [0] * len(z_tr[:8])), classes=[0, 1])

    def epoch_fn(m):
        neg = rng.uniform(-1, 1, size=n_tr)  # mẫu âm được lấy mới ở mỗi epoch
        X = np.vstack([np.column_stack([z_tr, u_tr[:, i]]), np.column_stack([z_tr, neg])])
        y = np.concatenate([ones, zeros])
        m.partial_fit(X, y)

    def val_fn(m):
        p = np.clip(m.predict_proba(xv)[:, 1], 1e-12, 1 - 1e-12)
        return float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))

    return train_loop(clf, epoch_fn, val_fn, args.max_epochs, args.patience, f"mNRE {NAMES[i]}")


# ----------------------------------------------------------------------------------
# Posterior biên trên lưới và các đại lượng suy ra
# ----------------------------------------------------------------------------------
def grid_ratio(clf, z, grid, chunk_rows=200_000):
    """Trả về tỉ số r_hat(x, u) trên lưới, hình (n, G), chưa chuẩn hóa."""
    n, G = len(z), len(grid)
    out = np.empty((n, G))
    per = max(1, chunk_rows // G)
    for s in range(0, n, per):
        zz = z[s:s + per]
        rows = np.empty((len(zz) * G, z.shape[1] + 1))
        rows[:, :-1] = np.repeat(zz, G, axis=0)
        rows[:, -1] = np.tile(grid, len(zz))
        out[s:s + per] = np.exp(logit_of(clf.predict_proba(rows)[:, 1])).reshape(len(zz), G)
    return out


def posterior_density(r, grid):
    """Mật độ posterior trong không gian u (tích phân bằng 1 trên [-1, 1])."""
    area = trapezoid(r, grid, axis=1)
    return r / area[:, None]


def summarize_posterior(dens, grid):
    """Trung bình, trung vị, độ lệch chuẩn của posterior trong không gian u."""
    w = np.gradient(grid)
    mass = dens * w
    mass /= mass.sum(axis=1, keepdims=True)
    mean = mass @ grid
    sd = np.sqrt(np.maximum(mass @ grid ** 2 - mean ** 2, 0))
    cdf = np.cumsum(mass, axis=1)
    med = grid[np.argmax(cdf >= 0.5, axis=1)]
    return mean, med, sd, mass


def hpd_levels(mass, dens, grid, u_true):
    """Mức HPD nhỏ nhất chứa giá trị thật: alpha* = khối lượng của vùng có mật độ >= mật độ tại giá trị thật."""
    d_true = np.array([np.interp(u_true[k], grid, dens[k]) for k in range(len(u_true))])
    return np.array([mass[k][dens[k] >= d_true[k]].sum() for k in range(len(u_true))])


# ----------------------------------------------------------------------------------
# Hàm thông tin g_H, g_KL cho từng tham số (dùng cho SHAP)
# ----------------------------------------------------------------------------------
class InfoGame:
    def __init__(self, pre, clfs, grids):
        """grids: danh sách 6 lưới 1 chiều (mỗi tham số một lưới, vì độ rộng posterior khác nhau nhiều)."""
        self.pre, self.clfs, self.grids = pre, clfs, grids

    def __call__(self, x_raw, block=1500):
        """x_raw: (n, 63) -> (n, 12): 6 giá trị H^2 rồi 6 giá trị KL."""
        n = len(x_raw)
        out = np.empty((n, 2 * M))
        for s in range(0, n, block):
            z = self.pre(x_raw[s:s + block])
            for i in range(M):
                g = self.grids[i]
                r = grid_ratio(self.clfs[i], z, g)
                rt = r / (0.5 * trapezoid(r, g, axis=1))[:, None]  # chuẩn hóa: E_U[r] = 1
                out[s:s + block, i] = 1.0 - 0.5 * trapezoid(np.sqrt(rt), g, axis=1)
                out[s:s + block, M + i] = 0.5 * trapezoid(rt * np.log(np.maximum(rt, 1e-300)), g, axis=1)
        return out


# ----------------------------------------------------------------------------------
# Kernel SHAP (tự cài đặt, hỗ trợ nhiều đầu ra)
# ----------------------------------------------------------------------------------
def kernel_shap(value_fn, d, n_samples, rng, v_empty, v_full):
    """
    value_fn(masks[m, d] bool) -> (m, K). Trả về (d, K) giá trị Shapley xấp xỉ.
    Liệt kê đầy đủ các kích thước ở hai đầu (1 và d-1) nếu đủ ngân sách, phần còn lại lấy mẫu theo kernel Shapley.
    """
    sizes = np.arange(1, d)
    W = (d - 1) / (sizes * (d - sizes))
    W = W / W.sum()
    masks, weights = [], []
    budget = n_samples
    lo_s, hi_s = 1, d - 1
    while lo_s <= hi_s:
        n_comb = comb(d, lo_s) * (1 if lo_s == hi_s else 2)
        if n_comb > budget:
            break
        for sub in itertools.combinations(range(d), lo_s):
            mk = np.zeros(d, bool)
            mk[list(sub)] = True
            masks.append(mk)
            weights.append(W[lo_s - 1] / comb(d, lo_s))
            if lo_s != hi_s:
                masks.append(~mk)
                weights.append(W[hi_s - 1] / comb(d, hi_s))
        budget -= n_comb
        lo_s += 1
        hi_s -= 1
    if lo_s <= hi_s and budget > 1:
        mid_sizes = np.arange(lo_s, hi_s + 1)
        p = W[mid_sizes - 1]
        mass_left = p.sum()
        p = p / p.sum()
        n_pairs = max(1, budget // 2)
        w_each = mass_left / (2 * n_pairs)
        for _ in range(n_pairs):
            s = int(rng.choice(mid_sizes, p=p))
            mk = np.zeros(d, bool)
            mk[rng.permutation(d)[:s]] = True
            masks.append(mk)
            weights.append(w_each)
            masks.append(~mk)
            weights.append(w_each)
    masks = np.array(masks)
    weights = np.array(weights)
    v = value_fn(masks)                              # (m, K)
    delta = v_full - v_empty                          # (K,)
    zc = masks.astype(float)
    # Loại biến cuối bằng ràng buộc tổng phi = delta (tính hiệu quả)
    Xp = zc[:, :d - 1] - zc[:, [d - 1]]
    yp = (v - v_empty) - zc[:, [d - 1]] * delta[None, :]
    sw = np.sqrt(weights)[:, None]
    sol = np.linalg.lstsq(Xp * sw, yp * sw, rcond=None)[0]   # (d-1, K)
    phi = np.vstack([sol, (delta - sol.sum(axis=0))[None, :]])
    return phi


def make_value_fn(F, x_o, B, max_rows=6000):
    nb = len(B)
    d = len(x_o)

    def value_fn(masks):
        m = len(masks)
        per = max(1, max_rows // nb)
        outs = []
        for s in range(0, m, per):
            mm = masks[s:s + per]
            H = np.where(mm[:, None, :], x_o[None, None, :], B[None, :, :]).reshape(-1, d)
            outs.append(F(H).reshape(len(mm), nb, -1).mean(axis=1))
        return np.vstack(outs)

    return value_fn


# ----------------------------------------------------------------------------------
# Hình vẽ
# ----------------------------------------------------------------------------------
def plot_calibration(levels_per_param, path):
    nominal = np.linspace(0.02, 0.98, 49)
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(M):
        cov = [(levels_per_param[i] <= a).mean() for a in nominal]
        ax.plot(nominal, cov, label=LABELS[i])
    ax.plot([0, 1], [0, 1], "k--", label="lý tưởng")
    ax.set_xlabel("Mức HPD danh nghĩa")
    ax.set_ylabel("Độ phủ thực nghiệm (tập kiểm tra)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_posterior_example(grid, dens_list, u_true, means, path, title):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
    for i, ax in enumerate(axes.ravel()):
        th = LO[i] + (grid + 1) / 2 * (HI[i] - LO[i])
        pdf = dens_list[i] / ((HI[i] - LO[i]) / 2.0)
        ax.plot(th, pdf, label="posterior mNRE")
        ax.axhline(1.0 / (HI[i] - LO[i]), color="gray", ls=":", label="prior đều")
        ax.axvline(from_u(u_true)[i], color="r", label="giá trị thật")
        ax.axvline(from_u(means)[i], color="g", ls="--", label="trung bình posterior")
        ax.set_title(LABELS[i])
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shap_maps(maps, path, title):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for i, ax in enumerate(axes.ravel()):
        im = ax.imshow(maps[i], origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(9))
        ax.set_xticklabels([f"{z:g}" for z in Z_VALUES], fontsize=7)
        ax.set_yticks(range(7))
        ax.set_yticklabels(T_LABELS, fontsize=7)
        ax.set_xlabel("z (moneyness chuẩn hóa)")
        ax.set_ylabel("T (năm)")
        ax.set_title(LABELS[i])
        fig.colorbar(im, ax=ax)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Thực nghiệm thu nhỏ: ước lượng tỉ số marginal + Hellinger-SHAP")
    ap.add_argument("--data", required=True, help="File .npz từ gen_data.py")
    ap.add_argument("--out", default="results")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--grid", type=int, default=1025, help="Số điểm lưới 1 chiều khi đánh giá posterior (độ phủ, trung bình...)")
    ap.add_argument("--n-shap", type=int, default=50, help="Số bề mặt kiểm tra được giải thích bằng SHAP (0 = bỏ qua)")
    ap.add_argument("--bg", type=int, default=50, help="Số bề mặt nền cục bộ (bài dùng 100)")
    ap.add_argument("--shap-samples", type=int, default=512, help="Số tập liên minh cho Kernel SHAP")
    ap.add_argument("--shap-grid", type=int, default=33, help="Số điểm lưới TỐI THIỂU khi tính g trong SHAP")
    ap.add_argument("--shap-grid-max", type=int, default=257, help="Số điểm lưới TỐI ĐA cho tham số có posterior rất hẹp")
    ap.add_argument("--reuse", action="store_true", help="Dùng lại mô hình đã lưu trong thư mục --out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-bp", type=float, default=0.0,
                    help="Cộng nhiễu Gauss độc lập (độ lệch chuẩn tính bằng điểm cơ bản của IV) vào mọi bề mặt, "
                         "để mô phỏng nhiễu Monte Carlo của bài gốc (dùng thư mục --out riêng)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    hidden = tuple(args.hidden)

    d = np.load(args.data)
    theta, x, split = d["theta"], d["x"], d["split"]
    if args.noise_bp > 0:
        x = x + np.random.default_rng(args.seed + 99).normal(0.0, args.noise_bp * 1e-4, size=x.shape)
        log(f"Đã cộng nhiễu Gauss độc lập, độ lệch chuẩn {args.noise_bp:g} điểm cơ bản, vào mọi bề mặt IV")
    tr, va, te = split == 0, split == 1, split == 2
    log(f"Dữ liệu: {len(theta)} mẫu | train {tr.sum()} | val {va.sum()} | test {te.sum()}")
    if te.sum() < 20 or va.sum() < 20:
        log("CẢNH BÁO: tập kiểm định/kiểm tra rất nhỏ, kết quả sẽ nhiễu.")
    if theta.min(axis=0).min() < -1e9:
        raise ValueError("Dữ liệu không hợp lệ")
    u = to_u(theta)
    if (u.min() < -1.001) or (u.max() > 1.001):
        log("CẢNH BÁO: có theta nằm ngoài khoảng prior của bài (Table 1); kiểm tra lại prior đã dùng khi sinh dữ liệu.")

    pre = Preproc(x[tr])
    log(f"PCA: {N_PC} thành phần giải thích {100 * pre.explained:.5f}% phương sai")
    z_tr, z_va, z_te = pre(x[tr]), pre(x[va]), pre(x[te])
    u_tr, u_va, u_te = u[tr], u[va], u[te]
    grid = np.linspace(-1, 1, args.grid)

    model_path = os.path.join(args.out, "models.pkl")
    if args.reuse and os.path.isfile(model_path):
        with open(model_path, "rb") as f:
            point, clfs, hists = pickle.load(f)
        log("Đã nạp mô hình từ " + model_path)
    else:
        log("Huấn luyện mô hình...")
        t0 = time.time()
        point, h_point = train_point(z_tr, u_tr, z_va, u_va, hidden, args, rng)
        clfs, h_clf = [], []
        for i in range(M):
            c, h = train_mnre(i, z_tr, u_tr, z_va, u_va, hidden, args, rng)
            clfs.append(c)
            h_clf.append(h)
        hists = (h_point, h_clf)
        with open(model_path, "wb") as f:
            pickle.dump((point, clfs, hists), f)
        log(f"Huấn luyện xong sau {(time.time() - t0) / 60:.1f} phút")

    # ---------------- C1. Hồi phục tham số trên tập kiểm tra ----------------
    pred_point = from_u(point.predict(z_te))
    dens, means, meds, sds, masses = [], [], [], [], []
    for i in range(M):
        r = grid_ratio(clfs[i], z_te, grid)
        dd = posterior_density(r, grid)
        mean, med, sd, mass = summarize_posterior(dd, grid)
        dens.append(dd), means.append(mean), meds.append(med), sds.append(sd), masses.append(mass)
    th_te = theta[te]
    rng_param = HI - LO
    rows = []
    for i in range(M):
        mean_th = LO[i] + (means[i] + 1) / 2 * rng_param[i]
        med_th = LO[i] + (meds[i] + 1) / 2 * rng_param[i]
        for name, pred in (("MLP_MSE", pred_point[:, i]), ("mNRE_mean", mean_th), ("mNRE_median", med_th)):
            err = pred - th_te[:, i]
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            r2 = float(1 - np.sum(err ** 2) / np.sum((th_te[:, i] - th_te[:, i].mean()) ** 2))
            rows.append([NAMES[i], name, f"{rmse:.5g}", f"{mae:.5g}", f"{rmse / rng_param[i]:.4f}", f"{r2:.4f}"])
    write_csv(os.path.join(args.out, "metrics_recovery.csv"),
              ["param", "estimator", "RMSE", "MAE", "RMSE_over_prior_range", "R2"], rows)

    # ---------------- C2. Độ phủ HPD (calibration) ----------------
    levels = [hpd_levels(masses[i], dens[i], grid, u_te[:, i]) for i in range(M)]
    nominals = [0.5, 0.8, 0.9, 0.95]
    cov_rows = []
    for i in range(M):
        cov_rows.append([NAMES[i]] + [f"{(levels[i] <= a).mean():.3f}" for a in nominals])
    write_csv(os.path.join(args.out, "coverage_hpd.csv"), ["param"] + [f"nominal_{a}" for a in nominals], cov_rows)
    plot_calibration(levels, os.path.join(args.out, "fig_calibration.png"))

    # ---------------- C3. Mức co hẹp posterior so với prior ----------------
    prior_sd_u = np.sqrt(1.0 / 3.0)  # độ lệch chuẩn của U[-1, 1]
    contr_rows = [[NAMES[i], f"{sds[i].mean():.4f}", f"{prior_sd_u:.4f}", f"{(sds[i] / prior_sd_u).mean():.4f}"]
                  for i in range(M)]
    write_csv(os.path.join(args.out, "contraction.csv"),
              ["param", "mean_posterior_sd_u", "prior_sd_u", "mean_ratio_post_over_prior_sd"], contr_rows)

    k = 0
    plot_posterior_example(grid, [dens[i][k] for i in range(M)], u_te[k], np.array([means[i][k] for i in range(M)]),
                           os.path.join(args.out, "fig_posterior_example.png"), "Posterior biên cho một bề mặt IV kiểm tra")

    log("Kết quả hồi phục (RMSE / khoảng prior) và độ phủ HPD:")
    for i in range(M):
        e_mlp = np.sqrt(np.mean((pred_point[:, i] - th_te[:, i]) ** 2)) / rng_param[i]
        e_nre = np.sqrt(np.mean((LO[i] + (means[i] + 1) / 2 * rng_param[i] - th_te[:, i]) ** 2)) / rng_param[i]
        log(f"  {NAMES[i]:10s} MLP {e_mlp:.3f} | mNRE-mean {e_nre:.3f} | phủ 90%: {(levels[i] <= 0.9).mean():.2f} | "
            f"sd posterior/prior: {(sds[i] / prior_sd_u).mean():.2f}")

    # ---------------- D. Hellinger-SHAP ----------------
    if args.n_shap > 0:
        # Lưới phải đủ mịn so với độ rộng posterior: bước lưới h <= độ lệch chuẩn nhỏ (phân vị 5%) của posterior.
        n_grid = [int(np.clip(np.ceil(2.0 / max(np.percentile(sds[i], 5), 1e-4)) + 1, args.shap_grid, args.shap_grid_max))
                  for i in range(M)]
        log("Số điểm lưới SHAP theo tham số: " + ", ".join(f"{NAMES[i]}={n_grid[i]}" for i in range(M)))
        game = InfoGame(pre, clfs, [np.linspace(-1, 1, n) for n in n_grid])
        x_pool = x[tr]
        z_pool = z_tr
        te_idx = np.where(te)[0]
        pick = rng.choice(len(te_idx), size=min(args.n_shap, len(te_idx)), replace=False)
        sh_all = []
        t0 = time.time()
        for n_done, j in enumerate(pick, 1):
            xo = x[te_idx[j]]
            zo = pre(xo[None, :])[0]
            near = np.argsort(((z_pool - zo) ** 2).sum(axis=1))[:args.bg]
            B = x_pool[near]
            v_empty = game(B).mean(axis=0)
            v_full = game(xo[None, :])[0]
            phi = kernel_shap(make_value_fn(game, xo, B), x.shape[1], args.shap_samples, rng, v_empty, v_full)
            sh_all.append(phi)
            if n_done == 1 or n_done % 5 == 0 or n_done == len(pick):
                el = time.time() - t0
                log(f"  SHAP {n_done}/{len(pick)} | {el / n_done:.1f} s/bề mặt | còn khoảng {el / n_done * (len(pick) - n_done) / 60:.1f} phút")
        sh = np.array(sh_all)                         # (n, 63, 12)
        np.savez(os.path.join(args.out, "shap_values.npz"), shap=sh, test_pick=te_idx[pick])
        mean_abs = np.abs(sh).mean(axis=0)            # (63, 12)
        maps_H = [mean_abs[:, i].reshape(7, 9) for i in range(M)]
        maps_KL = [mean_abs[:, M + i].reshape(7, 9) for i in range(M)]
        plot_shap_maps(maps_H, os.path.join(args.out, "fig_shap_H2.png"),
                       f"Hellinger-SHAP: trung bình |SHAP| trên {len(pick)} bề mặt kiểm tra")
        plot_shap_maps(maps_KL, os.path.join(args.out, "fig_shap_KL.png"),
                       f"KL-SHAP: trung bình |SHAP| trên {len(pick)} bề mặt kiểm tra")
        half = len(sh) // 2
        sr = []
        for i in range(M):
            rho_s = spearmanr(mean_abs[:, i], mean_abs[:, M + i])[0]
            if half >= 2:  # độ ổn định: so hai nửa độc lập của các bề mặt được giải thích
                a = np.abs(sh[:half, :, i]).mean(axis=0)
                b = np.abs(sh[half:2 * half, :, i]).mean(axis=0)
                stab = f"{spearmanr(a, b)[0]:.3f}"
            else:
                stab = "nan"
            top = np.argsort(-mean_abs[:, i])[:5]
            top_txt = "; ".join(f"T={T_LABELS[t // 9]},z={Z_VALUES[t % 9]:g}" for t in top)
            sr.append([NAMES[i], f"{rho_s:.3f}", stab, top_txt])
        write_csv(os.path.join(args.out, "shap_summary.csv"),
                  ["param", "spearman_H2_vs_KL", "split_half_spearman_H2", "top5_features_H2"], sr)
        log("Tương quan hạng H^2 vs KL, độ ổn định chia đôi, và 5 ô quan trọng nhất theo H^2:")
        for r in sr:
            log(f"  {r[0]:10s} H2-vs-KL={r[1]} | ổn định={r[2]} | {r[3]}")
    log(f"Xong. Kết quả nằm trong thư mục: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()