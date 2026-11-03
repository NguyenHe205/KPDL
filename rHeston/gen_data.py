#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinh dữ liệu (theta, bề mặt IV 63 chiều) cho mô hình rough Heston.

Dùng hàm Fourier trong repo của Simon Breneis
(approximations_to_fractional_stochastic_volterra_equations), KHÔNG sửa file nào trong repo.
Script tự xử lý ba lỗi môi trường đã gặp khi chạy repo:
  1. orthopy / quadpy (đòi giấy phép)  -> thay bằng hai hàm tự viết (Chebyshev + Golub-Welsch)
  2. rpy2 / mlfr (cần R)               -> thay bằng stub (chỉ HQE của repo mới cần)
  3. numpy >= 2.4 không có np.trapz    -> gán np.trapz = np.trapezoid

Cách chạy (xem README trong tin nhắn):
  python gen_data.py --repo <thư mục repo Python> --selftest
  python gen_data.py --repo <thư mục repo Python> --n 20 --out data/test --workers 2
  python gen_data.py --repo <thư mục repo Python> --n 5000 --out data/rh5k --workers 6

Chia tập: mỗi mẫu được gán ngẫu nhiên (theo chỉ số) vào train/val/test với xác suất 90/5/5.
Quy ước đầu ra: x có 63 cột, thứ tự cột = kỳ hạn (7) x moneyness z (9), theo hàng:
  x[:, i*9 + j] = IV tại kỳ hạn T[i], moneyness chuẩn hóa z[j]
"""
import argparse
import csv
import os
import sys
import time
import types
from collections import deque
import multiprocessing as mp
from multiprocessing.connection import wait

# Mỗi tiến trình con chỉ dùng 1 luồng tính toán; nếu không, nhiều tiến trình x nhiều luồng BLAS
# sẽ tranh nhau CPU và làm mỗi mẫu chậm đi nhiều lần. Phải đặt TRƯỚC khi import numpy.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

# ----------------------------------------------------------------------------------
# Lưới quan sát (theo bài báo, mục 5.1.1) và khoảng tham số (Table 1 của bài báo)
# ----------------------------------------------------------------------------------
Z = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
T = np.array([5 / 252, 10 / 252, 1 / 12, 1 / 4, 1 / 2, 1.0, 2.0])
S0 = 100.0
PARAM_NAMES = ["kappa", "theta_bar", "nu", "rho", "V0", "H"]  # theta_bar = phương sai dài hạn

PRIORS = {
    # (kappa, theta_bar, nu, rho, V0, H)
    "paper": (np.array([0.10, 0.01, 0.05, -0.95, 0.01, 0.05]),
              np.array([2.00, 0.25, 0.45, -0.10, 0.25, 0.45])),
    # Khoảng hẹp hơn ở vùng khó (H, V0, theta_bar nhỏ) - phương pháp Fourier hay chậm/treo ở đó
    "narrow": (np.array([0.10, 0.02, 0.05, -0.95, 0.02, 0.10]),
               np.array([2.00, 0.25, 0.45, -0.10, 0.25, 0.45])),
}


# ----------------------------------------------------------------------------------
# Chuẩn bị repo (stub + vá lỗi)
# ----------------------------------------------------------------------------------
def _chebyshev(moments):
    """Thuật toán Chebyshev: moment thô -> hệ số truy hồi (thay orthopy.tools.chebyshev)."""
    mu = np.asarray(moments, dtype=float)
    n = len(mu) // 2
    alpha = np.zeros(n)
    beta = np.zeros(n)
    alpha[0] = mu[1] / mu[0]
    beta[0] = mu[0]
    sig_prev = np.zeros(2 * n)
    sig = mu.copy()
    for k in range(1, n):
        new = np.zeros(2 * n)
        for l in range(k, 2 * n - k):
            new[l] = sig[l + 1] - alpha[k - 1] * sig[l] - beta[k - 1] * sig_prev[l]
        alpha[k] = new[k + 1] / new[k] - sig[k] / sig[k - 1]
        beta[k] = new[k] / sig[k - 1]
        sig_prev, sig = sig, new
    return alpha, beta, mu[0]


def _scheme_from_rc(alpha, beta, int_1, mode="numpy"):
    """Golub-Welsch: hệ số truy hồi -> nút và trọng số (thay quadpy.tools.scheme_from_rc)."""
    from scipy.linalg import eigh_tridiagonal
    x, v = eigh_tridiagonal(np.asarray(alpha, float), np.sqrt(np.asarray(beta[1:], float)))
    return x, int_1 * v[0] ** 2


def _not_available(*args, **kwargs):
    raise NotImplementedError("Hàm này cần R/rpy2 (Mittag-Leffler); script này không dùng đến.")


def setup_repo(repo):
    """Thêm repo vào sys.path, cài stub, vá numpy. Gọi trong mỗi tiến trình."""
    repo = os.path.abspath(repo)
    if not os.path.isfile(os.path.join(repo, "rHestonFourier.py")):
        raise FileNotFoundError(f"Không thấy rHestonFourier.py trong: {repo}")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    orthopy = types.ModuleType("orthopy")
    orthopy.tools = types.SimpleNamespace(chebyshev=_chebyshev)
    quadpy = types.ModuleType("quadpy")
    quadpy.tools = types.SimpleNamespace(scheme_from_rc=_scheme_from_rc)
    sys.modules["orthopy"] = orthopy
    sys.modules["quadpy"] = quadpy
    sys.modules["mlfr"] = types.SimpleNamespace(mittag_leffler=_not_available)
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    import warnings
    warnings.filterwarnings("ignore")


def compute_surface(theta, n_nodes=6, rel_tol=1e-3):
    """Bề mặt IV 7x9 cho một vector theta = (kappa, theta_bar, nu, rho, V0, H)."""
    import rHestonFourier as rf
    kappa, theta_bar, nu, rho, v0, h = [float(v) for v in theta]
    common = dict(S_0=S0, lambda_=kappa, rho=rho, nu=nu, theta=theta_bar, V_0=v0,
                  T=T, N=n_nodes, H=h, implied_vol=True, rel_tol=rel_tol)
    # Bước 1: IV tại giá thực thi at-the-money cho từng kỳ hạn
    atm = np.asarray(rf.eur_call_put(K=S0 * np.ones((len(T), 1)), **common)).reshape(len(T))
    # Bước 2: giá thực thi ứng với z = log(K/F0) / (sigma_ATM * sqrt(T)) (lãi suất 0 nên F0 = S0)
    strikes = S0 * np.exp(np.outer(atm * np.sqrt(T), Z))
    iv = np.asarray(rf.eur_call_put(K=strikes, **common))
    if iv.shape != (len(T), len(Z)) or not np.all(np.isfinite(iv)) or iv.min() <= 0:
        raise ValueError("IV không hợp lệ (NaN hoặc <= 0)")
    return iv


# ----------------------------------------------------------------------------------
# Tiến trình con: nhận (idx, theta), trả về kết quả
# ----------------------------------------------------------------------------------
def worker_main(conn, repo, n_nodes, rel_tol):
    try:
        setup_repo(repo)
    except Exception as e:  # lỗi thiết lập: báo ngay rồi thoát
        conn.send((-1, "setup_error", repr(e)))
        return
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        idx, theta = msg
        t0 = time.time()
        try:
            iv = compute_surface(theta, n_nodes, rel_tol)
            conn.send((idx, "ok", iv.reshape(-1), time.time() - t0))
        except Exception as e:
            conn.send((idx, "error", repr(e)[:120], time.time() - t0))


class Slot:
    def __init__(self, ctx, repo, n_nodes, rel_tol):
        self.parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=worker_main, args=(child, repo, n_nodes, rel_tol), daemon=True)
        self.proc.start()
        child.close()
        self.task = None
        self.t0 = 0.0

    def kill(self):
        try:
            self.proc.terminate()
            self.proc.join(2)
        except Exception:
            pass
        try:
            self.parent.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------------
# Ghi/đọc CSV (có thể chạy tiếp nếu bị dừng giữa chừng)
# ----------------------------------------------------------------------------------
def csv_header():
    return (["idx", "status", "seconds"] + PARAM_NAMES + [f"iv_{i}" for i in range(len(T) * len(Z))])


def read_done(path):
    done = set()
    if os.path.isfile(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done.add(int(row["idx"]))
    return done


def drop_failed(path):
    """Xóa các dòng không phải 'ok' khỏi CSV để chạy lại đúng các mẫu đó. Trả về số dòng đã xóa."""
    if not os.path.isfile(path):
        return 0
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0
    header, body = rows[0], rows[1:]
    keep = [r for r in body if r[1] == "ok"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(keep)
    return len(body) - len(keep)


def draw_theta(seed, idx, lo, hi):
    rng = np.random.default_rng([seed, idx])
    return lo + (hi - lo) * rng.random(6)


def run(args):
    lo, hi = PRIORS[args.prior]
    csv_path = args.out + ".csv"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if args.retry_failed:
        n_drop = drop_failed(csv_path)
        print(f"Đã bỏ {n_drop} dòng lỗi/quá giờ để chạy lại (các dòng ok được giữ nguyên).", flush=True)
    done = read_done(csv_path)
    pending = deque(i for i in range(args.n) if i not in done)
    print(f"Prior: {args.prior} | cần {args.n} mẫu | đã có {len(done)} | còn {len(pending)} | "
          f"workers={args.workers} | giới hạn {args.timeout}s/mẫu | N nút Markov={args.nodes}", flush=True)
    if not pending:
        return
    new_file = not os.path.isfile(csv_path)
    fh = open(csv_path, "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(csv_header())
        fh.flush()

    ctx = mp.get_context("spawn")  # giống nhau trên Windows/Linux/macOS
    slots = [Slot(ctx, args.repo, args.nodes, args.rel_tol) for _ in range(args.workers)]
    counts = {"ok": 0, "error": 0, "timeout": 0}
    n_todo = len(pending)
    t_start = time.time()
    nan_iv = [float("nan")] * (len(T) * len(Z))

    def record(idx, status, seconds, iv=None):
        theta = draw_theta(args.seed, idx, lo, hi)
        row = [idx, status, f"{seconds:.2f}"] + [f"{v:.8g}" for v in theta]
        row += [f"{v:.8g}" for v in (iv if iv is not None else nan_iv)]
        writer.writerow(row)
        fh.flush()
        counts[status if status in counts else "error"] += 1
        total = sum(counts.values())
        if total % max(1, args.report_every) == 0 or total == n_todo:
            el = time.time() - t_start
            print(f"[{total}/{n_todo}] ok={counts['ok']} lỗi={counts['error']} "
                  f"quá giờ={counts['timeout']} | {el / 60:.1f} phút", flush=True)

    try:
        while True:
            for k, s in enumerate(slots):
                if s.task is None and pending:
                    idx = pending.popleft()
                    s.parent.send((idx, draw_theta(args.seed, idx, lo, hi)))
                    s.task, s.t0 = idx, time.time()
            busy = [s for s in slots if s.task is not None]
            if not busy:
                break
            ready = wait([s.parent for s in busy], timeout=1.0)
            now = time.time()
            for k, s in enumerate(slots):
                if s.task is None:
                    continue
                if s.parent in ready:
                    try:
                        msg = s.parent.recv()
                    except (EOFError, OSError):
                        record(s.task, "error", now - s.t0)
                        s.kill()
                        slots[k] = Slot(ctx, args.repo, args.nodes, args.rel_tol)
                        continue
                    if msg[1] == "setup_error":
                        raise RuntimeError("Lỗi thiết lập repo: " + str(msg[2]))
                    idx, status, payload, secs = msg
                    record(idx, status, secs, payload if status == "ok" else None)
                    s.task = None
                elif now - s.t0 > args.timeout:
                    record(s.task, "timeout", now - s.t0)
                    s.kill()
                    slots[k] = Slot(ctx, args.repo, args.nodes, args.rel_tol)
    finally:
        for s in slots:
            try:
                s.parent.send(None)
            except Exception:
                pass
            s.kill()
        fh.close()
    print("Xong bước sinh.", counts, flush=True)


def finalize(args):
    """Đọc CSV, giữ các mẫu 'ok', chia 90/5/5 và lưu .npz."""
    csv_path = args.out + ".csv"
    rows = []
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    status = np.array([r[1] for r in body])
    ok = status == "ok"
    ids = np.array([int(r[0]) for r in body])
    theta = np.array([[float(v) for v in r[3:9]] for r in body])
    x = np.array([[float(v) for v in r[9:]] for r in body])
    print(f"Tổng {len(body)} dòng: ok={ok.sum()}, lỗi={(status == 'error').sum()}, "
          f"quá giờ={(status == 'timeout').sum()}")
    theta, x, ids = theta[ok], x[ok], ids[ok]
    n = len(theta)
    # Chia tập theo TỪNG chỉ số mẫu (không phụ thuộc tổng số mẫu), nên khi sinh thêm dữ liệu
    # thì mẫu cũ không đổi tập -> không bị rò rỉ giữa train/val/test giữa các phiên bản.
    split = np.zeros(n, dtype=np.int8)  # 0 = train, 1 = val, 2 = test
    for k, i in enumerate(ids):
        u = np.random.default_rng([args.seed + 1, int(i)]).random()
        split[k] = 0 if u < 0.90 else (1 if u < 0.95 else 2)
    np.savez(args.out + ".npz", theta=theta, x=x, idx=ids, split=split,
             param_names=np.array(PARAM_NAMES), T=T, z=Z)
    print(f"Đã lưu {args.out}.npz: theta {theta.shape}, x {x.shape}; "
          f"train/val/test = {(split == 0).sum()}/{(split == 1).sum()}/{(split == 2).sum()}")
    if n:
        print(f"IV nhỏ nhất/lớn nhất: {x.min():.3f} / {x.max():.3f}")


def selftest(args):
    """Chạy một theta cố định để so với giá trị tham chiếu."""
    setup_repo(args.repo)
    theta = np.array([1.0, 0.10, 0.30, -0.70, 0.05, 0.15])
    t0 = time.time()
    iv = compute_surface(theta, args.nodes, args.rel_tol)
    print(f"theta = {theta} (kappa, theta_bar, nu, rho, V0, H)")
    print(f"Thời gian: {time.time() - t0:.1f}s | hình dạng bề mặt: {iv.shape} (kỳ hạn x moneyness)")
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    print("Hàng T = 1 năm (z từ -2.5 đến 1.5):")
    print(iv[5])
    print("IV at-the-money (z = 0) theo 7 kỳ hạn:")
    print(iv[:, 5])


def main():
    p = argparse.ArgumentParser(description="Sinh dữ liệu IV rough Heston bằng Fourier")
    p.add_argument("--repo", required=True, help="Thư mục repo Python đã giải nén")
    p.add_argument("--out", default="data/rh", help="Tiền tố file đầu ra (.csv và .npz)")
    p.add_argument("--n", type=int, default=20, help="Số vector theta cần sinh")
    p.add_argument("--workers", type=int, default=2, help="Số tiến trình chạy song song")
    p.add_argument("--timeout", type=float, default=120.0, help="Giây tối đa cho mỗi mẫu")
    p.add_argument("--prior", choices=sorted(PRIORS), default="paper")
    p.add_argument("--nodes", type=int, default=6, help="Số nút của xấp xỉ Markov (6 chính xác, 3 nhanh hơn nhưng lệch)")
    p.add_argument("--rel-tol", type=float, default=1e-3, help="Sai số tương đối cho phép của IV")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report-every", type=int, default=5)
    p.add_argument("--retry-failed", action="store_true",
                   help="Chạy lại các mẫu trước đó bị lỗi hoặc quá giờ (dùng cùng --seed, --prior, --n)")
    p.add_argument("--selftest", action="store_true", help="Chạy thử một theta cố định rồi thoát")
    p.add_argument("--finalize-only", action="store_true", help="Chỉ đóng gói CSV thành .npz")
    args = p.parse_args()

    if args.selftest:
        selftest(args)
        return
    if not args.finalize_only:
        run(args)
    finalize(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()