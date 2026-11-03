For obtaining good Markovian approximations (good quadrature rules), call 
nodes, weights = RoughKernel.quadrature_rule(H, T, N)
where H is the Hurst parameter, T the maturity and N the approximating dimension (N=2 or N=3 is usually enough).

For efficiently simulating the Markovian approximation of the rough Heston model, call
rHestonMarkovSimulation.samples()
where one uses the nodes and weights generated before.

# Thực nghiệm rough Heston (môn Khai phá dữ liệu)

Mã sinh dữ liệu và chạy thực nghiệm thu nhỏ cho bài báo arXiv 2609.31570
(Brigo, Huser, Leonte: *Uncertainty and Explainability in Deep Rough Volatility*).

## Các file

| File | Việc |
|---|---|
| `gen_data.py` | Sinh dữ liệu (θ, bề mặt IV 63 chiều) bằng phương pháp Fourier của repo Breneis |
| `run_experiments.py` | Huấn luyện ước lượng điểm và mNRE, độ phủ HPD, Hellinger-SHAP |
| `run_extension.py` | Đường cong học, nhiều hạt giống, bootstrap |
| `requirements.txt` | Thư viện cần cài |
| `data/rh5k.csv` | **Tiến độ sinh dữ liệu** (các chỉ số đã xử lý). Bắt buộc phải có để chạy tiếp |

## 0. Quy tắc quan trọng

- **Chỉ MỘT máy được chạy bước 1 (sinh dữ liệu)** và ghi vào `data/rh5k.csv`. Hai máy cùng ghi sẽ làm hỏng tiến độ.
- **Không đổi** các tham số mặc định: `--seed 0`, `--prior paper`, `--timeout 120`, `--nodes 6`, `--rel-tol 1e-3`. Mỗi chỉ số mẫu ứng với một θ cố định theo hạt giống; đổi tham số sẽ làm lệch dữ liệu đã có.
- Không dùng `--retry-failed`.

## 1. Chuẩn bị (làm một lần)

1. Cài Python 3.11 trở lên.
2. Tải repo của Simon Breneis (không có trong repo này): https://github.com/SimonBreneis/approximations_to_fractional_stochastic_volterra_equations → Code → Download ZIP → giải nén **vào thư mục gốc của repo này**, ngang hàng với `gen_data.py`. Thư mục sau khi giải nén tên là `approximations_to_fractional_stochastic_volterra_equations-master`, và bên trong phải thấy ngay file `rHestonFourier.py` (nếu Windows tạo hai lớp thư mục cùng tên thì dùng lớp trong cùng).
3. Tạo môi trường và cài thư viện (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
4. Kiểm tra môi trường (khoảng 15–30 giây):
```powershell
python gen_data.py --repo ".\approximations_to_fractional_stochastic_volterra_equations-master" --selftest
```
Kết quả phải gần như sau:
```
Hàng T = 1 năm: [0.3428 0.3261 0.3086 0.2904 0.2715 0.2522 0.233 0.2154 0.2015]
IV at-the-money: [0.2206 0.222 0.2251 0.2337 0.242 0.2522 0.263]
```

## 2. Sinh dữ liệu đến 5.000 chỉ số

Chọn số tiến trình: **số tiến trình = phần nguyên của (RAM trống trong GB chia 1.3), tối đa 6, và nhỏ hơn số lõi vật lý**.
Mỗi tiến trình dùng khoảng 0.75 GB (tối đa khoảng 1.2 GB ở vùng tham số khó). Đóng ứng dụng nặng, cắm sạc, không để máy ngủ.

```powershell
python gen_data.py --repo ".\approximations_to_fractional_stochastic_volterra_equations-master" --n 5000 --out data\rh5k --workers 6 --report-every 20
```

- Lệnh tự bỏ qua các chỉ số đã có trong `data/rh5k.csv`; còn khoảng 2.070 chỉ số.
- Dừng bằng `Ctrl+C`, chạy lại đúng lệnh trên để tiếp tục.
- Trong 10–15 phút đầu hãy xem Task Manager: nếu RAM vượt khoảng 90% thì giảm `--workers`.
- Khoảng 60–65% chỉ số sẽ thành công, phần còn lại bị ghi là `timeout` hoặc `error` (đây là bình thường).

## 3. Đóng gói dữ liệu

```powershell
python gen_data.py --repo ".\approximations_to_fractional_stochastic_volterra_equations-master" --out data\rh5k --finalize-only
copy data\rh5k.npz data\rh_v2.npz
```

## 4. Chạy thực nghiệm (chạy lần lượt, không song song)

```powershell
python run_experiments.py --data data\rh_v2.npz --out results_v2 --n-shap 150 --bg 50
python run_extension.py --data data\rh_v2.npz --out results_ext_v2
python run_experiments.py --data data\rh_v2.npz --out results_noise_v2 --noise-bp 10 --n-shap 100 --bg 50
```

## 5. Gửi kết quả

Commit và push `data/rh5k.csv` (đã cập nhật), `data/rh_v2.npz`, và các thư mục `results_v2`, `results_ext_v2`, `results_noise_v2`
(chỉ cần các file `.csv` và `.png`; có thể bỏ file `models.pkl`).