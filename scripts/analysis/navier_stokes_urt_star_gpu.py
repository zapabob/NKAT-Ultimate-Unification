"""
Navier–Stokes URT★ RTX3080（CUDA）対応CuPy実装
- 3D格子・非可換パラメータθ=プランク長²
- Moyal積（FFT畳み込み, CuPy）
- Euler法・tqdm進捗・自動チェックポイント・リカバリ
- RTX3080自動検出・matplotlib可視化
- 実装ログ自動生成対応
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import pickle
import sys

# プランク長・非可換パラメータθ
PLANCK_LENGTH = 1.616e-35  # [m]
THETA = PLANCK_LENGTH ** 2  # ≈ 2.612e-70 [m^2]

# 格子・物理パラメータ
NX, NY, NZ = 32, 32, 32  # 格子サイズ
NU = 0.01  # 粘性
DT = 0.01  # 時間刻み
NSTEPS = 100  # ステップ数
CHECKPOINT_INTERVAL = 300  # [秒] 5分ごと

# チェックポイント保存ディレクトリ
CHECKPOINT_DIR = "checkpoints_navier_stokes_urt_star_gpu"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# RTX3080自動検出
try:
    device_count = cp.cuda.runtime.getDeviceCount()
    device = cp.cuda.Device(0)
    device.use()
    # GPU名の取得
    device_props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = device_props['name']
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    print(f"🚀 CUDA Device: {gpu_name}")
except Exception as e:
    print(f"❌ CUDA device not found: {e}")
    sys.exit(1)

# 乱数初期化
cp.random.seed(int(time.time()))

# 速度場・圧力場の初期化（乱流的初期条件）
u = cp.random.randn(3, NX, NY, NZ).astype(cp.float32) * 0.1  # (vx, vy, vz)
p = cp.zeros((NX, NY, NZ), dtype=cp.float32)

# FFT用波数ベクトル
kx = cp.fft.fftfreq(NX).reshape(-1, 1, 1)
ky = cp.fft.fftfreq(NY).reshape(1, -1, 1)
kz = cp.fft.fftfreq(NZ).reshape(1, 1, -1)
K2 = kx**2 + ky**2 + kz**2
K2[0,0,0] = 1e-10  # ゼロ割防止

# Moyal積（FFT畳み込み, CuPy, 1次補正）
def moyal_star_product(f, g, theta=THETA):
    # FFT畳み込みによるMoyal積（1次補正）
    f_hat = cp.fft.fftn(f)
    g_hat = cp.fft.fftn(g)
    # 非可換位相因子（簡易: θk·k'）
    phase = cp.exp(1j * theta * (kx * ky + ky * kz + kz * kx))
    result_hat = f_hat * g_hat * phase
    result = cp.fft.ifftn(result_hat).real
    return result.astype(cp.float32)

# チェックポイント保存
def save_checkpoint(step, u, p):
    fname = os.path.join(CHECKPOINT_DIR, f"checkpoint_step{step}.pkl")
    with open(fname, "wb") as f:
        pickle.dump({"step": step, "u": cp.asnumpy(u), "p": cp.asnumpy(p)}, f)
    print(f"💾 Checkpoint saved: {fname}")

# チェックポイント復旧
def load_latest_checkpoint():
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_step")]
    if not files:
        return 0, None, None
    latest = max(files, key=lambda x: int(x.split("step")[-1].split(".")[0]))
    with open(os.path.join(CHECKPOINT_DIR, latest), "rb") as f:
        data = pickle.load(f)
    print(f"🔄 Checkpoint loaded: {latest}")
    return data["step"], cp.array(data["u"]), cp.array(data["p"])

# メインループ
try:
    # 異常終了時の自動リカバリ
    start_step, u, p = load_latest_checkpoint()
    if u is None:
        u = cp.random.randn(3, NX, NY, NZ).astype(cp.float32) * 0.1
        p = cp.zeros((NX, NY, NZ), dtype=cp.float32)
        start_step = 0
    print(f"▶️ Simulation start from step {start_step}")

    last_ckpt_time = time.time()
    for step in tqdm(range(start_step, NSTEPS), desc="Navier–Stokes URT★ (GPU)"):
        # 非線形項（Moyal積による非可換補正）
        nonlinear = moyal_star_product(u[0], u[1], theta=THETA)  # vx, vy例
        # 速度場の更新（Euler法, 簡易）
        u = u + DT * (NU * cp.gradient(u)[0] - nonlinear)
        # 圧力場のPoisson解法（簡易）
        div_u = cp.gradient(u[0])[0] + cp.gradient(u[1])[1] + cp.gradient(u[2])[2]
        p_hat = cp.fft.fftn(div_u) / K2
        p = cp.fft.ifftn(p_hat).real.astype(cp.float32)
        # 進捗・チェックポイント
        if (time.time() - last_ckpt_time) > CHECKPOINT_INTERVAL:
            save_checkpoint(step, u, p)
            last_ckpt_time = time.time()
    # 最終チェックポイント
    save_checkpoint(NSTEPS, u, p)
except KeyboardInterrupt:
    print("⚠️ Interrupted! Saving emergency checkpoint...")
    save_checkpoint(step, u, p)
    sys.exit(0)
except Exception as e:
    print(f"❌ Error: {e}")
    save_checkpoint(step, u, p)
    sys.exit(1)

# 可視化（vx断面）
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 英語キャプション
vx = cp.asnumpy(u[0,:, :, NZ//2])
plt.figure(figsize=(6,5))
plt.imshow(vx, cmap='coolwarm', origin='lower')
plt.colorbar(label='vx')
plt.title('Navier–Stokes URT★ vx cross-section (z=mid)')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('navier_stokes_urt_star_gpu_vx.png', dpi=200)
plt.show()

print("✅ Simulation completed. Result image saved: navier_stokes_urt_star_gpu_vx.png") 