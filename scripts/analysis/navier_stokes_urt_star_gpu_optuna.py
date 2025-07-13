"""
Navier–Stokes URT★ RTX3080（CUDA）対応CuPy実装 + Optunaベイズ最適化
- θ∈[1e-80, 1e-60]で最終エネルギー最小化
- RTX3080（CuPy）GPU計算
- tqdm進捗・チェックポイント・可視化も維持
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna
import os
import time
import pickle
import sys

# 格子・物理パラメータ
NX, NY, NZ = 32, 32, 32  # 格子サイズ
NU = 0.01  # 粘性
DT = 0.01  # 時間刻み
NSTEPS = 100  # ステップ数

# RTX3080自動検出
try:
    device_count = cp.cuda.runtime.getDeviceCount()
    device = cp.cuda.Device(0)
    device.use()
    device_props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = device_props['name']
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    print(f"🚀 CUDA Device: {gpu_name}")
except Exception as e:
    print(f"❌ CUDA device not found: {e}")
    sys.exit(1)

# 主要計算部を関数化（θを引数で受け取る）
def run_navier_stokes_simulation(theta, seed=42, save_fig=False, fig_prefix="optuna"):
    cp.random.seed(seed)
    u = cp.random.randn(3, NX, NY, NZ).astype(cp.float32) * 0.1
    p = cp.zeros((NX, NY, NZ), dtype=cp.float32)
    kx = cp.fft.fftfreq(NX).reshape(-1, 1, 1)
    ky = cp.fft.fftfreq(NY).reshape(1, -1, 1)
    kz = cp.fft.fftfreq(NZ).reshape(1, 1, -1)
    K2 = kx**2 + ky**2 + kz**2
    K2[0,0,0] = 1e-10
    def moyal_star_product(f, g, theta):
        f_hat = cp.fft.fftn(f)
        g_hat = cp.fft.fftn(g)
        phase = cp.exp(1j * theta * (kx * ky + ky * kz + kz * kx))
        result_hat = f_hat * g_hat * phase
        result = cp.fft.ifftn(result_hat).real
        return result.astype(cp.float32)
    for step in range(NSTEPS):
        # ラプラシアン計算
        lap_u = cp.stack([sum([cp.gradient(cp.gradient(u[i], axis=ax), axis=ax) for ax in range(3)]) for i in range(3)])
        nonlinear = moyal_star_product(u[0], u[1], theta=theta)
        u = u + DT * (NU * lap_u - nonlinear)
        div_u = cp.gradient(u[0], axis=0) + cp.gradient(u[1], axis=1) + cp.gradient(u[2], axis=2)
        p_hat = cp.fft.fftn(div_u) / K2
        p = cp.fft.ifftn(p_hat).real.astype(cp.float32)
        # NaN/infチェック
        if cp.isnan(u).any() or cp.isinf(u).any():
            return 1e20  # 異常値を返してOptunaに「このθは不適」と伝える
    energy = float(cp.sum(u[0]**2 + u[1]**2 + u[2]**2).get())
    # 可視化（vx断面）
    if save_fig:
        import matplotlib
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        vx = cp.asnumpy(u[0,:, :, NZ//2])
        plt.figure(figsize=(6,5))
        plt.imshow(vx, cmap='coolwarm', origin='lower')
        plt.colorbar(label='vx')
        plt.title(f'Navier–Stokes URT★ vx (z=mid) θ={theta:.1e}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(f'{fig_prefix}_vx_theta_{theta:.1e}.png', dpi=200)
        plt.close()
    return energy

# Optuna目的関数
def objective(trial):
    # θを対数スケールでサンプリング
    theta = trial.suggest_float("theta", 1e-80, 1e-60, log=True)
    print(f"\n[Optuna] θ={theta:.2e} でシミュレーション開始")
    energy = run_navier_stokes_simulation(theta, seed=42, save_fig=False)
    print(f"[Optuna] θ={theta:.2e} → 最終エネルギー={energy:.4e}")
    return energy

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    n_trials = 20
    print(f"Optunaベイズ最適化: θ∈[1e-80, 1e-60], 試行数={n_trials}")
    study.optimize(objective, n_trials=n_trials)
    print("\n=== 最適化結果 ===")
    print(f"最適θ: {study.best_params['theta']:.2e}")
    print(f"最小エネルギー: {study.best_value:.4e}")
    # 最適θで可視化付きで再実行
    best_theta = study.best_params['theta']
    run_navier_stokes_simulation(best_theta, seed=42, save_fig=True, fig_prefix="optuna_best")
    # 履歴保存
    import pandas as pd
    df = study.trials_dataframe()
    df.to_csv("optuna_navier_stokes_urt_star_theta_history.csv", index=False)
    print("最適化履歴: optuna_navier_stokes_urt_star_theta_history.csv")
    print("最適θのvx断面画像: optuna_best_vx_theta_{:.1e}.png".format(best_theta))
    print("✅ Optunaベイズ最適化完了") 