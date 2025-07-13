"""
Navier–Stokes URT★ RTX3080高機能実装
- Runge-Kutta/Crank-Nicolson/Euler法選択可
- URT展開・高次Moyal積・Optuna最適化
- 主要物理量ログ・可視化・自動リカバリ
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import os
import time
import pickle
import sys
import optuna
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.core import navier_stokes_urt_star as nsu

import matplotlib
matplotlib.use('Agg')  # これを必ず先に！

# --- 設定ファイル読込 ---
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# --- シミュレーション本体 ---
def run_simulation(config):
    # パラメータ展開
    nx, ny, nz = config['nx'], config['ny'], config['nz']
    nu = config['nu']
    dt = config['dt']
    nsteps = config['nsteps']
    theta = config['theta']
    method = config['method']
    moyal_order = config.get('moyal_order', 1)
    urt_modes = cp.array(config['urt_modes'])
    # 格子生成
    x = cp.linspace(0, 2*np.pi, nx)
    y = cp.linspace(0, 2*np.pi, ny)
    z = cp.linspace(0, 2*np.pi, nz)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    x_grid = X  # 1D展開用
    # 初期速度場
    u = cp.random.normal(0, 0.1, (nx, ny, nz)).astype(cp.float32)
    p = cp.zeros((nx, ny, nz), dtype=cp.float32)
    # 進化関数選択
    if method == 'rk4':
        step_func = nsu.rk4_step
    elif method == 'euler':
        step_func = nsu.euler_step
    elif method == 'crank-nicolson':
        step_func = nsu.crank_nicolson_step
    else:
        raise ValueError('Unknown method')
    # 右辺関数
    def rhs(u, p):
        # 拡散＋Moyal積非線形項（簡易例）
        lap = sum(cp.gradient(cp.gradient(u, axis=i), axis=i) for i in range(3))
        nonlinear = nsu.moyal_star_product(u, u, theta, order=moyal_order)
        return nu * lap - nonlinear
    # ログ
    energy_log = []
    vorticity_log = []
    # tqdm進捗
    for step in tqdm(range(nsteps)):
        u = step_func(u, p, rhs, dt)
        if step % 10 == 0:
            energy_log.append(nsu.calc_energy(u))
            vorticity_log.append(nsu.calc_vorticity(u))
        # チェックポイント・異常終了対応は省略（本番で追加）
    # URT展開
    coeffs = nsu.urt_expand(u, urt_modes, x_grid)
    # 可視化
    plt.figure()
    plt.plot(np.arange(len(energy_log)), energy_log)
    plt.title('Total Energy (English)')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.savefig('energy_log.png')
    plt.close()
    # URTモードスペクトル
    plt.figure()
    plt.plot(cp.asnumpy(urt_modes), cp.asnumpy(cp.abs(coeffs)))
    plt.title('URT Mode Spectrum (English)')
    plt.xlabel('Riemann Zero')
    plt.ylabel('Amplitude')
    plt.savefig('urt_mode_spectrum.png')
    plt.close()
    # ログ保存
    np.savetxt('energy_log.csv', np.array(energy_log), delimiter=',')
    np.savetxt('vorticity_log.csv', np.array(vorticity_log), delimiter=',')
    return u, coeffs

# --- メイン ---
if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'navier_stokes_urt_star_config.yaml')
    config = load_config(config_path)
    run_simulation(config) 