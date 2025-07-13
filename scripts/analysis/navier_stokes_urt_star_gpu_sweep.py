import os
import itertools
import subprocess
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import time

# --- Sweepパラメータ設定 ---
THETA_LIST = [2.6e-70, 1e-68, 1e-66]
NU_LIST = [0.01, 0.001]
NX_LIST = [32, 64]
SEED_LIST = [42, 123]
METHOD_LIST = ['rk4', 'euler']
MOYAL_ORDER_LIST = [1, 2]

# --- テンプレート読込 ---
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'navier_stokes_urt_star_config.yaml')
with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    base_config = yaml.safe_load(f)

# --- Sweep組み合わせ生成 ---
param_grid = list(itertools.product(THETA_LIST, NU_LIST, NX_LIST, SEED_LIST, METHOD_LIST, MOYAL_ORDER_LIST))

# --- 結果保存 ---
results = []
RESULTS_DIR = 'Results/sweep_b'
os.makedirs(RESULTS_DIR, exist_ok=True)

for i, (theta, nu, nx, seed, method, moyal_order) in enumerate(tqdm(param_grid, desc='Sweep B')):
    config = base_config.copy()
    config['theta'] = theta
    config['nu'] = nu
    config['nx'] = config['ny'] = config['nz'] = nx
    config['method'] = method
    config['moyal_order'] = moyal_order
    config['nsteps'] = 100
    config['dt'] = 0.01
    np.random.seed(seed)
    # yaml保存
    config_path = os.path.join(RESULTS_DIR, f'config_{i}.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    # 実行
    cmd = ['py', '-3', 'navier_stokes_urt_star_gpu_advanced.py']
    env = os.environ.copy()
    env['CONFIG_PATH'] = config_path
    try:
        ret = subprocess.run(cmd, cwd=os.path.dirname(__file__), env=env, timeout=1800)
        # 結果取得
        energy_log = np.loadtxt('energy_log.csv', delimiter=',')
        vorticity_log = np.loadtxt('vorticity_log.csv', delimiter=',')
        converged = np.all(np.isfinite(energy_log)) and (np.abs(energy_log[-1] - energy_log[0]) < 0.1 * np.abs(energy_log[0]))
        results.append({
            'theta': theta, 'nu': nu, 'nx': nx, 'seed': seed, 'method': method, 'moyal_order': moyal_order,
            'energy_init': float(energy_log[0]), 'energy_final': float(energy_log[-1]),
            'vorticity_mean': float(np.mean(vorticity_log)),
            'converged': converged
        })
        # バックアップ
        if os.path.exists('energy_log.csv'):
            shutil.copy('energy_log.csv', os.path.join(RESULTS_DIR, f'energy_log_{i}.csv'))
        if os.path.exists('vorticity_log.csv'):
            shutil.copy('vorticity_log.csv', os.path.join(RESULTS_DIR, f'vorticity_log_{i}.csv'))
    except Exception as e:
        results.append({
            'theta': theta, 'nu': nu, 'nx': nx, 'seed': seed, 'method': method, 'moyal_order': moyal_order,
            'energy_init': np.nan, 'energy_final': np.nan, 'vorticity_mean': np.nan, 'converged': False, 'error': str(e)
        })
    # チェックポイント・バックアップローテーション
    backup_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('energy_log_')]
    backup_files = [f for f in backup_files if os.path.exists(os.path.join(RESULTS_DIR, f))]
    backup_files = sorted(backup_files, key=lambda x: os.path.getmtime(os.path.join(RESULTS_DIR, x)))
    if len(backup_files) > 10:
        for f in backup_files[:-10]:
            os.remove(os.path.join(RESULTS_DIR, f))
    time.sleep(1)

# --- 結果保存 ---
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, 'sweep_results.csv'), index=False)

# --- 可視化 ---
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.figure()
df['converged'].value_counts().plot(kind='bar')
plt.title('Convergence Rate (English)')
plt.xlabel('Converged')
plt.ylabel('Count')
plt.savefig(os.path.join(RESULTS_DIR, 'convergence_rate.png'))
plt.close()

print('✅ Sweep B completed. Results saved.') 