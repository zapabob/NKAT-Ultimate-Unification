import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import yaml
import logging
import traceback

# ログ設定
logging.basicConfig(filename='logs/theory_bridge_analysis.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- データパス設定 ---
RESULTS_DIR = 'Results/sweep_b'
THEORY_DIR = 'docs/theory/'

# --- matplotlibの日本語文字化け防止 ---
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# --- 理論式（例: 多重フラクタル次元, リーマン零点スペクトル, 非可換KA表現） ---
def theoretical_fractal_tau(q, alpha_k, lambda_k, lambda_max):
    # τ(q) = Σ α_k (λ_k/λ_max)^q
    return np.sum([a * (l/lambda_max)**q for a, l in zip(alpha_k, lambda_k)])

def riemann_zero_spectrum(n=10):
    # 最初のn個のリーマン零点虚部
    return np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178, 40.918719, 43.327073, 48.005150, 49.773832][:n])

def noncommutative_ka_representation(x, theta, kappa):
    # 簡易: f(x) = Σ exp(iλx) + θ補正
    lam = riemann_zero_spectrum()
    return np.sum(np.exp(1j*lam*x)) + theta*np.sum(np.sin(lam*x + kappa))

# --- Sweep結果読込（エラーガード付き） ---
def safe_load_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        logging.warning(f"{path}が存在しないか空です。新規作成します。")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except pd.errors.EmptyDataError:
        logging.error(f"{path}が空データです。新規作成します。")
        return pd.DataFrame()
    except KeyError as e:
        logging.error(f"{path}のKeyError: {e}。新規作成します。")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"{path}の読み込みエラー: {e}\n{traceback.format_exc()}")
        return pd.DataFrame()

sweep_path = os.path.join(RESULTS_DIR, 'sweep_results.csv')
df = safe_load_csv(sweep_path)

# --- スペクトルデータの自動比較 ---
fractal_qs = np.linspace(0.5, 3, 20)
alpha_k = np.ones(5) / 5
lambda_k = riemann_zero_spectrum(5)
lambda_max = np.max(lambda_k)
tau_theory = [theoretical_fractal_tau(q, alpha_k, lambda_k, lambda_max) for q in fractal_qs]

# --- 各試行の数値スペクトル・自己相似指数・フラクタル次元推定 ---
fractal_dim_results = []
if not df.empty:
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Theory-Num Bridge'):
        try:
            # URTモードスペクトル読込
            urt_path = os.path.join(RESULTS_DIR, f'energy_log_{i}.csv')
            if not os.path.exists(urt_path):
                fractal_dim_results.append({'idx': i, 'fractal_dim': np.nan, 'tau_diff': np.nan, 'error': 'energy_log_missing'})
                continue
            energy_log = np.loadtxt(urt_path, delimiter=',')
            # 自己相似指数推定（log-log回帰）
            x = np.arange(1, len(energy_log)+1)
            y = np.abs(energy_log)
            mask = (y > 0)
            if np.sum(mask) < 5:
                fractal_dim_results.append({'idx': i, 'fractal_dim': np.nan, 'tau_diff': np.nan, 'error': 'insufficient_data'})
                continue
            coeffs = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
            fractal_dim = -coeffs[0]
            # 理論値との比較
            tau_num = fractal_dim
            tau_diff = np.abs(tau_num - np.mean(tau_theory))
            fractal_dim_results.append({'idx': i, 'fractal_dim': fractal_dim, 'tau_diff': tau_diff})
        except Exception as e:
            logging.error(f"fractal_dim計算エラー: {e}\n{traceback.format_exc()}")
            fractal_dim_results.append({'idx': i, 'fractal_dim': np.nan, 'tau_diff': np.nan, 'error': str(e)})
else:
    logging.warning('sweep_results.csvが空です。fractal_dim_results.csvは空で出力されます。')

# 有効データ数チェック
valid_count = sum(np.isfinite(r.get('fractal_dim', np.nan)) for r in fractal_dim_results)

# --- 結果保存 ---
df_dim = pd.DataFrame(fractal_dim_results)
df_dim.to_csv(os.path.join(RESULTS_DIR, 'fractal_dim_results.csv'), index=False)
logging.info(f'fractal_dim_results.csv written. Valid entries: {valid_count}')
print(f'fractal_dim_results.csv written. Valid entries: {valid_count}')

# --- 可視化 ---
plt.figure()
plt.plot(fractal_qs, tau_theory, label='Theory tau(q)')
if ('fractal_dim' in df_dim.columns) and (df_dim['fractal_dim'].notna().sum() > 0):
    plt.scatter([1]*len(df_dim), df_dim['fractal_dim'], label='Numerical Fractal Dim', alpha=0.5)
plt.title('Fractal Dimension: Theory vs Numerical (English)')
plt.xlabel('q or index')
plt.ylabel('tau(q) / Fractal Dim')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'fractal_dim_comparison.png'))
plt.close()

plt.figure()
if ('tau_diff' in df_dim.columns) and (df_dim['tau_diff'].notna().sum() > 0):
    plt.hist(df_dim['tau_diff'].dropna(), bins=20)
plt.title('Difference: Numerical vs Theory tau(q) (English)')
plt.xlabel('Difference')
plt.ylabel('Count')
plt.savefig(os.path.join(RESULTS_DIR, 'fractal_dim_difference.png'))
plt.close()

print('✅ Theory-Num Bridge (C) completed. Results saved.') 