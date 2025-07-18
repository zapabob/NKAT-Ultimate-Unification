import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback
import logging

# ログ設定
logging.basicConfig(filename='logs/parameter_sweep_analysis.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

RESULTS_CSV = 'sweep_results.csv'
PARAMETER_SETS = [
    # 例: (theta, nu, grid_size, Q_max, moyal_order, seed)
    (0.1, 0.01, 64, 10, 2, 42),
    (0.2, 0.01, 64, 10, 2, 43),
    (0.1, 0.02, 64, 10, 2, 44),
    # ... 必要に応じて追加
]

# 物理量計算のダミー関数（本来はNavier–Stokes URT★の計算結果を使う）
def compute_physical_quantities(theta, nu, grid_size, Q_max, moyal_order, seed):
    np.random.seed(seed)
    energy = np.abs(np.random.normal(loc=1.0, scale=0.1))
    vorticity = np.abs(np.random.normal(loc=0.5, scale=0.05))
    spectrum_slope = -5/3 + np.random.normal(loc=0, scale=0.05)
    convergence = np.random.choice([True, False], p=[0.9, 0.1])
    return {
        'theta': theta,
        'nu': nu,
        'grid_size': grid_size,
        'Q_max': Q_max,
        'moyal_order': moyal_order,
        'seed': seed,
        'energy': energy,
        'vorticity': vorticity,
        'spectrum_slope': spectrum_slope,
        'convergence': convergence
    }

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
        logging.error(f"{path}の読み込みエラー: {e}")
        return pd.DataFrame()

def main():
    results = []
    for params in tqdm(PARAMETER_SETS, desc='Parameter Sweep'):
        try:
            res = compute_physical_quantities(*params)
            results.append(res)
        except Exception as e:
            logging.error(f"パラメータ{params}でエラー: {e}\n{traceback.format_exc()}")
            continue
    df_new = pd.DataFrame(results)
    df_old = safe_load_csv(RESULTS_CSV)
    if not df_old.empty:
        df = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates()
    else:
        df = df_new
    try:
        df.to_csv(RESULTS_CSV, index=False)
        logging.info(f"{RESULTS_CSV}を保存しました。レコード数: {len(df)}")
    except Exception as e:
        logging.error(f"{RESULTS_CSV}の保存エラー: {e}\n{traceback.format_exc()}")

if __name__ == '__main__':
    main() 