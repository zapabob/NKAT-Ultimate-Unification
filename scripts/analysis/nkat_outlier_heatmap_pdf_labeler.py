import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import traceback
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde

# ログ設定
logging.basicConfig(filename='logs/outlier_heatmap_pdf_labeler.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

RESULTS_DIR = 'Results/sweep_b'

# matplotlib英語表記
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 安全なCSV読込
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

# データ読込
df_sweep = safe_load_csv(os.path.join(RESULTS_DIR, 'sweep_results.csv'))
df_dim = safe_load_csv(os.path.join(RESULTS_DIR, 'fractal_dim_results.csv'))

# 外れ値判定（tau_diff, energy, vorticity, spectrum_slope）
def detect_outliers(series, threshold=3.0):
    if series.isnull().all():
        return pd.Series([False]*len(series))
    z = (series - series.mean()) / (series.std() + 1e-8)
    return np.abs(z) > threshold

# 残差ヒートマップ・PDF・クラスタラベル付与
def main():
    if df_sweep.empty or df_dim.empty:
        logging.warning('sweep_results.csvまたはfractal_dim_results.csvが空です。E1出力はスキップされます。')
        return
    # マージ
    df = df_sweep.copy()
    if 'idx' in df_dim.columns:
        df = df.merge(df_dim[['idx', 'tau_diff', 'fractal_dim']], left_index=True, right_on='idx', how='left')
    else:
        df['tau_diff'] = np.nan
        df['fractal_dim'] = np.nan
    # 外れ値判定
    df['outlier_tau'] = detect_outliers(df['tau_diff'])
    df['outlier_energy'] = detect_outliers(df['energy'])
    df['outlier_vorticity'] = detect_outliers(df['vorticity'])
    df['outlier_slope'] = detect_outliers(df['spectrum_slope'])
    df['any_outlier'] = df[['outlier_tau','outlier_energy','outlier_vorticity','outlier_slope']].any(axis=1)
    # クラスタリング（tau_diff, energy, vorticity, spectrum_slope）
    X = df[['tau_diff','energy','vorticity','spectrum_slope']].fillna(0).values
    try:
        clustering = DBSCAN(eps=1.5, min_samples=3).fit(X)
        df['cluster'] = clustering.labels_
    except Exception as e:
        logging.error(f"クラスタリングエラー: {e}\n{traceback.format_exc()}")
        df['cluster'] = -1
    # 残差ヒートマップ
    try:
        plt.figure(figsize=(8,6))
        plt.scatter(df['energy'], df['tau_diff'], c=df['any_outlier'], cmap='coolwarm', alpha=0.7, label='Outlier')
        plt.xlabel('Energy')
        plt.ylabel('tau_diff')
        plt.title('Residual Heatmap (English)')
        plt.colorbar(label='Outlier')
        plt.savefig(os.path.join(RESULTS_DIR, 'outlier_residual_heatmap.png'))
        plt.close()
    except Exception as e:
        logging.error(f"ヒートマップ描画エラー: {e}\n{traceback.format_exc()}")
    # 渦度PDF
    try:
        vorticity = df['vorticity'].dropna()
        if len(vorticity) > 10:
            kde = gaussian_kde(vorticity)
            x_grid = np.linspace(vorticity.min(), vorticity.max(), 100)
            plt.figure()
            plt.plot(x_grid, kde(x_grid), label='Vorticity PDF')
            plt.title('Vorticity PDF (English)')
            plt.xlabel('Vorticity')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(os.path.join(RESULTS_DIR, 'vorticity_pdf.png'))
            plt.close()
    except Exception as e:
        logging.error(f"渦度PDF描画エラー: {e}\n{traceback.format_exc()}")
    # クラスタラベル付きCSV出力
    try:
        df.to_csv(os.path.join(RESULTS_DIR, 'outlier_labeled_results.csv'), index=False)
    except Exception as e:
        logging.error(f"outlier_labeled_results.csv保存エラー: {e}\n{traceback.format_exc()}")
    # TeX出力（外れ値テーブル）
    try:
        outlier_df = df[df['any_outlier']]
        tex_path = os.path.join(RESULTS_DIR, 'outlier_table.tex')
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(outlier_df.to_latex(index=False, float_format='%.4g'))
    except Exception as e:
        logging.error(f"TeX出力エラー: {e}\n{traceback.format_exc()}")
    print('E1: Outlier heatmap, vorticity PDF, cluster labeling, TeX/CSV/PNG output completed.')

if __name__ == '__main__':
    main() 