import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from sklearn.cluster import DBSCAN
import sys

RESULTS_DIR = 'Results/sweep_b'

# --- データ読込 ---
df = pd.read_csv(os.path.join(RESULTS_DIR, 'sweep_results.csv'))

csv_path = os.path.join(RESULTS_DIR, 'fractal_dim_results.csv')

if not os.path.exists(csv_path):
    print(f"[警告] ファイルが存在しません: {csv_path}")
    df_dim = None
elif os.path.getsize(csv_path) == 0:
    print(f"[警告] ファイルが空です: {csv_path}")
    df_dim = None
else:
    try:
        df_dim = pd.read_csv(csv_path)
        if df_dim.empty:
            print(f"[警告] CSVはヘッダーのみ、またはデータがありません: {csv_path}")
            df_dim = None
    except pd.errors.EmptyDataError:
        print(f"[警告] pandasでEmptyDataError: {csv_path}")
        df_dim = None
    except Exception as e:
        print(f"[エラー] CSV読み込み失敗: {csv_path}\n{e}")
        df_dim = None

# 以降、df_dimがNoneなら安全にスキップ・分岐
if df_dim is None:
    # 必要に応じてreturnやcontinue、またはダミー処理
    print('Error: fractal_dim_results.csv is empty. Please check upstream data generation.')
    sys.exit(1)

# --- パラメータ軸抽出 ---
theta_list = sorted(df['theta'].unique())
nu_list = sorted(df['nu'].unique())

# --- 残差行列生成 ---
residual_matrix = np.full((len(nu_list), len(theta_list)), np.nan)
for i, nu in enumerate(nu_list):
    for j, theta in enumerate(theta_list):
        idxs = df[(df['theta']==theta)&(df['nu']==nu)].index
        if len(idxs)==0: continue
        # 対応するfractal_dimのtau_diff平均
        tau_diffs = df_dim[df_dim['idx'].isin(idxs)]['tau_diff'].dropna()
        if len(tau_diffs)>0:
            residual_matrix[i,j] = tau_diffs.mean()

# --- 残差ヒートマップ ---
sns.set(style='white')
plt.figure(figsize=(8,6))
sns.heatmap(residual_matrix, cmap='coolwarm', annot=True, fmt='.2e', xticklabels=[f'{t:.1e}' for t in theta_list], yticklabels=[f'{n:.1e}' for n in nu_list])
plt.title('Residual Heatmap |τ_num – τ_theory| (English)')
plt.xlabel('θ index')
plt.ylabel('ν index')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'residual_heatmap.png'))
plt.close()

# --- 外れ値ケース抽出（上位10%） ---
thresh = np.nanpercentile(df_dim['tau_diff'], 90)
outlier_cases = df_dim[df_dim['tau_diff']>=thresh]['idx'].values

# --- 渦度PDF比較 ---
plt.figure(figsize=(8,6))
for idx in tqdm(outlier_cases, desc='Outlier Vorticity PDF'):
    vort_path = os.path.join(RESULTS_DIR, f'vorticity_log_{idx}.csv')
    if not os.path.exists(vort_path): continue
    vort = np.loadtxt(vort_path, delimiter=',')
    hist, bins = np.histogram(np.abs(vort), bins=100, density=True)
    plt.semilogy(bins[:-1], hist, alpha=0.5, label=f'case {idx}')
plt.title('Vorticity PDF (Outliers, log scale, English)')
plt.xlabel('|ω|')
plt.ylabel('PDF')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'outlier_vorticity_pdf.png'))
plt.close()

# --- 異常クラスタ自動ラベル（DBSCAN） ---
X = df_dim[['fractal_dim','tau_diff']].dropna().values
if len(X)>0:
    clustering = DBSCAN(eps=0.2, min_samples=2).fit(X)
    df_dim['cluster'] = -1
    df_dim.loc[df_dim[['fractal_dim','tau_diff']].dropna().index, 'cluster'] = clustering.labels_
    # クラスタごとに色分け散布図
    plt.figure(figsize=(8,6))
    for c in np.unique(clustering.labels_):
        mask = (df_dim['cluster']==c)
        plt.scatter(df_dim[mask]['fractal_dim'], df_dim[mask]['tau_diff'], label=f'Cluster {c}')
    plt.title('Outlier Clusters (DBSCAN, English)')
    plt.xlabel('Fractal Dim')
    plt.ylabel('τ_diff')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'outlier_clusters.png'))
    plt.close()

# --- CSV/TeX出力 ---
df_dim.to_csv(os.path.join(RESULTS_DIR, 'outlier_analysis_labeled.csv'), index=False)
with open(os.path.join(RESULTS_DIR, 'outlier_clusters_table.tex'), 'w', encoding='utf-8') as f:
    f.write(df_dim[['idx','fractal_dim','tau_diff','cluster']].to_latex(index=False, float_format='%.3e'))

print('✅ E1: Outlier analysis completed. PNG/CSV/TeX saved.') 