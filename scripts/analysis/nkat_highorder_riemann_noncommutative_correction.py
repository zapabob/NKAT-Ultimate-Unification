import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import traceback

# ログ設定
logging.basicConfig(filename='logs/highorder_riemann_noncommutative_correction.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

RESULTS_DIR = 'Results/sweep_b'

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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

def riemann_zero_spectrum(n=20):
    # 最初のn個のリーマン零点虚部
    return np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178, 40.918719, 43.327073, 48.005150, 49.773832,
                     52.970321, 56.446247, 59.347044, 60.83178, 65.112544, 67.079811, 69.546401, 72.067158, 75.704690, 77.144840][:n])

def moyal_star_correction(x, m=2, theta=0.1):
    # 高次Moyal積補正（m=2,4,6...）
    lam = riemann_zero_spectrum(m)
    corr = 0
    for l in lam:
        corr += np.sin(l*x) * (theta**(m//2)) / np.math.factorial(m)
    return corr

def theoretical_fractal_tau_highorder(q, alpha_k, lambda_k, lambda_max, moyal_order=2, theta=0.1):
    # 高次Moyal補正付きτ(q)
    base = np.sum([a * (l/lambda_max)**q for a, l in zip(alpha_k, lambda_k)])
    corr = moyal_star_correction(q, m=moyal_order, theta=theta)
    return base + corr

def main():
    df_sweep = safe_load_csv(os.path.join(RESULTS_DIR, 'sweep_results.csv'))
    df_dim = safe_load_csv(os.path.join(RESULTS_DIR, 'fractal_dim_results.csv'))
    if df_sweep.empty or df_dim.empty:
        logging.warning('sweep_results.csvまたはfractal_dim_results.csvが空です。E2出力はスキップされます。')
        return
    # パラメータ例
    fractal_qs = np.linspace(0.5, 3, 20)
    alpha_k = np.ones(10) / 10
    lambda_k = riemann_zero_spectrum(10)
    lambda_max = np.max(lambda_k)
    # 高次Moyal積補正付き理論値
    tau_theory_m4 = [theoretical_fractal_tau_highorder(q, alpha_k, lambda_k, lambda_max, moyal_order=4, theta=0.1) for q in fractal_qs]
    tau_theory_m6 = [theoretical_fractal_tau_highorder(q, alpha_k, lambda_k, lambda_max, moyal_order=6, theta=0.1) for q in fractal_qs]
    # 数値値
    tau_num = df_dim['fractal_dim'].values if 'fractal_dim' in df_dim.columns else np.zeros(len(fractal_qs))
    # 差分
    diff_m4 = np.abs(tau_num[:len(tau_theory_m4)] - tau_theory_m4)
    diff_m6 = np.abs(tau_num[:len(tau_theory_m6)] - tau_theory_m6)
    # 結果保存
    df_out = pd.DataFrame({
        'q': fractal_qs[:len(tau_num)],
        'tau_num': tau_num[:len(fractal_qs)],
        'tau_theory_m4': tau_theory_m4,
        'tau_theory_m6': tau_theory_m6,
        'diff_m4': diff_m4,
        'diff_m6': diff_m6
    })
    try:
        df_out.to_csv(os.path.join(RESULTS_DIR, 'highorder_riemann_noncommutative_comparison.csv'), index=False)
    except Exception as e:
        logging.error(f"CSV保存エラー: {e}\n{traceback.format_exc()}")
    # 可視化
    try:
        plt.figure()
        plt.plot(fractal_qs, tau_theory_m4, label='Theory tau(q) m=4')
        plt.plot(fractal_qs, tau_theory_m6, label='Theory tau(q) m=6')
        plt.scatter(fractal_qs[:len(tau_num)], tau_num[:len(fractal_qs)], label='Numerical Fractal Dim', alpha=0.6)
        plt.title('High-order Riemann/Noncommutative Correction (English)')
        plt.xlabel('q')
        plt.ylabel('tau(q) / Fractal Dim')
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, 'highorder_riemann_noncommutative_comparison.png'))
        plt.close()
    except Exception as e:
        logging.error(f"可視化エラー: {e}\n{traceback.format_exc()}")
    # TeX出力
    try:
        tex_path = os.path.join(RESULTS_DIR, 'highorder_riemann_noncommutative_comparison.tex')
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(df_out.to_latex(index=False, float_format='%.4g'))
    except Exception as e:
        logging.error(f"TeX出力エラー: {e}\n{traceback.format_exc()}")
    print('E2: High-order Riemann/noncommutative correction, comparison, PNG/CSV/TeX output completed.')

if __name__ == '__main__':
    main() 