#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT理論によるBSD予想：楕円曲線の2ビット量子セルネットワーク・L関数・情報エントロピー可視化
Author: NKAT Research Team
Date: 2024-07-14
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sympy
from sympy.abc import x, y
import os
import json
from datetime import datetime

# --- 楕円曲線の定義 ---
def rational_points_on_curve(a, b, search_bound=20):
    """y^2 = x^3 + a x + b の有理点（整数点）を探索"""
    points = []
    for xi in range(-search_bound, search_bound+1):
        rhs = xi**3 + a*xi + b
        if rhs < 0:
            continue
        yi = int(np.sqrt(rhs))
        if yi*yi == rhs:
            points.append((xi, yi))
            if yi != 0:
                points.append((xi, -yi))
    return points

# --- 2ビット量子セル列による点の符号化 ---
def int_to_cell_bits(n, min_len=1):
    bits = list(bin(abs(n))[2:])
    bits = [int(b) for b in bits]
    if len(bits) % 2 == 1:
        bits = [0] + bits
    while len(bits) < min_len:
        bits = [0, 0] + bits
    return [bits[i:i+2] for i in range(0, len(bits), 2)]

def point_to_cell_state(point):
    xi, yi = point
    x_bits = int_to_cell_bits(xi, min_len=4)
    y_bits = int_to_cell_bits(yi, min_len=4)
    return x_bits + y_bits

# --- L関数の近似（有限オイラー積） ---
def approx_L_E(s, a, b, N=100):
    """L(E, s)の有限オイラー積近似（素数p<=N）"""
    from sympy.ntheory import primerange
    L = 1.0
    for p in primerange(2, N+1):
        # 楕円曲線のmod p上の点数
        count = 0
        for xi in range(p):
            rhs = (xi**3 + a*xi + b) % p
            y2s = set([y2 for y2 in range(p) if (y2*y2)%p == rhs])
            count += len(y2s)
        a_p = p + 1 - count
        L *= 1.0 / (1 - a_p * p**(-s) + p**(1-2*s))
    return L

def L_curve_plot(a, b, s_range=(0.5, 2.0), num=100):
    s_vals = np.linspace(s_range[0], s_range[1], num)
    L_vals = []
    for s in tqdm(s_vals, desc='L(E,s)'):  # tqdm進捗バー
        try:
            Ls = approx_L_E(s, a, b, N=50)
            L_vals.append(Ls)
        except Exception:
            L_vals.append(np.nan)
    return s_vals, np.array(L_vals)

# --- 情報エントロピー ---
def info_entropy_cell_states(cell_states):
    """セル状態列の情報エントロピー（bit単位）"""
    flat = sum(cell_states, [])
    vals, counts = np.unique(flat, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

# --- 可視化 ---
def plot_elliptic_curve_points(points, a, b, outdir):
    plt.figure(figsize=(8,6))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.scatter(xs, ys, c='blue', label='Rational Points')
    plt.title(f'Elliptic Curve: y^2 = x^3 + {a}x + {b}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'elliptic_curve_points.png'))
    plt.close()

def plot_L_curve(s_vals, L_vals, outdir):
    plt.figure(figsize=(8,6))
    plt.plot(s_vals, L_vals, label='L(E,s) (approx)')
    plt.xlabel('s')
    plt.ylabel('L(E,s)')
    plt.title('Approximate L-function of Elliptic Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'elliptic_curve_L_function.png'))
    plt.close()

def plot_entropy(entropies, outdir):
    plt.figure(figsize=(8,6))
    plt.plot(entropies, marker='o')
    plt.xlabel('Point Index')
    plt.ylabel('Information Entropy (bits)')
    plt.title('Information Entropy of Cell States (Elliptic Curve Points)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'elliptic_curve_entropy.png'))
    plt.close()

# --- メイン実行 ---
def main():
    print('NKAT理論によるBSD予想：楕円曲線・L関数・情報エントロピー可視化')
    outdir = 'Results/visualizations/elliptic_curve_bsd_nkat'
    os.makedirs(outdir, exist_ok=True)
    a, b = -1, 0  # y^2 = x^3 - x
    points = rational_points_on_curve(a, b, search_bound=20)
    print(f'Found {len(points)} integer points')
    plot_elliptic_curve_points(points, a, b, outdir)
    # セル状態列
    cell_states = [point_to_cell_state(p) for p in points]
    entropies = [info_entropy_cell_states([cs]) for cs in cell_states]
    plot_entropy(entropies, outdir)
    # L関数
    s_vals, L_vals = L_curve_plot(a, b, s_range=(0.5, 2.0), num=40)
    plot_L_curve(s_vals, L_vals, outdir)
    # データ保存
    data = {
        'a': a, 'b': b,
        'points': points,
        'entropies': entropies,
        's_vals': s_vals.tolist(),
        'L_vals': L_vals.tolist(),
        'datetime': datetime.now().isoformat()
    }
    with open(os.path.join(outdir, 'elliptic_curve_bsd_nkat_data.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f'可視化・データ保存完了: {outdir}')

if __name__ == '__main__':
    main() 