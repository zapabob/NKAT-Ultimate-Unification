#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT理論によるコラッツ写像の非可換セルネットワーク実装・可視化
Author: NKAT Research Team
Date: 2024-07-14
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

# --- 2ビット量子セル列による整数表現 ---
def int_to_cell_bits(n, min_len=1):
    """整数nを2ビットセル列（リスト）に変換"""
    bits = list(bin(n)[2:])
    bits = [int(b) for b in bits]
    if len(bits) % 2 == 1:
        bits = [0] + bits  # 2ビット単位に揃える
    while len(bits) < min_len:
        bits = [0, 0] + bits
    # 2ビットごとに分割
    return [bits[i:i+2] for i in range(0, len(bits), 2)]

def cell_bits_to_int(cell_bits):
    """2ビットセル列から整数に戻す"""
    bits = sum(cell_bits, [])
    return int(''.join(str(b) for b in bits), 2)

# --- 非可換演算子によるコラッツ操作 ---
def collatz_step_nkat(n):
    """コラッツ写像の1ステップ（非可換セル演算風）"""
    if n % 2 == 0:
        return n // 2, 'even'
    else:
        return 3 * n + 1, 'odd'

def collatz_orbit_nkat(n0, max_steps=1000):
    """n0から始めてコラッツ軌道を返す（状態・エントロピー・ビット長も記録）"""
    n = n0
    orbit = [n]
    entropies = [np.log2(n+1)]
    bit_lengths = [len(bin(n))-2]
    steps = 0
    while n != 1 and steps < max_steps:
        n, parity = collatz_step_nkat(n)
        orbit.append(n)
        entropies.append(np.log2(n+1))
        bit_lengths.append(len(bin(n))-2)
        steps += 1
    return orbit, entropies, bit_lengths

# --- 可視化 ---
def plot_collatz_orbits(orbits, entropies, bit_lengths, n_list, outdir):
    plt.figure(figsize=(10, 6))
    for i, orbit in enumerate(orbits):
        plt.plot(orbit, label=f'n={n_list[i]}')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('Collatz Orbits (NKAT Theory, 2-bit Cell Representation)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'collatz_orbits.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, entropy in enumerate(entropies):
        plt.plot(entropy, label=f'n={n_list[i]}')
    plt.xlabel('Step')
    plt.ylabel('log2(n+1)')
    plt.title('Information Entropy along Collatz Orbits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'collatz_entropy.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, bits in enumerate(bit_lengths):
        plt.plot(bits, label=f'n={n_list[i]}')
    plt.xlabel('Step')
    plt.ylabel('Bit Length')
    plt.title('Bit Length along Collatz Orbits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'collatz_bitlength.png'))
    plt.close()

# --- メイン実行 ---
def main():
    print('NKAT理論によるコラッツ写像・2ビット量子セルネットワーク実装')
    outdir = 'Results/visualizations/collatz_nkat'
    os.makedirs(outdir, exist_ok=True)
    n_list = [3, 7, 11, 19, 27, 97, 871, 6171]  # 代表的な初期値
    orbits = []
    entropies = []
    bit_lengths = []
    for n0 in tqdm(n_list, desc='Collatz Orbits'):  # tqdm進捗バー
        orbit, entropy, bits = collatz_orbit_nkat(n0)
        orbits.append(orbit)
        entropies.append(entropy)
        bit_lengths.append(bits)
    plot_collatz_orbits(orbits, entropies, bit_lengths, n_list, outdir)
    # データ保存
    data = {
        'n_list': n_list,
        'orbits': orbits,
        'entropies': entropies,
        'bit_lengths': bit_lengths,
        'datetime': datetime.now().isoformat()
    }
    with open(os.path.join(outdir, 'collatz_nkat_data.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f'可視化・データ保存完了: {outdir}')

if __name__ == '__main__':
    main() 