#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Colab用コラッツ予想検証ノートブック
Collatz Conjecture Verification for Google Colab

著者: NKAT研究チーム
日付: 2025年7月20日

Google Colab無料版で実行可能な軽量版コラッツ予想検証システム
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import time
import json
from datetime import datetime
import gc

# Colab環境設定
def setup_colab():
    """Colab環境の設定"""
    try:
        import google.colab
        print("✅ Google Colab環境を検出しました")
        plt.rcParams['figure.figsize'] = (10, 6)
        return True
    except ImportError:
        print("ℹ️ ローカル環境で実行中です")
        return False

class ColabCollatzVerifier:
    """Colab用コラッツ検証システム"""
    
    def __init__(self):
        self.results = []
        self.stats = {}
    
    def collatz_step(self, n):
        """コラッツ演算子（最適化版）"""
        if n % 2 == 0:
            return n // 2
        else:
            return 3 * n + 1
    
    def verify_number(self, n):
        """単一数の検証"""
        sequence = [n]
        current = n
        steps = 0
        max_value = n
        
        while current != 1 and steps < 10000:  # 制限付き
            current = self.collatz_step(current)
            sequence.append(current)
            steps += 1
            max_value = max(max_value, current)
        
        return {
            'n': n,
            'steps': steps,
            'max_value': max_value,
            'converged': current == 1,
            'sequence_length': len(sequence)
        }
    
    def verify_range(self, start, end, batch_size=100):
        """範囲検証"""
        print(f"🔬 検証開始: {start:,} から {end:,} まで")
        
        results = []
        for i in tqdm(range(start, end + 1, batch_size), desc="検証中"):
            batch_end = min(i + batch_size, end + 1)
            for n in range(i, batch_end):
                result = self.verify_number(n)
                results.append(result)
            
            # メモリ管理
            if len(results) % 1000 == 0:
                gc.collect()
        
        self.results = results
        return results
    
    def analyze_results(self):
        """結果分析"""
        df = pd.DataFrame(self.results)
        
        self.stats = {
            'total_tested': len(df),
            'converged': df['converged'].sum(),
            'convergence_rate': df['converged'].mean() * 100,
            'avg_steps': df['steps'].mean(),
            'max_steps': df['steps'].max(),
            'avg_max_value': df['max_value'].mean(),
            'max_value_ever': df['max_value'].max()
        }
        
        return self.stats
    
    def visualize_results(self):
        """結果可視化"""
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Collatz Conjecture Verification Results', fontsize=16)
        
        # ステップ数分布
        axes[0, 0].hist(df['steps'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Steps Distribution')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Count')
        
        # 最大値分布
        axes[0, 1].hist(df['max_value'], bins=30, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Max Value Distribution')
        axes[0, 1].set_xlabel('Max Value')
        axes[0, 1].set_ylabel('Count')
        
        # ステップ数 vs 最大値
        axes[1, 0].scatter(df['steps'], df['max_value'], alpha=0.6, s=20)
        axes[1, 0].set_title('Steps vs Max Value')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Max Value')
        
        # 収束率
        converged = df['converged'].sum()
        total = len(df)
        axes[1, 1].pie([converged, total-converged], 
                       labels=['Converged', 'Not Converged'],
                       colors=['lightblue', 'lightcoral'],
                       autopct='%1.1f%%')
        axes[1, 1].set_title('Convergence Rate')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_summary(self):
        """結果サマリー表示"""
        print("\n" + "="*50)
        print("📊 検証結果サマリー")
        print("="*50)
        print(f"総テスト数: {self.stats['total_tested']:,}")
        print(f"収束数: {self.stats['converged']:,}")
        print(f"収束率: {self.stats['convergence_rate']:.2f}%")
        print(f"平均ステップ数: {self.stats['avg_steps']:.2f}")
        print(f"最大ステップ数: {self.stats['max_steps']}")
        print(f"平均最大値: {self.stats['avg_max_value']:.0f}")
        print(f"軌道内最大値: {self.stats['max_value_ever']:,}")
        print("="*50)

def main():
    """メイン実行関数"""
    print("🚀 Google Colab用コラッツ予想検証システム")
    print("="*50)
    
    # Colab環境設定
    setup_colab()
    
    # 検証システム初期化
    verifier = ColabCollatzVerifier()
    
    # 検証実行（Colab制限内）
    print("\n🔬 検証実行中...")
    results = verifier.verify_range(1, 10000, batch_size=100)
    
    # 結果分析
    stats = verifier.analyze_results()
    
    # 可視化
    verifier.visualize_results()
    
    # サマリー表示
    verifier.print_summary()
    
    print("\n✅ 検証完了！")
    print("**Don't hold back. Give it your all deep think!!**")

if __name__ == "__main__":
    main() 