#!/usr/bin/env python3
"""
🔍 NKAT NaN診断レポート生成システム
数値安定性の詳細分析とデバッグ情報を提供
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

def analyze_numerical_stability():
    """数値安定性分析"""
    
    print("🔍 NKAT NaN診断レポート生成中...")
    print("=" * 60)
    
    # テスト用データ生成
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 基本的な数値範囲テスト
    print("📊 1. 数値範囲テスト")
    test_ranges = [
        ("Normal", torch.randn(100, 4, device=device)),
        ("Large", torch.randn(100, 4, device=device) * 1e6),
        ("Small", torch.randn(100, 4, device=device) * 1e-6),
        ("Mixed", torch.cat([
            torch.randn(50, 4, device=device) * 1e6,
            torch.randn(50, 4, device=device) * 1e-6
        ]))
    ]
    
    stability_results = {}
    
    for name, data in test_ranges:
        # 基本統計
        mean_val = torch.mean(data).item()
        std_val = torch.std(data).item()
        min_val = torch.min(data).item()
        max_val = torch.max(data).item()
        
        # NaN/Inf検出
        nan_count = torch.isnan(data).sum().item()
        inf_count = torch.isinf(data).sum().item()
        
        # 対数計算テスト
        log_safe = torch.log(torch.clamp(torch.abs(data), min=1e-12))
        log_nan_count = torch.isnan(log_safe).sum().item()
        
        stability_results[name] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'log_nan_count': log_nan_count,
            'range_ratio': max_val / (abs(min_val) + 1e-12)
        }
        
        print(f"   {name:8}: mean={mean_val:.2e}, std={std_val:.2e}")
        print(f"            range=[{min_val:.2e}, {max_val:.2e}]")
        print(f"            NaN={nan_count}, Inf={inf_count}, LogNaN={log_nan_count}")
    
    # 2. 物理損失関数テスト
    print("\n📊 2. 物理損失関数安定性テスト")
    
    # スペクトラル次元計算テスト
    test_fields = [
        torch.randn(32, 4, 4, device=device),  # Normal
        torch.randn(32, 4, 4, device=device) * 1e-3,  # Small
        torch.randn(32, 4, 4, device=device) * 1e3,   # Large
    ]
    
    spectral_results = {}
    
    for i, field in enumerate(test_fields):
        field_name = ["Normal", "Small", "Large"][i]
        
        # スペクトラル次元計算
        field_norm = torch.norm(field, dim=-1) + 1e-12
        log_field = torch.log(field_norm)
        spectral_dim = 4.0 + 0.1 * torch.mean(log_field)
        
        # 結果分析
        spectral_results[field_name] = {
            'field_norm_range': [torch.min(field_norm).item(), torch.max(field_norm).item()],
            'log_field_range': [torch.min(log_field).item(), torch.max(log_field).item()],
            'spectral_dim': spectral_dim.item(),
            'nan_in_norm': torch.isnan(field_norm).sum().item(),
            'nan_in_log': torch.isnan(log_field).sum().item(),
            'nan_in_spectral': torch.isnan(spectral_dim).item()
        }
        
        print(f"   {field_name:8}: spectral_dim={spectral_dim.item():.6f}")
        print(f"            norm_range=[{torch.min(field_norm).item():.2e}, {torch.max(field_norm).item():.2e}]")
        print(f"            NaN: norm={torch.isnan(field_norm).sum().item()}, log={torch.isnan(log_field).sum().item()}")
    
    # 3. θ-parameter running テスト
    print("\n📊 3. θ-parameter running安定性テスト")
    
    theta_test_cases = [
        ("Normal", torch.tensor([1e-70, 1e-60, 1e-50], device=device)),
        ("Extreme", torch.tensor([1e-80, 1e-100, 1e-120], device=device)),
        ("Mixed", torch.tensor([1e-40, 1e-80, 1e-10], device=device))
    ]
    
    energy_scales = torch.logspace(10, 18, 3, device=device)
    
    theta_results = {}
    
    for name, theta_values in theta_test_cases:
        # θ-running計算
        log_energy = torch.log10(torch.clamp(energy_scales, min=1e-10))
        running_target = -0.1 * log_energy
        theta_log = torch.log10(torch.clamp(theta_values, min=1e-80))
        
        # MSE計算
        mse_loss = torch.nn.functional.mse_loss(theta_log, running_target)
        
        theta_results[name] = {
            'theta_range': [torch.min(theta_values).item(), torch.max(theta_values).item()],
            'theta_log_range': [torch.min(theta_log).item(), torch.max(theta_log).item()],
            'running_target_range': [torch.min(running_target).item(), torch.max(running_target).item()],
            'mse_loss': mse_loss.item(),
            'nan_in_theta_log': torch.isnan(theta_log).sum().item(),
            'nan_in_mse': torch.isnan(mse_loss).item()
        }
        
        print(f"   {name:8}: mse_loss={mse_loss.item():.6f}")
        print(f"            theta_log_range=[{torch.min(theta_log).item():.2f}, {torch.max(theta_log).item():.2f}]")
        print(f"            NaN: theta_log={torch.isnan(theta_log).sum().item()}, mse={torch.isnan(mse_loss).item()}")
    
    # 4. 推奨修正案
    print("\n💡 4. 推奨修正案")
    
    recommendations = []
    
    # 数値範囲チェック
    for name, result in stability_results.items():
        if result['range_ratio'] > 1e12:
            recommendations.append(f"⚠️ {name}データの範囲が広すぎます (ratio={result['range_ratio']:.2e})")
        if result['log_nan_count'] > 0:
            recommendations.append(f"⚠️ {name}データで対数計算時にNaN発生")
    
    # スペクトラル次元チェック
    for name, result in spectral_results.items():
        if result['nan_in_spectral']:
            recommendations.append(f"⚠️ {name}フィールドでスペクトラル次元計算時にNaN発生")
        if abs(result['spectral_dim'] - 4.0) > 10:
            recommendations.append(f"⚠️ {name}フィールドでスペクトラル次元が異常値 ({result['spectral_dim']:.2f})")
    
    # θ-parameterチェック
    for name, result in theta_results.items():
        if result['nan_in_mse']:
            recommendations.append(f"⚠️ {name}θ値でMSE計算時にNaN発生")
    
    if not recommendations:
        recommendations.append("✅ 数値安定性に大きな問題は検出されませんでした")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    # 5. 修正コード提案
    print("\n🔧 5. 修正コード提案")
    
    safe_code = '''
# NaN安全版数値計算関数
def safe_log(x, min_val=1e-12):
    """安全な対数計算"""
    x_safe = torch.clamp(x, min=min_val)
    result = torch.log(x_safe)
    return torch.where(torch.isnan(result), torch.tensor(0.0, device=x.device), result)

def safe_spectral_dimension(dirac_field, eps=1e-12):
    """NaN安全スペクトラル次元計算"""
    field_norm = torch.norm(dirac_field, dim=-1) + eps
    log_field = safe_log(field_norm)
    spectral_dim = 4.0 + 0.1 * torch.mean(log_field)
    return torch.clamp(spectral_dim, 0.1, 10.0)  # 物理的範囲に制限

def safe_theta_running(theta_values, energy_scale, eps=1e-15):
    """NaN安全θ-running計算"""
    log_energy = safe_log(torch.clamp(energy_scale, min=eps))
    running_target = -0.1 * log_energy
    theta_log = safe_log(torch.clamp(theta_values, min=1e-80))
    return torch.nn.functional.mse_loss(theta_log, running_target)
'''
    
    print(safe_code)
    
    # 6. レポート保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_data = {
        'timestamp': timestamp,
        'stability_results': stability_results,
        'spectral_results': spectral_results,
        'theta_results': theta_results,
        'recommendations': recommendations,
        'device': str(device),
        'torch_version': torch.__version__
    }
    
    report_file = f"nkat_diagnostic_report_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 診断レポート保存: {report_file}")
    
    # 7. 可視化
    create_diagnostic_plots(stability_results, spectral_results, theta_results, timestamp)
    
    return report_data

def create_diagnostic_plots(stability_results, spectral_results, theta_results, timestamp):
    """診断結果の可視化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔍 NKAT数値安定性診断レポート', fontsize=16, fontweight='bold')
    
    # 1. 数値範囲分析
    names = list(stability_results.keys())
    ranges = [stability_results[name]['range_ratio'] for name in names]
    
    axes[0, 0].bar(names, ranges)
    axes[0, 0].set_title('📊 数値範囲比率')
    axes[0, 0].set_ylabel('Max/Min比率 (log scale)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. NaN発生統計
    nan_counts = [stability_results[name]['nan_count'] + stability_results[name]['inf_count'] 
                  for name in names]
    
    axes[0, 1].bar(names, nan_counts, color='red', alpha=0.7)
    axes[0, 1].set_title('⚠️ NaN/Inf発生数')
    axes[0, 1].set_ylabel('異常値数')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. スペクトラル次元分析
    spectral_names = list(spectral_results.keys())
    spectral_dims = [spectral_results[name]['spectral_dim'] for name in spectral_names]
    
    axes[1, 0].bar(spectral_names, spectral_dims, color='blue', alpha=0.7)
    axes[1, 0].axhline(y=4.0, color='green', linestyle='--', label='理論値 (4.0)')
    axes[1, 0].set_title('🎯 スペクトラル次元計算')
    axes[1, 0].set_ylabel('スペクトラル次元')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. θ-parameter MSE分析
    theta_names = list(theta_results.keys())
    mse_values = [theta_results[name]['mse_loss'] for name in theta_names]
    
    axes[1, 1].bar(theta_names, mse_values, color='orange', alpha=0.7)
    axes[1, 1].set_title('🔄 θ-parameter MSE損失')
    axes[1, 1].set_ylabel('MSE損失')
    axes[1, 1].set_yscale('log')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    plot_file = f"nkat_diagnostic_plots_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 診断グラフ保存: {plot_file}")

if __name__ == "__main__":
    print("🔍 NKAT NaN診断システム開始")
    report = analyze_numerical_stability()
    print("\n✅ 診断完了！") 