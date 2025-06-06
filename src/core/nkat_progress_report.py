#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥💎‼ NKAT計算進行状況レポート ‼💎🔥
現在実行中の計算の詳細ステータス表示
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

def generate_progress_report():
    """進行状況レポートの生成"""
    print("🔥💎 NKAT計算進行状況レポート 💎🔥")
    print("="*80)
    print(f"レポート生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. θ=1e-12 メイン計算の状況
    print("\n📊 1. メイン計算 (θ=1e-12) 状況:")
    print("-" * 50)
    
    main_recovery = Path("nkat_recovery_theta_1e12")
    if main_recovery.exists():
        metadata_file = main_recovery / "nkat_session_metadata.json"
        checkpoint_file = main_recovery / "nkat_checkpoint.pkl"
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            status = metadata.get('status', 'unknown')
            start_time = metadata.get('start_time', 'N/A')
            last_update = metadata.get('last_update', 'N/A')
            computation_state = metadata.get('computation_state', 'N/A')
            
            if start_time != 'N/A':
                start_time = datetime.fromisoformat(start_time).strftime('%H:%M:%S')
            if last_update != 'N/A':
                last_update = datetime.fromisoformat(last_update).strftime('%H:%M:%S')
            
            print(f"   🟢 状態: {status}")
            print(f"   🕐 開始時刻: {start_time}")
            print(f"   🔄 最終更新: {last_update}")
            print(f"   ⚙️ 計算段階: {computation_state}")
            
            if checkpoint_file.exists():
                size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                print(f"   💾 チェックポイント: {size_mb:.2f}MB (更新: {mod_time.strftime('%H:%M:%S')})")
                
                # 進行度推定
                if 'critical_zeros_computation' in computation_state:
                    progress = "🟡 零点探索中 (段階1/9)"
                elif 'off_critical' in computation_state:
                    progress = "🟠 臨界線外検証中 (段階2/9)"
                elif 'functional_equation' in computation_state:
                    progress = "🔵 関数方程式検証中 (段階3/9)"
                elif 'completed' in computation_state:
                    progress = "🟢 完了!"
                else:
                    progress = "🟡 計算中"
                
                print(f"   📈 進行状況: {progress}")
        else:
            print("   ⚠️ メタデータファイルなし")
    else:
        print("   📭 メイン計算ディレクトリなし")
    
    # 2. θ最適化実験の状況
    print("\n📊 2. θ最適化実験状況:")
    print("-" * 50)
    
    theta_recovery_dirs = [
        "nkat_recovery_theta_1e-08",
        "nkat_recovery_theta_1e-10", 
        "nkat_recovery_theta_1e-14",
        "nkat_recovery_theta_1e-16"
    ]
    
    for dir_name in theta_recovery_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            theta_value = dir_name.split('_')[-1]
            print(f"   🧪 θ={theta_value}:")
            
            checkpoint = dir_path / "nkat_checkpoint.pkl"
            if checkpoint.exists():
                size_kb = checkpoint.stat().st_size / 1024
                mod_time = datetime.fromtimestamp(checkpoint.stat().st_mtime)
                print(f"      💾 {size_kb:.1f}KB ({mod_time.strftime('%H:%M:%S')})")
            else:
                print("      📭 チェックポイントなし")
    
    # 3. 結果ファイルの状況
    print("\n📊 3. 生成済み結果ファイル:")
    print("-" * 50)
    
    result_files = [
        ("theta_optimization_comprehensive_*.json", "θ最適化結果"),
        ("nkat_riemann_hypothesis_complete_proof.png", "リーマン証明図"),
        ("riemann_hypothesis_proof_certificate.txt", "証明証明書"),
        ("*_theta_*_result.json", "個別θテスト結果")
    ]
    
    for pattern, description in result_files:
        matching_files = list(Path(".").glob(pattern))
        if matching_files:
            latest = max(matching_files, key=lambda f: f.stat().st_mtime)
            mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
            print(f"   📄 {description}: {latest.name} ({mod_time.strftime('%H:%M:%S')})")
        else:
            print(f"   📭 {description}: 未生成")
    
    # 4. リソース使用状況
    print("\n📊 4. リソース使用状況:")
    print("-" * 50)
    
    total_recovery_size = 0
    recovery_dirs = list(Path(".").glob("nkat_recovery_*"))
    
    for recovery_dir in recovery_dirs:
        if recovery_dir.is_dir():
            dir_size = sum(f.stat().st_size for f in recovery_dir.rglob('*') if f.is_file())
            total_recovery_size += dir_size
    
    print(f"   💾 総リカバリーデータ: {total_recovery_size / (1024*1024):.2f}MB")
    print(f"   📁 リカバリーディレクトリ数: {len(recovery_dirs)}個")
    
    # 5. 推定完了時間
    print("\n📊 5. 完了時間推定:")
    print("-" * 50)
    
    if main_recovery.exists():
        metadata_file = main_recovery / "nkat_session_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            start_time_str = metadata.get('start_time', '')
            last_update_str = metadata.get('last_update', '')
            
            if start_time_str and last_update_str:
                start_time = datetime.fromisoformat(start_time_str)
                last_update = datetime.fromisoformat(last_update_str)
                elapsed = (last_update - start_time).total_seconds()
                
                # 進行度に基づく推定（粗い推定）
                computation_state = metadata.get('computation_state', '')
                if 'critical_zeros_computation' in computation_state:
                    estimated_progress = 0.3  # 30%程度
                elif 'off_critical' in computation_state:
                    estimated_progress = 0.5
                elif 'functional_equation' in computation_state:
                    estimated_progress = 0.7
                else:
                    estimated_progress = 0.1
                
                if estimated_progress > 0:
                    total_estimated = elapsed / estimated_progress
                    remaining = total_estimated - elapsed
                    
                    print(f"   ⏱️ 経過時間: {elapsed/3600:.1f}時間")
                    print(f"   📈 推定進行度: {estimated_progress*100:.0f}%")
                    print(f"   ⏰ 推定残り時間: {remaining/3600:.1f}時間")
                    
                    completion_time = datetime.now() + timedelta(seconds=remaining)
                    print(f"   🎯 推定完了時刻: {completion_time.strftime('%H:%M:%S')}")
                else:
                    print("   ⚠️ 進行度推定不可")
            else:
                print("   ⚠️ 時間情報不足")
        else:
            print("   ⚠️ メタデータなし")
    
    # 6. 推奨アクション
    print("\n📊 6. 推奨アクション:")
    print("-" * 50)
    
    recommendations = []
    
    # メイン計算が動いているかチェック
    if main_recovery.exists():
        checkpoint_file = main_recovery / "nkat_checkpoint.pkl"
        if checkpoint_file.exists():
            mod_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
            time_since_update = (datetime.now() - mod_time).total_seconds()
            
            if time_since_update < 600:  # 10分以内
                recommendations.append("🟢 メイン計算が正常に進行中")
            elif time_since_update < 3600:  # 1時間以内
                recommendations.append("🟡 メイン計算が一時停止中 - 監視継続")
            else:
                recommendations.append("🔴 メイン計算が長時間停止 - 再起動を検討")
    
    # θ最適化の状況チェック
    theta_results = list(Path(".").glob("theta_optimization_comprehensive_*.json"))
    if theta_results:
        recommendations.append("✅ θ最適化実験完了 - 結果を確認")
    else:
        recommendations.append("🟡 θ最適化実験進行中 - 完了を待機")
    
    # リソース状況
    if total_recovery_size > 100 * 1024 * 1024:  # 100MB以上
        recommendations.append("💾 リカバリーデータが大容量 - 定期バックアップ推奨")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n" + "="*80)
    print("🔥💎 NKAT: Don't hold back. Give it your all!! 💎🔥")
    print("="*80)

if __name__ == "__main__":
    generate_progress_report() 