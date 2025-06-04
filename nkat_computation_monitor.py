#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥💎‼ NKAT計算監視システム ‼💎🔥
実行中の計算状況をリアルタイム監視
電源断リカバリー状況も確認
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

def monitor_recovery_directories():
    """リカバリーディレクトリの監視"""
    recovery_dirs = [
        "nkat_recovery_theta_1e12",
        "nkat_recovery_theta_1e-08",
        "nkat_recovery_theta_1e-10", 
        "nkat_recovery_theta_1e-14",
        "nkat_recovery_theta_1e-16",
        "nkat_theta_optimization_results",
        "nkat_full_computation_theta_1e-12"
    ]
    
    print("🔥💎 NKAT計算監視システム起動 💎🔥")
    print("="*70)
    print(f"監視開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    while True:
        try:
            print(f"\n📊 計算状況監視 - {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 50)
            
            for dir_name in recovery_dirs:
                dir_path = Path(dir_name)
                
                if dir_path.exists():
                    print(f"\n📁 {dir_name}:")
                    
                    # チェックポイントファイル確認
                    checkpoint_file = dir_path / "nkat_checkpoint.pkl"
                    if checkpoint_file.exists():
                        size_kb = checkpoint_file.stat().st_size / 1024
                        mod_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                        print(f"   💾 チェックポイント: {size_kb:.1f}KB (更新: {mod_time.strftime('%H:%M:%S')})")
                    
                    # メタデータ確認
                    metadata_file = dir_path / "nkat_session_metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            
                            status = metadata.get('status', 'unknown')
                            last_update = metadata.get('last_update', 'N/A')
                            computation_state = metadata.get('computation_state', 'N/A')
                            
                            if last_update != 'N/A':
                                last_update = datetime.fromisoformat(last_update).strftime('%H:%M:%S')
                            
                            print(f"   📊 状態: {status}")
                            print(f"   🕐 最終更新: {last_update}")
                            print(f"   ⚙️ 計算段階: {computation_state}")
                            
                        except Exception as e:
                            print(f"   ⚠️ メタデータ読込エラー: {e}")
                    
                    # 結果ファイル確認
                    json_files = list(dir_path.glob("*.json"))
                    if json_files:
                        print(f"   📄 結果ファイル: {len(json_files)}個")
                        
                        latest_result = max(json_files, key=lambda f: f.stat().st_mtime)
                        mod_time = datetime.fromtimestamp(latest_result.stat().st_mtime)
                        print(f"   📝 最新結果: {latest_result.name} ({mod_time.strftime('%H:%M:%S')})")
                
                else:
                    print(f"\n📁 {dir_name}: 未作成")
            
            # プロセス確認
            print(f"\n🖥️ Pythonプロセス確認:")
            try:
                # PowerShellでのプロセス確認
                import subprocess
                result = subprocess.run([
                    "powershell", "-Command", 
                    "Get-Process | Where-Object {$_.ProcessName -eq 'python'} | Select-Object ProcessName, Id, CPU"
                ], capture_output=True, text=True, timeout=5)
                
                if result.stdout.strip():
                    print(result.stdout)
                else:
                    print("   📴 Pythonプロセスなし")
                    
            except Exception as e:
                print(f"   ⚠️ プロセス確認エラー: {e}")
            
            print("-" * 50)
            print("🔄 60秒後に再監視...")
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\n🛑 監視終了")
            break
        except Exception as e:
            print(f"\n❌ 監視エラー: {e}")
            time.sleep(10)

def quick_status_check():
    """簡易状況確認"""
    print("📊 NKAT計算状況簡易チェック")
    print("=" * 40)
    
    status_summary = {}
    
    recovery_dirs = [
        "nkat_recovery_theta_1e12",
        "nkat_recovery_theta_1e-08",
        "nkat_recovery_theta_1e-10", 
        "nkat_recovery_theta_1e-14",
        "nkat_recovery_theta_1e-16",
        "nkat_theta_optimization_results",
        "nkat_full_computation_theta_1e-12"
    ]
    
    for dir_name in recovery_dirs:
        dir_path = Path(dir_name)
        
        if dir_path.exists():
            checkpoint_file = dir_path / "nkat_checkpoint.pkl"
            metadata_file = dir_path / "nkat_session_metadata.json"
            
            status = "存在"
            last_activity = "不明"
            
            if checkpoint_file.exists():
                mod_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                last_activity = mod_time.strftime('%H:%M:%S')
                
                # 最近の活動判定（10分以内）
                if (datetime.now() - mod_time).seconds < 600:
                    status = "🔥 アクティブ"
                else:
                    status = "⏸️ 停止中"
            
            status_summary[dir_name] = {
                'status': status,
                'last_activity': last_activity
            }
        else:
            status_summary[dir_name] = {
                'status': '未作成',
                'last_activity': 'N/A'
            }
    
    for dir_name, info in status_summary.items():
        print(f"{dir_name:<35} | {info['status']:<12} | {info['last_activity']}")
    
    return status_summary

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_status_check()
    else:
        monitor_recovery_directories() 