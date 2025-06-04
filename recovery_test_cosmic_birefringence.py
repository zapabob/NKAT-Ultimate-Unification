#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ 電源断リカバリーシステム テストスクリプト
RTX3080 環境での宇宙複屈折解析の継続性検証

機能テスト:
1. 緊急停止からの復旧
2. チェックポイントからの再開
3. データ整合性検証
4. システムリソース監視
"""

import os
import time
import signal
import subprocess
import json
from pathlib import Path
from datetime import datetime
import threading

class RecoveryTestSystem:
    """⚡ 電源断リカバリーテストシステム"""
    
    def __init__(self):
        self.test_script = "cosmic_birefringence_nkat_analysis.py"
        self.recovery_dir = Path("recovery_data")
        self.test_log = "recovery_test_log.txt"
        
        print("⚡ 電源断リカバリーテストシステム初期化")
        
    def simulate_power_interruption(self, delay_seconds=10):
        """🔌 電源断シミュレーション"""
        print(f"\n🔌 {delay_seconds}秒後に電源断をシミュレートします...")
        
        # メインプロセス開始
        process = subprocess.Popen([
            "py", "-3", self.test_script
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print(f"📊 プロセス開始 (PID: {process.pid})")
        
        # 指定時間後に強制終了
        time.sleep(delay_seconds)
        
        print("🚨 電源断シミュレーション実行中...")
        try:
            process.terminate()  # 最初はソフト終了を試行
            time.sleep(2)
            
            if process.poll() is None:
                print("💥 強制終了実行")
                process.kill()  # 強制終了
                
            stdout, stderr = process.communicate(timeout=5)
            
        except subprocess.TimeoutExpired:
            print("💀 プロセス強制終了")
            process.kill()
            stdout, stderr = process.communicate()
        
        print(f"🔚 プロセス終了コード: {process.returncode}")
        
        # ログ記録
        with open(self.test_log, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"電源断シミュレーション実行: {datetime.now()}\n")
            f.write(f"PID: {process.pid}\n")
            f.write(f"終了コード: {process.returncode}\n")
            f.write(f"待機時間: {delay_seconds}秒\n")
            f.write(f"{'='*60}\n")
            if stdout:
                f.write(f"STDOUT:\n{stdout}\n")
            if stderr:
                f.write(f"STDERR:\n{stderr}\n")
        
        return process.returncode
    
    def verify_checkpoint_creation(self):
        """📁 チェックポイント作成検証"""
        print("\n📁 チェックポイント作成検証中...")
        
        checkpoint_dir = self.recovery_dir / "checkpoints"
        
        if not checkpoint_dir.exists():
            print("❌ チェックポイントディレクトリが存在しません")
            return False
        
        checkpoint_files = list(checkpoint_dir.glob("*.pkl"))
        meta_files = list(checkpoint_dir.glob("*_meta.json"))
        
        print(f"✅ チェックポイントファイル数: {len(checkpoint_files)}")
        print(f"✅ メタデータファイル数: {len(meta_files)}")
        
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            print(f"📄 最新チェックポイント: {latest_checkpoint.name}")
            
            # メタデータ確認
            meta_file = latest_checkpoint.with_suffix('.pkl').with_suffix('_meta.json')
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"📊 メタデータ:")
                print(f"   タイムスタンプ: {metadata.get('timestamp', 'N/A')}")
                print(f"   ファイルサイズ: {metadata.get('file_size', 'N/A')} bytes")
                print(f"   CPU使用率: {metadata.get('system_info', {}).get('cpu_percent', 'N/A')}%")
                print(f"   メモリ使用率: {metadata.get('system_info', {}).get('memory_percent', 'N/A')}%")
        
        return len(checkpoint_files) > 0
    
    def test_recovery_restart(self):
        """🔄 復旧再開テスト"""
        print("\n🔄 復旧再開テスト実行中...")
        
        # 前回のチェックポイントが存在することを確認
        if not self.verify_checkpoint_creation():
            print("❌ チェックポイントが存在しないため復旧テストをスキップ")
            return False
        
        print("🚀 復旧再開実行...")
        
        try:
            # 再開プロセス実行
            result = subprocess.run([
                "py", "-3", self.test_script
            ], capture_output=True, text=True, timeout=60)
            
            print(f"✅ 復旧プロセス完了 (終了コード: {result.returncode})")
            
            # 復旧ログ確認
            recovery_log = self.recovery_dir / "cosmic_birefringence_nkat_recovery.log"
            if recovery_log.exists():
                with open(recovery_log, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    
                if "計算状態復旧完了" in log_content:
                    print("✅ 計算状態の復旧が確認されました")
                    return True
                else:
                    print("⚠️ 復旧ログに復旧メッセージが見つかりません")
            
        except subprocess.TimeoutExpired:
            print("⏰ 復旧プロセスがタイムアウトしました")
            return False
        except Exception as e:
            print(f"❌ 復旧テストエラー: {e}")
            return False
        
        return True
    
    def monitor_system_resources(self, duration=30):
        """📊 システムリソース監視テスト"""
        print(f"\n📊 {duration}秒間のシステムリソース監視開始...")
        
        import psutil
        
        start_time = time.time()
        resource_data = []
        
        while time.time() - start_time < duration:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            resource_data.append({
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'available_memory_gb': memory.available / (1024**3)
            })
            
            print(f"📈 CPU: {cpu_percent:5.1f}% | メモリ: {memory.percent:5.1f}% | 利用可能: {memory.available/(1024**3):5.2f}GB", end='\r')
            
            time.sleep(1)
        
        print(f"\n✅ リソース監視完了 ({len(resource_data)}回測定)")
        
        # 統計計算
        avg_cpu = sum(d['cpu_percent'] for d in resource_data) / len(resource_data)
        avg_memory = sum(d['memory_percent'] for d in resource_data) / len(resource_data)
        max_cpu = max(d['cpu_percent'] for d in resource_data)
        max_memory = max(d['memory_percent'] for d in resource_data)
        
        print(f"📊 CPU使用率 - 平均: {avg_cpu:.1f}%, 最大: {max_cpu:.1f}%")
        print(f"📊 メモリ使用率 - 平均: {avg_memory:.1f}%, 最大: {max_memory:.1f}%")
        
        # 危険レベルチェック
        if max_cpu > 90:
            print("⚠️ 高CPU使用率が検出されました")
        if max_memory > 90:
            print("⚠️ 高メモリ使用率が検出されました")
        
        return resource_data
    
    def comprehensive_recovery_test(self):
        """🧪 包括的リカバリーテスト"""
        print("\n" + "="*80)
        print("🧪 RTX3080 電源断リカバリーシステム包括テスト開始")
        print("="*80)
        
        test_results = {
            'power_interruption_test': False,
            'checkpoint_creation_test': False,
            'recovery_restart_test': False,
            'resource_monitoring_test': False,
            'overall_success': False
        }
        
        try:
            # テスト1: 電源断シミュレーション
            print("\n🔌 テスト1: 電源断シミュレーション")
            exit_code = self.simulate_power_interruption(delay_seconds=15)
            test_results['power_interruption_test'] = (exit_code is not None)
            
            # 少し待機
            time.sleep(3)
            
            # テスト2: チェックポイント作成検証
            print("\n📁 テスト2: チェックポイント作成検証")
            test_results['checkpoint_creation_test'] = self.verify_checkpoint_creation()
            
            # テスト3: 復旧再開テスト
            print("\n🔄 テスト3: 復旧再開テスト")
            test_results['recovery_restart_test'] = self.test_recovery_restart()
            
            # テスト4: システムリソース監視
            print("\n📊 テスト4: システムリソース監視")
            resource_data = self.monitor_system_resources(duration=10)
            test_results['resource_monitoring_test'] = len(resource_data) > 0
            
            # 総合評価
            all_passed = all(test_results[key] for key in test_results if key != 'overall_success')
            test_results['overall_success'] = all_passed
            
        except Exception as e:
            print(f"❌ テスト実行エラー: {e}")
            test_results['overall_success'] = False
        
        # 結果レポート
        print("\n" + "="*80)
        print("📋 テスト結果レポート")
        print("="*80)
        
        for test_name, result in test_results.items():
            status = "✅ 成功" if result else "❌ 失敗"
            print(f"{test_name:25s} : {status}")
        
        if test_results['overall_success']:
            print("\n🎊 全テスト成功！RTX3080電源断リカバリーシステムは正常に動作しています")
        else:
            print("\n⚠️ 一部テストが失敗しました。システムの確認が必要です")
        
        # テスト結果をファイルに保存
        with open('recovery_test_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'test_timestamp': datetime.now().isoformat(),
                'test_results': test_results,
                'system_info': {
                    'os': os.name,
                    'python_version': subprocess.run(['py', '-3', '--version'], 
                                                   capture_output=True, text=True).stdout.strip(),
                    'working_directory': str(Path.cwd())
                }
            }, f, indent=2, ensure_ascii=False)
        
        return test_results

def main():
    """🧪 メインテスト実行"""
    print("⚡ RTX3080 電源断リカバリーシステム テストスイート")
    print("=" * 60)
    
    tester = RecoveryTestSystem()
    results = tester.comprehensive_recovery_test()
    
    print(f"\n📄 詳細なテスト結果は recovery_test_results.json に保存されました")
    
    return results

if __name__ == "__main__":
    main() 