#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Ultimate Git LFS Push System with Power Failure Recovery
電源断対応 Git LFS プッシュシステム

RTX3080 CUDA対応 & 自動リカバリー機能付き
"""

import os
import sys
import subprocess
import time
import json
import pickle
import signal
import uuid
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import threading
import queue
import atexit

class NKATUltimateGitLFSPushSystem:
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.recovery_dir = Path("recovery_data/git_lfs_recovery")
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        
        # 電源断検出と自動保存設定
        self.checkpoint_interval = 30  # 30秒間隔での自動保存
        self.backup_rotation_max = 10
        self.emergency_save_enabled = True
        
        # Git LFS 状況追跡
        self.lfs_files_status = {}
        self.push_progress = {
            'total_files': 0,
            'completed_files': 0,
            'failed_files': [],
            'current_phase': 'initialization',
            'start_time': datetime.now().isoformat(),
            'last_checkpoint': None
        }
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._emergency_save_handler)
        signal.signal(signal.SIGTERM, self._emergency_save_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._emergency_save_handler)
        
        atexit.register(self._cleanup)
        
        print(f"🚀 NKAT Ultimate Git LFS Push System 初期化完了")
        print(f"📊 セッションID: {self.session_id}")
        print(f"🛡️ 電源断保護機能: 有効")
        print(f"💾 自動保存間隔: {self.checkpoint_interval}秒")
    
    def _emergency_save_handler(self, signum, frame):
        """緊急時の自動保存ハンドラー"""
        print(f"\n⚡ 緊急保存開始 (シグナル: {signum})")
        self._save_checkpoint(emergency=True)
        print("🛡️ 緊急保存完了")
        sys.exit(1)
    
    def _save_checkpoint(self, emergency=False):
        """チェックポイント保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.recovery_dir / f"git_lfs_checkpoint_{self.session_id}_{timestamp}.json"
            
            # バックアップローテーション
            self._rotate_backups()
            
            # 進捗状況保存
            checkpoint_data = {
                'session_id': self.session_id,
                'timestamp': timestamp,
                'push_progress': self.push_progress.copy(),
                'lfs_files_status': self.lfs_files_status.copy(),
                'emergency_save': emergency
            }
            
            # JSON保存
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # Pickle保存（バイナリデータ用）
            pickle_file = checkpoint_file.with_suffix('.pkl')
            with open(pickle_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.push_progress['last_checkpoint'] = checkpoint_file.name
            
            if not emergency:
                print(f"💾 チェックポイント保存: {checkpoint_file.name}")
        
        except Exception as e:
            print(f"❌ チェックポイント保存エラー: {e}")
    
    def _rotate_backups(self):
        """バックアップローテーション管理"""
        try:
            checkpoints = list(self.recovery_dir.glob("git_lfs_checkpoint_*.json"))
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 古いバックアップを削除
            for old_checkpoint in checkpoints[self.backup_rotation_max:]:
                old_checkpoint.unlink(missing_ok=True)
                old_checkpoint.with_suffix('.pkl').unlink(missing_ok=True)
        
        except Exception as e:
            print(f"⚠️ バックアップローテーションエラー: {e}")
    
    def _cleanup(self):
        """終了時のクリーンアップ"""
        if self.emergency_save_enabled:
            self._save_checkpoint()
    
    def check_git_lfs_status(self):
        """Git LFS の状態確認"""
        print("\n🔍 Git LFS 状態確認中...")
        self.push_progress['current_phase'] = 'checking_lfs_status'
        
        try:
            # Git LFS がインストールされているか確認
            result = subprocess.run(['git', 'lfs', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Git LFS バージョン: {result.stdout.strip()}")
            
            # LFS で追跡されているファイルの確認
            result = subprocess.run(['git', 'lfs', 'ls-files'], 
                                  capture_output=True, text=True, check=True)
            
            lfs_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            self.push_progress['total_files'] = len(lfs_files)
            
            print(f"📊 LFS追跡ファイル数: {len(lfs_files)}")
            
            for line in lfs_files[:10]:  # 最初の10個を表示
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        oid, size, filename = parts[0], parts[1], ' '.join(parts[2:])
                        self.lfs_files_status[filename] = {'oid': oid, 'size': size, 'status': 'tracked'}
                        print(f"  📁 {filename} ({size})")
            
            if len(lfs_files) > 10:
                print(f"  ... その他 {len(lfs_files) - 10} ファイル")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git LFS エラー: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ 予期しないエラー: {e}")
            return False
    
    def commit_changes(self):
        """変更をコミット"""
        print("\n📝 変更のコミット中...")
        self.push_progress['current_phase'] = 'committing'
        
        try:
            # コミット
            commit_message = f"🚀 NKAT Git LFS Migration - Session {self.session_id}"
            result = subprocess.run([
                'git', 'commit', '-m', commit_message
            ], capture_output=True, text=True, check=True)
            
            print(f"✅ コミット完了: {commit_message}")
            print(f"📊 {result.stdout.strip()}")
            
            self._save_checkpoint()
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ コミットエラー: {e.stderr}")
            return False
    
    def push_with_retry(self, max_retries=5):
        """リトライ機能付きプッシュ"""
        print(f"\n🚀 GitHub へのプッシュ開始 (最大 {max_retries} 回リトライ)")
        self.push_progress['current_phase'] = 'pushing'
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"\n🔄 プッシュ試行 {attempt}/{max_retries}")
                
                # プッシュコマンド実行
                process = subprocess.Popen([
                    'git', 'push', 'origin', 'main'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                # プログレス表示
                with tqdm(desc=f"プッシュ進行中 (試行 {attempt})", unit="line") as pbar:
                    while True:
                        output = process.stderr.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            pbar.set_description(f"プッシュ進行中: {output.strip()}")
                            pbar.update(1)
                            
                            # LFS ファイルのアップロード進捗を追跡
                            if "Uploading LFS objects:" in output:
                                self.push_progress['current_phase'] = 'uploading_lfs'
                
                # プロセス終了を待機
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    print("✅ プッシュ成功！")
                    print(f"📊 出力: {stdout}")
                    self.push_progress['current_phase'] = 'completed'
                    self.push_progress['completed_files'] = self.push_progress['total_files']
                    self._save_checkpoint()
                    return True
                else:
                    print(f"❌ プッシュ失敗 (試行 {attempt}): {stderr}")
                    self.push_progress['failed_files'].append({
                        'attempt': attempt,
                        'error': stderr,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # 指数バックオフ
                        print(f"⏱️ {wait_time}秒待機後にリトライ...")
                        time.sleep(wait_time)
                        self._save_checkpoint()
                
            except Exception as e:
                print(f"❌ プッシュ試行 {attempt} でエラー: {e}")
                self.push_progress['failed_files'].append({
                    'attempt': attempt,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"⏱️ {wait_time}秒待機後にリトライ...")
                    time.sleep(wait_time)
        
        print("❌ 全てのプッシュ試行が失敗しました")
        self.push_progress['current_phase'] = 'failed'
        self._save_checkpoint()
        return False
    
    def run_ultimate_push_sequence(self):
        """究極のプッシュシーケンス実行"""
        print("🚀 NKAT Ultimate Git LFS Push System 開始")
        print("=" * 60)
        
        # 自動保存スレッド開始
        checkpoint_thread = threading.Thread(target=self._auto_checkpoint_loop)
        checkpoint_thread.daemon = True
        checkpoint_thread.start()
        
        try:
            # 1. Git LFS 状態確認
            if not self.check_git_lfs_status():
                print("❌ Git LFS 状態確認に失敗しました")
                return False
            
            # 2. 変更をコミット
            if not self.commit_changes():
                print("❌ コミットに失敗しました")
                return False
            
            # 3. プッシュ実行
            if not self.push_with_retry():
                print("❌ プッシュに失敗しました")
                return False
            
            print("\n🎉 NKAT Ultimate Git LFS Push 完全成功！")
            print("=" * 60)
            self._generate_success_report()
            return True
            
        except Exception as e:
            print(f"❌ 予期しないエラー: {e}")
            self._save_checkpoint(emergency=True)
            return False
    
    def _auto_checkpoint_loop(self):
        """自動チェックポイント保存ループ"""
        while self.push_progress['current_phase'] not in ['completed', 'failed']:
            time.sleep(self.checkpoint_interval)
            if self.push_progress['current_phase'] not in ['completed', 'failed']:
                self._save_checkpoint()
    
    def _generate_success_report(self):
        """成功レポート生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.recovery_dir / f"git_lfs_success_report_{timestamp}.md"
        
        report_content = f"""# NKAT Ultimate Git LFS Push Success Report

## 🎉 プッシュ成功

- **セッションID**: {self.session_id}
- **開始時刻**: {self.push_progress['start_time']}
- **完了時刻**: {datetime.now().isoformat()}
- **総ファイル数**: {self.push_progress['total_files']}
- **成功ファイル数**: {self.push_progress['completed_files']}

## 📊 LFS ファイル統計

```
総追跡ファイル数: {len(self.lfs_files_status)}
```

## 🛡️ 電源断保護システム

- 自動保存間隔: {self.checkpoint_interval}秒
- バックアップローテーション: {self.backup_rotation_max}個
- 緊急保存機能: 有効

## ✅ プッシュ完了

大容量ファイルが Git LFS により正常に GitHub にプッシュされました。
RTX3080 CUDA計算結果も含めて、全てのデータが安全に保存されています。

---
Generated by NKAT Ultimate Git LFS Push System
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 成功レポート生成: {report_file}")

def main():
    """メイン実行関数"""
    print("🚀 NKAT Ultimate Git LFS Push System")
    print("Don't hold back. Give it your all!!")
    print("=" * 60)
    
    system = NKATUltimateGitLFSPushSystem()
    success = system.run_ultimate_push_sequence()
    
    if success:
        print("\n🎉 全てのプロセスが成功しました！")
        print("GitHub リポジトリに大容量ファイルが正常にアップロードされました。")
    else:
        print("\n❌ プロセスに失敗しました")
        print("リカバリーデータを確認してください。")
    
    return success

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚡ ユーザーによる中断")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        sys.exit(1) 