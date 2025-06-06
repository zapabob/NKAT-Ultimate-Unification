#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ NKAT理論 究極電源断リカバリーシステム
マルチレベルチェックポイント + 自動復旧 + クラウドバックアップ機能を実装します

Don't hold back. Give it your all!! 🔥

NKAT Research Team 2025
"""

import os
import sys
import json
import pickle
import sqlite3
import hashlib
import shutil
import threading
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# CUDAの条件付きインポート
try:
    import cupy as cp
    CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    cp = np

class NKATUltimateRecoverySystem:
    """🛡️ NKAT究極電源断リカバリーシステム"""
    
    def __init__(self, base_dir="recovery_data", cloud_backup=True):
        """
        🏗️ 初期化
        
        Args:
            base_dir: リカバリーデータベースディレクトリ
            cloud_backup: クラウドバックアップ有効化
        """
        print("🛡️ NKAT 究極電源断リカバリーシステム起動！")
        print("="*70)
        
        self.base_dir = Path(base_dir)
        self.cloud_backup = cloud_backup
        
        # ディレクトリ構造構築
        self.setup_directory_structure()
        
        # データベース初期化
        self.setup_database()
        
        # 自動バックアップシステム
        self.auto_backup_enabled = True
        self.backup_interval = 60  # 60秒間隔
        
        # 緊急通知システム
        self.emergency_notification = True
        
        # バックアップスレッド開始
        self.start_auto_backup_thread()
        
        print("✅ リカバリーシステム初期化完了")
        print(f"📁 ベースディレクトリ: {self.base_dir.absolute()}")
        print(f"☁️ クラウドバックアップ: {'有効' if cloud_backup else '無効'}")
        
    def setup_directory_structure(self):
        """📁 ディレクトリ構造構築"""
        self.dirs = {
            'main': self.base_dir,
            'checkpoints': self.base_dir / 'checkpoints',
            'emergency': self.base_dir / 'emergency',
            'archives': self.base_dir / 'archives',
            'temp': self.base_dir / 'temp',
            'metadata': self.base_dir / 'metadata',
            'logs': self.base_dir / 'logs',
            'cloud_sync': self.base_dir / 'cloud_sync'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"📁 {len(self.dirs)}個のリカバリーディレクトリ作成完了")
    
    def setup_database(self):
        """🗃️ SQLiteデータベース初期化"""
        self.db_path = self.dirs['metadata'] / 'recovery.db'
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # チェックポイント管理テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    problem_name TEXT NOT NULL,
                    checkpoint_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    success_rate REAL,
                    metadata TEXT,
                    is_valid BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 復旧履歴テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recovery_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recovery_timestamp TEXT NOT NULL,
                    checkpoint_id INTEGER,
                    recovery_success BOOLEAN,
                    recovery_time_seconds REAL,
                    error_message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints (id)
                )
            """)
            
            # システム状態テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    system_status TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    gpu_usage REAL,
                    disk_usage REAL,
                    active_problems TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        
        print("🗃️ データベース初期化完了")
    
    def create_checkpoint(self, problem_name, data, checkpoint_type="standard"):
        """
        🔄 マルチレベルチェックポイント作成
        
        Args:
            problem_name: 問題名
            data: 保存データ
            checkpoint_type: チェックポイント種別
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        # チェックポイントファイル生成
        checkpoint_data = {
            'timestamp': timestamp,
            'problem_name': problem_name,
            'checkpoint_type': checkpoint_type,
            'data': self._serialize_for_storage(data),
            'system_info': self._get_system_info(),
            'nkat_version': '2025.06.04',
            'cuda_available': CUDA_AVAILABLE
        }
        
        # 保存パス決定
        file_name = f"{problem_name}_{checkpoint_type}_{timestamp}.pkl"
        file_path = self.dirs['checkpoints'] / file_name
        
        try:
            # メインチェックポイント保存
            with open(file_path, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # ファイルハッシュ計算
            file_hash = self._calculate_file_hash(file_path)
            file_size = file_path.stat().st_size
            
            # 緊急チェックポイント（高速アクセス用）
            if checkpoint_type in ['critical', 'milestone']:
                emergency_path = self.dirs['emergency'] / file_name
                shutil.copy2(file_path, emergency_path)
            
            # データベース記録
            self._record_checkpoint(
                timestamp, problem_name, checkpoint_type,
                str(file_path), file_hash, file_size, data
            )
            
            # クラウド同期
            if self.cloud_backup:
                self._schedule_cloud_sync(file_path)
            
            print(f"✅ チェックポイント作成: {problem_name} ({checkpoint_type})")
            print(f"   📁 サイズ: {file_size/1024**2:.2f}MB")
            print(f"   🔒 ハッシュ: {file_hash[:16]}...")
            
            return str(file_path)
            
        except Exception as e:
            print(f"❌ チェックポイント作成エラー: {e}")
            self._log_error('checkpoint_creation', str(e))
            return None
    
    def _serialize_for_storage(self, data):
        """💾 ストレージ用データシリアライズ"""
        if isinstance(data, dict):
            return {k: self._serialize_for_storage(v) for k, v in data.items()}
        elif hasattr(data, 'get') and CUDA_AVAILABLE:
            # CuPy配列処理
            return data.get() if hasattr(data, 'get') else data
        elif isinstance(data, np.ndarray):
            # NumPy配列は圧縮保存
            return {'__numpy_array__': True, 'data': data.tobytes(), 'shape': data.shape, 'dtype': str(data.dtype)}
        else:
            return data
    
    def _get_system_info(self):
        """📊 システム情報取得"""
        try:
            import psutil
            
            info = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('.').percent,
                'gpu_available': CUDA_AVAILABLE
            }
            
            if CUDA_AVAILABLE:
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        info['gpu_load'] = gpu.load
                        info['gpu_memory'] = gpu.memoryUtil
                        info['gpu_temperature'] = gpu.temperature
                except:
                    pass
            
            return info
        except:
            return {'status': 'system_info_unavailable'}
    
    def _calculate_file_hash(self, file_path):
        """🔒 ファイルハッシュ計算"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _record_checkpoint(self, timestamp, problem_name, checkpoint_type, file_path, file_hash, file_size, data):
        """📝 チェックポイント記録"""
        # 成功率計算（簡略化）
        success_rate = self._estimate_success_rate(data)
        
        # メタデータ生成
        metadata = json.dumps({
            'data_types': [type(v).__name__ for v in data.values()] if isinstance(data, dict) else [type(data).__name__],
            'complexity_estimate': len(str(data))
        })
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO checkpoints 
                (timestamp, problem_name, checkpoint_type, file_path, file_hash, file_size, success_rate, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, problem_name, checkpoint_type, file_path, file_hash, file_size, success_rate, metadata))
            conn.commit()
    
    def _estimate_success_rate(self, data):
        """📈 成功率推定"""
        if isinstance(data, dict):
            if 'verification' in data and 'confidence_score' in data['verification']:
                return data['verification']['confidence_score']
        return 0.5  # デフォルト
    
    def find_latest_checkpoint(self, problem_name=None, checkpoint_type=None):
        """🔍 最新チェックポイント検索"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM checkpoints WHERE is_valid = 1"
            params = []
            
            if problem_name:
                query += " AND problem_name = ?"
                params.append(problem_name)
            
            if checkpoint_type:
                query += " AND checkpoint_type = ?"
                params.append(checkpoint_type)
            
            query += " ORDER BY created_at DESC LIMIT 1"
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            if result:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, result))
            return None
    
    def recover_from_checkpoint(self, checkpoint_id=None, checkpoint_path=None):
        """🔄 チェックポイントからの復旧"""
        start_time = time.time()
        recovery_success = False
        error_message = None
        
        try:
            # チェックポイント情報取得
            if checkpoint_id:
                checkpoint_info = self._get_checkpoint_by_id(checkpoint_id)
                if not checkpoint_info:
                    raise ValueError(f"チェックポイントID {checkpoint_id} が見つかりません")
                checkpoint_path = checkpoint_info['file_path']
            
            if not checkpoint_path or not Path(checkpoint_path).exists():
                # 自動最適チェックポイント検索
                checkpoint_info = self._find_optimal_checkpoint()
                if not checkpoint_info:
                    raise ValueError("復旧可能なチェックポイントが見つかりません")
                checkpoint_path = checkpoint_info['file_path']
            
            print(f"🔄 チェックポイントから復旧中: {Path(checkpoint_path).name}")
            
            # データ読み込み
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # データ検証
            if not self._validate_checkpoint_data(checkpoint_data):
                raise ValueError("チェックポイントデータが破損しています")
            
            # データ復元
            restored_data = self._deserialize_from_storage(checkpoint_data['data'])
            
            recovery_success = True
            recovery_time = time.time() - start_time
            
            # 復旧履歴記録
            self._record_recovery(checkpoint_id, recovery_success, recovery_time, None)
            
            print(f"✅ 復旧完了！ ({recovery_time:.2f}秒)")
            print(f"   📅 タイムスタンプ: {checkpoint_data['timestamp']}")
            print(f"   🎯 問題: {checkpoint_data['problem_name']}")
            print(f"   📊 種別: {checkpoint_data['checkpoint_type']}")
            
            return restored_data
            
        except Exception as e:
            error_message = str(e)
            recovery_time = time.time() - start_time
            self._record_recovery(checkpoint_id, False, recovery_time, error_message)
            print(f"❌ 復旧エラー: {e}")
            return None
    
    def _get_checkpoint_by_id(self, checkpoint_id):
        """📋 ID指定チェックポイント取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,))
            result = cursor.fetchone()
            
            if result:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, result))
            return None
    
    def _find_optimal_checkpoint(self):
        """🎯 最適チェックポイント自動検索"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 成功率とタイムスタンプでソート
            cursor.execute("""
                SELECT * FROM checkpoints 
                WHERE is_valid = 1 
                ORDER BY success_rate DESC, created_at DESC 
                LIMIT 5
            """)
            
            candidates = cursor.fetchall()
            
            # 存在確認とファイル検証
            for candidate in candidates:
                file_path = candidate[4]  # file_path column
                if Path(file_path).exists():
                    # ハッシュ検証
                    current_hash = self._calculate_file_hash(file_path)
                    stored_hash = candidate[5]  # file_hash column
                    
                    if current_hash == stored_hash:
                        columns = [description[0] for description in cursor.description]
                        return dict(zip(columns, candidate))
            
            return None
    
    def _validate_checkpoint_data(self, checkpoint_data):
        """✅ チェックポイントデータ検証"""
        required_fields = ['timestamp', 'problem_name', 'data']
        return all(field in checkpoint_data for field in required_fields)
    
    def _deserialize_from_storage(self, data):
        """🔄 ストレージからデータ復元"""
        if isinstance(data, dict):
            if '__numpy_array__' in data:
                # NumPy配列復元
                return np.frombuffer(data['data'], dtype=data['dtype']).reshape(data['shape'])
            else:
                return {k: self._deserialize_from_storage(v) for k, v in data.items()}
        else:
            return data
    
    def _record_recovery(self, checkpoint_id, success, recovery_time, error_message):
        """📝 復旧履歴記録"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO recovery_history 
                (recovery_timestamp, checkpoint_id, recovery_success, recovery_time_seconds, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), checkpoint_id, success, recovery_time, error_message))
            conn.commit()
    
    def start_auto_backup_thread(self):
        """🔄 自動バックアップスレッド開始"""
        def auto_backup_loop():
            while self.auto_backup_enabled:
                try:
                    self._perform_maintenance()
                    time.sleep(self.backup_interval)
                except Exception as e:
                    self._log_error('auto_backup', str(e))
        
        self.backup_thread = threading.Thread(target=auto_backup_loop, daemon=True)
        self.backup_thread.start()
        print("🔄 自動バックアップスレッド開始")
    
    def _perform_maintenance(self):
        """🔧 定期メンテナンス実行"""
        # システム状態記録
        self._record_system_state()
        
        # 古いチェックポイントのアーカイブ
        self._archive_old_checkpoints()
        
        # 破損ファイルチェック
        self._check_file_integrity()
    
    def _record_system_state(self):
        """📊 システム状態記録"""
        system_info = self._get_system_info()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_state 
                (timestamp, system_status, cpu_usage, memory_usage, gpu_usage, disk_usage, active_problems)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                'running',
                system_info.get('cpu_percent', 0),
                system_info.get('memory_percent', 0),
                system_info.get('gpu_load', 0),
                system_info.get('disk_usage', 0),
                json.dumps(['nkat_millennium_analysis'])
            ))
            conn.commit()
    
    def _archive_old_checkpoints(self, days_threshold=7):
        """📦 古いチェックポイントのアーカイブ"""
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT file_path FROM checkpoints 
                WHERE created_at < ? AND checkpoint_type != 'critical'
            """, (threshold_date.isoformat(),))
            
            old_files = cursor.fetchall()
            
            if old_files:
                # アーカイブZIP作成
                archive_name = f"checkpoint_archive_{datetime.now().strftime('%Y%m%d')}.zip"
                archive_path = self.dirs['archives'] / archive_name
                
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for (file_path,) in old_files:
                        if Path(file_path).exists():
                            zipf.write(file_path, Path(file_path).name)
                            Path(file_path).unlink()  # 元ファイル削除
                
                print(f"📦 {len(old_files)}個のチェックポイントをアーカイブ: {archive_name}")
    
    def _check_file_integrity(self):
        """🔍 ファイル整合性チェック"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, file_path, file_hash FROM checkpoints WHERE is_valid = 1")
            
            for checkpoint_id, file_path, stored_hash in cursor.fetchall():
                if Path(file_path).exists():
                    current_hash = self._calculate_file_hash(file_path)
                    if current_hash != stored_hash:
                        # ファイル破損検出
                        cursor.execute("UPDATE checkpoints SET is_valid = 0 WHERE id = ?", (checkpoint_id,))
                        print(f"⚠️ 破損ファイル検出: {Path(file_path).name}")
            
            conn.commit()
    
    def _schedule_cloud_sync(self, file_path):
        """☁️ クラウド同期スケジュール"""
        # クラウド同期は簡略化実装
        cloud_path = self.dirs['cloud_sync'] / Path(file_path).name
        try:
            shutil.copy2(file_path, cloud_path)
            print(f"☁️ クラウド同期: {Path(file_path).name}")
        except Exception as e:
            self._log_error('cloud_sync', str(e))
    
    def _log_error(self, operation, error_message):
        """📝 エラーログ記録"""
        log_file = self.dirs['logs'] / f"error_log_{datetime.now().strftime('%Y%m%d')}.txt"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] {operation}: {error_message}\n")
    
    def generate_recovery_report(self):
        """📊 復旧レポート生成"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # チェックポイント統計
            cursor.execute("SELECT COUNT(*), AVG(success_rate) FROM checkpoints WHERE is_valid = 1")
            checkpoint_stats = cursor.fetchone()
            
            # 復旧統計
            cursor.execute("SELECT COUNT(*), SUM(recovery_success), AVG(recovery_time_seconds) FROM recovery_history")
            recovery_stats = cursor.fetchone()
            
            # 最新システム状態
            cursor.execute("SELECT * FROM system_state ORDER BY created_at DESC LIMIT 1")
            latest_state = cursor.fetchone()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'checkpoint_statistics': {
                    'total_checkpoints': checkpoint_stats[0] or 0,
                    'average_success_rate': checkpoint_stats[1] or 0.0
                },
                'recovery_statistics': {
                    'total_recoveries': recovery_stats[0] or 0,
                    'successful_recoveries': recovery_stats[1] or 0,
                    'average_recovery_time': recovery_stats[2] or 0.0
                },
                'system_status': {
                    'last_update': latest_state[1] if latest_state else 'unknown',
                    'cpu_usage': latest_state[3] if latest_state else 0,
                    'memory_usage': latest_state[4] if latest_state else 0,
                    'gpu_usage': latest_state[5] if latest_state else 0
                } if latest_state else {}
            }
            
            # レポートファイル保存
            report_file = self.dirs['logs'] / f"recovery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print("📊 復旧レポート生成完了")
            print(f"   💾 チェックポイント数: {report['checkpoint_statistics']['total_checkpoints']}")
            print(f"   🎯 平均成功率: {report['checkpoint_statistics']['average_success_rate']:.3f}")
            print(f"   🔄 復旧回数: {report['recovery_statistics']['total_recoveries']}")
            
            return report
    
    def emergency_shutdown_protection(self):
        """🚨 緊急シャットダウン保護"""
        print("🚨 緊急シャットダウン検出！緊急保護実行中...")
        
        # 重要データの緊急バックアップ
        emergency_backup = {
            'timestamp': datetime.now().isoformat(),
            'emergency_type': 'power_failure',
            'system_state': self._get_system_info()
        }
        
        emergency_file = self.dirs['emergency'] / f"emergency_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(emergency_file, 'w') as f:
            json.dump(emergency_backup, f, indent=2)
        
        # データベース緊急コミット
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA synchronous = FULL")
                conn.commit()
        except:
            pass
        
        print("✅ 緊急保護完了")
    
    def stop_auto_backup(self):
        """⏹️ 自動バックアップ停止"""
        self.auto_backup_enabled = False
        print("⏹️ 自動バックアップ停止")

def main():
    """🚀 メイン実行関数"""
    print("🛡️ NKAT 究極電源断リカバリーシステム")
    print("Don't hold back. Give it your all!! 🔥")
    print("="*70)
    
    try:
        # リカバリーシステム初期化
        recovery_system = NKATUltimateRecoverySystem()
        
        # テストチェックポイント作成
        test_data = {
            'problem': 'riemann_hypothesis',
            'results': {'zeros_found': 15, 'confidence': 0.95},
            'verification': {'confidence_score': 0.95}
        }
        
        checkpoint_path = recovery_system.create_checkpoint(
            'riemann_hypothesis', test_data, 'milestone'
        )
        
        # 復旧テスト
        if checkpoint_path:
            print("\n🔄 復旧テスト実行中...")
            recovered_data = recovery_system.recover_from_checkpoint(checkpoint_path=checkpoint_path)
            
            if recovered_data:
                print("✅ 復旧テスト成功！")
            else:
                print("❌ 復旧テスト失敗")
        
        # レポート生成
        recovery_system.generate_recovery_report()
        
        print("\n🏆 リカバリーシステム初期化・テスト完了！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 緊急停止検出")
        if 'recovery_system' in locals():
            recovery_system.emergency_shutdown_protection()
    except Exception as e:
        print(f"❌ エラー: {e}")
    finally:
        if 'recovery_system' in locals():
            recovery_system.stop_auto_backup()
        print("🔥 システム終了")

if __name__ == "__main__":
    main() 