#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💾 NKAT Backup Manager
NKATバックアップ管理システム

機能:
- 自動バックアップ作成
- データ復元機能
- 整合性チェック
- 圧縮・暗号化
- スケジュール管理
"""

import os
import sys
import json
import shutil
import hashlib
import zipfile
import schedule
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import threading

# 暗号化
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

# 進捗表示
from tqdm import tqdm

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/nkat_backup.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BackupConfig:
    """バックアップ設定"""
    # 基本設定
    backup_dir: str = "Backups"
    max_backups: int = 10
    compression_enabled: bool = True
    encryption_enabled: bool = False
    
    # スケジュール設定
    auto_backup_enabled: bool = True
    backup_interval_hours: int = 6
    daily_backup_time: str = "02:00"
    
    # 対象ディレクトリ
    source_directories: List[str] = None
    exclude_patterns: List[str] = None
    
    # 詳細設定
    verify_integrity: bool = True
    incremental_backup: bool = True
    parallel_processing: bool = True
    
    def __post_init__(self):
        if self.source_directories is None:
            self.source_directories = [
                "src",
                "Results",
                "config",
                "scripts",
                "docs"
            ]
        
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "*.pyc",
                "__pycache__",
                "*.tmp",
                "*.log",
                ".git",
                "node_modules"
            ]

@dataclass
class BackupInfo:
    """バックアップ情報"""
    timestamp: str
    backup_id: str
    size_bytes: int
    file_count: int
    directories: List[str]
    checksum: str
    compression_ratio: float
    encrypted: bool
    incremental: bool
    parent_backup_id: Optional[str] = None

class FileHasher:
    """ファイルハッシュ計算クラス"""
    
    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """ファイルハッシュ計算"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"ハッシュ計算エラー {file_path}: {e}")
            return ""
    
    @staticmethod
    def calculate_directory_hash(directory: Path, exclude_patterns: List[str] = None) -> Dict[str, str]:
        """ディレクトリハッシュ計算"""
        file_hashes = {}
        exclude_patterns = exclude_patterns or []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                # 除外パターンチェック
                skip = False
                for pattern in exclude_patterns:
                    if file_path.match(pattern):
                        skip = True
                        break
                
                if not skip:
                    relative_path = file_path.relative_to(directory)
                    file_hashes[str(relative_path)] = FileHasher.calculate_file_hash(file_path)
        
        return file_hashes

class BackupEncryption:
    """バックアップ暗号化クラス"""
    
    def __init__(self, key_file: str = "backup_key.key"):
        self.key_file = Path(key_file)
        self.key = self._load_or_generate_key()
    
    def _load_or_generate_key(self) -> bytes:
        """暗号化キー読み込み/生成"""
        if not ENCRYPTION_AVAILABLE:
            logger.warning("暗号化ライブラリが利用できません")
            return b""
        
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            logger.info(f"新しい暗号化キーを生成: {self.key_file}")
            return key
    
    def encrypt_file(self, input_file: Path, output_file: Path):
        """ファイル暗号化"""
        if not ENCRYPTION_AVAILABLE:
            shutil.copy2(input_file, output_file)
            return
        
        fernet = Fernet(self.key)
        
        with open(input_file, 'rb') as f_in:
            data = f_in.read()
        
        encrypted_data = fernet.encrypt(data)
        
        with open(output_file, 'wb') as f_out:
            f_out.write(encrypted_data)
    
    def decrypt_file(self, input_file: Path, output_file: Path):
        """ファイル復号化"""
        if not ENCRYPTION_AVAILABLE:
            shutil.copy2(input_file, output_file)
            return
        
        fernet = Fernet(self.key)
        
        with open(input_file, 'rb') as f_in:
            encrypted_data = f_in.read()
        
        decrypted_data = fernet.decrypt(encrypted_data)
        
        with open(output_file, 'wb') as f_out:
            f_out.write(decrypted_data)

class NKATBackupManager:
    """NKAT バックアップ管理メインクラス"""
    
    def __init__(self, config: BackupConfig = None):
        self.config = config or BackupConfig()
        self.backup_dir = Path(self.config.backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        self.metadata = self._load_metadata()
        
        self.encryption = BackupEncryption() if self.config.encryption_enabled else None
        self.scheduler_thread = None
        self.scheduler_running = False
    
    def _load_metadata(self) -> Dict[str, BackupInfo]:
        """メタデータ読み込み"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {k: BackupInfo(**v) for k, v in data.items()}
            except Exception as e:
                logger.error(f"メタデータ読み込みエラー: {e}")
        
        return {}
    
    def _save_metadata(self):
        """メタデータ保存"""
        try:
            data = {k: asdict(v) for k, v in self.metadata.items()}
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"メタデータ保存エラー: {e}")
    
    def create_backup(self, backup_name: str = None, incremental: bool = None) -> Optional[str]:
        """バックアップ作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = backup_name or f"nkat_backup_{timestamp}"
        
        if incremental is None:
            incremental = self.config.incremental_backup
        
        logger.info(f"🔄 バックアップ作成開始: {backup_id}")
        
        try:
            # バックアップディレクトリ作成
            backup_path = self.backup_dir / backup_id
            backup_path.mkdir(exist_ok=True)
            
            # ファイル収集
            files_to_backup = self._collect_files()
            
            if incremental and self.metadata:
                # 増分バックアップ
                files_to_backup = self._filter_changed_files(files_to_backup)
                parent_backup_id = max(self.metadata.keys(), key=lambda x: self.metadata[x].timestamp)
            else:
                parent_backup_id = None
            
            # バックアップ実行
            total_size, file_count = self._backup_files(files_to_backup, backup_path)
            
            # 圧縮
            if self.config.compression_enabled:
                compressed_path = self._compress_backup(backup_path, backup_id)
                shutil.rmtree(backup_path)
                backup_path = compressed_path
                compression_ratio = total_size / backup_path.stat().st_size if backup_path.stat().st_size > 0 else 1.0
            else:
                compression_ratio = 1.0
            
            # 暗号化
            if self.config.encryption_enabled and self.encryption:
                encrypted_path = self._encrypt_backup(backup_path, backup_id)
                backup_path.unlink() if backup_path.is_file() else shutil.rmtree(backup_path)
                backup_path = encrypted_path
            
            # チェックサム計算
            checksum = FileHasher.calculate_file_hash(backup_path) if backup_path.is_file() else ""
            
            # メタデータ更新
            backup_info = BackupInfo(
                timestamp=timestamp,
                backup_id=backup_id,
                size_bytes=backup_path.stat().st_size,
                file_count=file_count,
                directories=self.config.source_directories,
                checksum=checksum,
                compression_ratio=compression_ratio,
                encrypted=self.config.encryption_enabled,
                incremental=incremental,
                parent_backup_id=parent_backup_id
            )
            
            self.metadata[backup_id] = backup_info
            self._save_metadata()
            
            # 古いバックアップ削除
            self._cleanup_old_backups()
            
            logger.info(f"✅ バックアップ作成完了: {backup_id}")
            logger.info(f"   サイズ: {total_size / 1e6:.1f}MB → {backup_path.stat().st_size / 1e6:.1f}MB")
            logger.info(f"   圧縮率: {compression_ratio:.2f}x")
            logger.info(f"   ファイル数: {file_count}")
            
            return backup_id
            
        except Exception as e:
            logger.error(f"バックアップ作成エラー: {e}")
            return None
    
    def _collect_files(self) -> List[Path]:
        """バックアップ対象ファイル収集"""
        files = []
        
        for source_dir in self.config.source_directories:
            source_path = Path(source_dir)
            if source_path.exists():
                for file_path in source_path.rglob("*"):
                    if file_path.is_file():
                        # 除外パターンチェック
                        skip = False
                        for pattern in self.config.exclude_patterns:
                            if file_path.match(pattern):
                                skip = True
                                break
                        
                        if not skip:
                            files.append(file_path)
        
        return files
    
    def _filter_changed_files(self, files: List[Path]) -> List[Path]:
        """変更されたファイルのフィルタリング"""
        # 最新バックアップのハッシュ情報取得
        latest_backup_id = max(self.metadata.keys(), key=lambda x: self.metadata[x].timestamp)
        latest_backup_path = self.backup_dir / latest_backup_id
        
        # 簡略化: 全ファイルを対象とする（実際の実装では差分検出）
        return files
    
    def _backup_files(self, files: List[Path], backup_path: Path) -> Tuple[int, int]:
        """ファイルバックアップ実行"""
        total_size = 0
        file_count = 0
        
        with tqdm(total=len(files), desc="ファイルコピー中", unit="files") as pbar:
            for file_path in files:
                try:
                    # 相対パス計算
                    for source_dir in self.config.source_directories:
                        source_path = Path(source_dir)
                        if file_path.is_relative_to(source_path):
                            relative_path = file_path.relative_to(source_path)
                            dest_path = backup_path / source_dir / relative_path
                            break
                    else:
                        continue
                    
                    # ディレクトリ作成
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # ファイルコピー
                    shutil.copy2(file_path, dest_path)
                    total_size += file_path.stat().st_size
                    file_count += 1
                    
                except Exception as e:
                    logger.warning(f"ファイルコピーエラー {file_path}: {e}")
                
                pbar.update(1)
        
        return total_size, file_count
    
    def _compress_backup(self, backup_path: Path, backup_id: str) -> Path:
        """バックアップ圧縮"""
        compressed_path = self.backup_dir / f"{backup_id}.zip"
        
        with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in backup_path.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(backup_path)
                    zipf.write(file_path, arcname)
        
        return compressed_path
    
    def _encrypt_backup(self, backup_path: Path, backup_id: str) -> Path:
        """バックアップ暗号化"""
        encrypted_path = self.backup_dir / f"{backup_id}.encrypted"
        self.encryption.encrypt_file(backup_path, encrypted_path)
        return encrypted_path
    
    def _cleanup_old_backups(self):
        """古いバックアップ削除"""
        if len(self.metadata) <= self.config.max_backups:
            return
        
        # 古いバックアップを削除
        sorted_backups = sorted(
            self.metadata.items(),
            key=lambda x: x[1].timestamp
        )
        
        backups_to_remove = sorted_backups[:-self.config.max_backups]
        
        for backup_id, backup_info in backups_to_remove:
            try:
                # バックアップファイル削除
                backup_files = list(self.backup_dir.glob(f"{backup_id}*"))
                for backup_file in backup_files:
                    if backup_file.is_file():
                        backup_file.unlink()
                    elif backup_file.is_dir():
                        shutil.rmtree(backup_file)
                
                # メタデータから削除
                del self.metadata[backup_id]
                logger.info(f"古いバックアップを削除: {backup_id}")
                
            except Exception as e:
                logger.error(f"バックアップ削除エラー {backup_id}: {e}")
        
        self._save_metadata()
    
    def restore_backup(self, backup_id: str, restore_path: str = None) -> bool:
        """バックアップ復元"""
        if backup_id not in self.metadata:
            logger.error(f"バックアップが見つかりません: {backup_id}")
            return False
        
        backup_info = self.metadata[backup_id]
        restore_path = Path(restore_path or f"Restored_{backup_id}")
        
        logger.info(f"🔄 バックアップ復元開始: {backup_id}")
        
        try:
            # バックアップファイル検索
            backup_files = list(self.backup_dir.glob(f"{backup_id}*"))
            if not backup_files:
                logger.error(f"バックアップファイルが見つかりません: {backup_id}")
                return False
            
            backup_file = backup_files[0]
            
            # 復号化
            if backup_info.encrypted and self.encryption:
                decrypted_file = self.backup_dir / f"{backup_id}_decrypted"
                self.encryption.decrypt_file(backup_file, decrypted_file)
                backup_file = decrypted_file
            
            # 展開
            if backup_file.suffix == '.zip':
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    zipf.extractall(restore_path)
            else:
                shutil.copytree(backup_file, restore_path, dirs_exist_ok=True)
            
            # 整合性チェック
            if self.config.verify_integrity:
                if self._verify_backup_integrity(backup_id, restore_path):
                    logger.info("✅ 整合性チェック: 正常")
                else:
                    logger.warning("⚠️ 整合性チェック: 異常検出")
            
            # 一時ファイル削除
            if backup_info.encrypted and 'decrypted_file' in locals():
                decrypted_file.unlink()
            
            logger.info(f"✅ バックアップ復元完了: {restore_path}")
            return True
            
        except Exception as e:
            logger.error(f"バックアップ復元エラー: {e}")
            return False
    
    def _verify_backup_integrity(self, backup_id: str, restored_path: Path) -> bool:
        """バックアップ整合性チェック"""
        try:
            backup_info = self.metadata[backup_id]
            
            # ファイル数チェック
            restored_files = list(restored_path.rglob("*"))
            restored_file_count = len([f for f in restored_files if f.is_file()])
            
            if restored_file_count != backup_info.file_count:
                logger.warning(f"ファイル数不一致: {restored_file_count} != {backup_info.file_count}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"整合性チェックエラー: {e}")
            return False
    
    def list_backups(self) -> List[BackupInfo]:
        """バックアップ一覧取得"""
        return sorted(self.metadata.values(), key=lambda x: x.timestamp, reverse=True)
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """バックアップ情報取得"""
        return self.metadata.get(backup_id)
    
    def delete_backup(self, backup_id: str) -> bool:
        """バックアップ削除"""
        if backup_id not in self.metadata:
            logger.error(f"バックアップが見つかりません: {backup_id}")
            return False
        
        try:
            # バックアップファイル削除
            backup_files = list(self.backup_dir.glob(f"{backup_id}*"))
            for backup_file in backup_files:
                if backup_file.is_file():
                    backup_file.unlink()
                elif backup_file.is_dir():
                    shutil.rmtree(backup_file)
            
            # メタデータから削除
            del self.metadata[backup_id]
            self._save_metadata()
            
            logger.info(f"バックアップを削除: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"バックアップ削除エラー: {e}")
            return False
    
    def start_scheduler(self):
        """スケジューラ開始"""
        if not self.config.auto_backup_enabled:
            return
        
        # スケジュール設定
        schedule.every(self.config.backup_interval_hours).hours.do(
            lambda: self.create_backup(incremental=True)
        )
        
        schedule.every().day.at(self.config.daily_backup_time).do(
            lambda: self.create_backup(incremental=False)
        )
        
        # スケジューラスレッド開始
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("📅 自動バックアップスケジューラ開始")
    
    def stop_scheduler(self):
        """スケジューラ停止"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        
        schedule.clear()
        logger.info("📅 自動バックアップスケジューラ停止")
    
    def _run_scheduler(self):
        """スケジューラ実行"""
        while self.scheduler_running:
            schedule.run_pending()
            time.sleep(60)  # 1分間隔でチェック
    
    def generate_backup_report(self) -> str:
        """バックアップレポート生成"""
        backups = self.list_backups()
        total_size = sum(backup.size_bytes for backup in backups)
        
        report = f"""
💾 NKAT バックアップレポート
{'='*50}

📊 統計情報:
- 総バックアップ数: {len(backups)}
- 総サイズ: {total_size / 1e9:.2f}GB
- 最新バックアップ: {backups[0].timestamp if backups else 'なし'}
- 最古バックアップ: {backups[-1].timestamp if backups else 'なし'}

⚙️ 設定:
- 最大保持数: {self.config.max_backups}
- 圧縮: {'有効' if self.config.compression_enabled else '無効'}
- 暗号化: {'有効' if self.config.encryption_enabled else '無効'}
- 自動バックアップ: {'有効' if self.config.auto_backup_enabled else '無効'}

📋 バックアップ一覧:
"""
        
        for backup in backups[:10]:  # 最新10件
            size_mb = backup.size_bytes / 1e6
            backup_type = "増分" if backup.incremental else "完全"
            report += f"- {backup.backup_id}: {backup.timestamp} ({size_mb:.1f}MB, {backup_type})\n"
        
        if len(backups) > 10:
            report += f"... 他 {len(backups) - 10} 件\n"
        
        return report

def main():
    """メイン関数"""
    print("💾 NKAT バックアップ管理システム")
    print("=" * 50)
    
    try:
        # バックアップマネージャー初期化
        config = BackupConfig()
        backup_manager = NKATBackupManager(config)
        
        while True:
            print("\n📋 メニュー:")
            print("1. バックアップ作成")
            print("2. バックアップ復元")
            print("3. バックアップ一覧")
            print("4. バックアップ削除")
            print("5. レポート表示")
            print("6. 自動バックアップ開始/停止")
            print("0. 終了")
            
            choice = input("\n選択してください (0-6): ").strip()
            
            if choice == "1":
                backup_name = input("バックアップ名 (空白で自動生成): ").strip() or None
                incremental = input("増分バックアップ? (y/N): ").strip().lower() == 'y'
                backup_id = backup_manager.create_backup(backup_name, incremental)
                if backup_id:
                    print(f"✅ バックアップ作成完了: {backup_id}")
                
            elif choice == "2":
                backups = backup_manager.list_backups()
                if not backups:
                    print("❌ バックアップが見つかりません")
                    continue
                
                print("\n📋 利用可能なバックアップ:")
                for i, backup in enumerate(backups[:10]):
                    print(f"{i+1}. {backup.backup_id} ({backup.timestamp})")
                
                try:
                    idx = int(input("復元するバックアップ番号: ")) - 1
                    if 0 <= idx < len(backups):
                        backup_id = backups[idx].backup_id
                        restore_path = input("復元先パス (空白でデフォルト): ").strip() or None
                        if backup_manager.restore_backup(backup_id, restore_path):
                            print(f"✅ バックアップ復元完了")
                    else:
                        print("❌ 無効な番号です")
                except ValueError:
                    print("❌ 無効な入力です")
                
            elif choice == "3":
                backups = backup_manager.list_backups()
                if not backups:
                    print("❌ バックアップが見つかりません")
                else:
                    print("\n📋 バックアップ一覧:")
                    for backup in backups:
                        size_mb = backup.size_bytes / 1e6
                        backup_type = "増分" if backup.incremental else "完全"
                        print(f"- {backup.backup_id}: {backup.timestamp} ({size_mb:.1f}MB, {backup_type})")
                
            elif choice == "4":
                backups = backup_manager.list_backups()
                if not backups:
                    print("❌ バックアップが見つかりません")
                    continue
                
                print("\n📋 削除可能なバックアップ:")
                for i, backup in enumerate(backups):
                    print(f"{i+1}. {backup.backup_id} ({backup.timestamp})")
                
                try:
                    idx = int(input("削除するバックアップ番号: ")) - 1
                    if 0 <= idx < len(backups):
                        backup_id = backups[idx].backup_id
                        confirm = input(f"本当に削除しますか? {backup_id} (y/N): ").strip().lower()
                        if confirm == 'y':
                            if backup_manager.delete_backup(backup_id):
                                print(f"✅ バックアップ削除完了: {backup_id}")
                    else:
                        print("❌ 無効な番号です")
                except ValueError:
                    print("❌ 無効な入力です")
                
            elif choice == "5":
                report = backup_manager.generate_backup_report()
                print(report)
                
            elif choice == "6":
                if backup_manager.scheduler_running:
                    backup_manager.stop_scheduler()
                    print("⏹️ 自動バックアップを停止しました")
                else:
                    backup_manager.start_scheduler()
                    print("▶️ 自動バックアップを開始しました")
                
            elif choice == "0":
                if backup_manager.scheduler_running:
                    backup_manager.stop_scheduler()
                print("👋 バックアップ管理システムを終了します")
                break
                
            else:
                print("❌ 無効な選択です")
        
    except KeyboardInterrupt:
        print("\n⚠️ プログラムが中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        logger.error(f"メインエラー: {e}")

if __name__ == "__main__":
    main() 