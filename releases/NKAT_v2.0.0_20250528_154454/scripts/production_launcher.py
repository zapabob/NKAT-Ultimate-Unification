#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT リーマン予想解析システム - 本番用ランチャー
Production Launcher for NKAT Riemann Hypothesis Analysis System

本番環境での安全で安定した起動を保証する包括的なランチャー

Author: NKAT Research Team
Date: 2025-05-28
Version: 2.0.0 - Production Release
License: MIT
"""

import sys
import os
import json
import logging
import argparse
import signal
import time
import subprocess
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import warnings

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_production_logging(log_level: str = "INFO") -> logging.Logger:
    """本番用ログ設定"""
    # ログディレクトリ作成
    log_dir = project_root / "logs" / "production"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # ログファイル名（日付付き）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"nkat_production_{timestamp}.log"
    
    # ロガー設定
    logger = logging.getLogger('NKAT_Production')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # ファイルハンドラー（詳細ログ）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # コンソールハンドラー（簡潔ログ）
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"本番ログ開始: {log_file}")
    return logger

def load_production_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """本番設定ファイル読み込み"""
    if config_path is None:
        config_path = project_root / "config" / "production_config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"設定ファイルの形式が正しくありません: {e}")

def check_system_requirements(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """システム要件チェック"""
    logger.info("システム要件チェック開始")
    
    requirements_met = True
    
    # Python バージョンチェック
    python_version = sys.version_info
    if python_version < (3, 8):
        logger.error(f"Python 3.8以上が必要です。現在: {python_version}")
        requirements_met = False
    else:
        logger.info(f"Python バージョン: {python_version}")
    
    # GPU チェック
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name}, VRAM: {total_memory:.1f} GB")
            
            # RTX3080 推奨チェック
            if "RTX 3080" not in gpu_name:
                logger.warning("RTX 3080以外のGPUです。性能が制限される可能性があります。")
            
            # VRAM チェック
            required_vram = config.get('gpu_optimization', {}).get('gpu_memory_limit_gb', 9.0)
            if total_memory < required_vram:
                logger.warning(f"VRAM不足の可能性: 必要{required_vram}GB, 利用可能{total_memory:.1f}GB")
        else:
            logger.error("CUDA対応GPUが見つかりません")
            requirements_met = False
    except ImportError:
        logger.error("PyTorchがインストールされていません")
        requirements_met = False
    
    # メモリチェック
    memory = psutil.virtual_memory()
    memory_gb = memory.total / 1e9
    required_memory = config.get('security', {}).get('memory_limit_gb', 30)
    
    if memory_gb < required_memory:
        logger.warning(f"メモリ不足の可能性: 必要{required_memory}GB, 利用可能{memory_gb:.1f}GB")
    else:
        logger.info(f"システムメモリ: {memory_gb:.1f} GB")
    
    # ディスク容量チェック
    disk_usage = psutil.disk_usage(str(project_root))  # WindowsPathを文字列に変換
    free_space_gb = disk_usage.free / 1e9
    if free_space_gb < 10:  # 最低10GB必要
        logger.warning(f"ディスク容量不足: 残り{free_space_gb:.1f}GB")
    
    return requirements_met

def check_dependencies(logger: logging.Logger) -> bool:
    """依存関係チェック"""
    logger.info("依存関係チェック開始")
    
    required_packages = [
        'torch', 'numpy', 'scipy', 'matplotlib', 'plotly', 
        'streamlit', 'h5py', 'tqdm', 'psutil', 'GPUtil', 'mpmath'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.debug(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} が見つかりません")
    
    if missing_packages:
        logger.error(f"不足パッケージ: {', '.join(missing_packages)}")
        logger.info("pip install -r requirements.txt を実行してください")
        return False
    
    logger.info("すべての依存関係が満たされています")
    return True

def setup_signal_handlers(logger: logging.Logger):
    """シグナルハンドラー設定"""
    def signal_handler(signum, frame):
        logger.info(f"シグナル {signum} を受信。安全にシャットダウンします...")
        # クリーンアップ処理
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def create_production_directories(logger: logging.Logger):
    """本番用ディレクトリ作成"""
    directories = [
        "logs/production",
        "results/production",
        "results/production/checkpoints",
        "results/production/analysis",
        "results/production/reports",
        "results/production/exports"
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ディレクトリ作成: {full_path}")

def validate_config(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """設定ファイル検証"""
    logger.info("設定ファイル検証開始")
    
    required_sections = [
        'system_info', 'nkat_parameters', 'gpu_optimization',
        'checkpoint_settings', 'monitoring', 'numerical_settings'
    ]
    
    for section in required_sections:
        if section not in config:
            logger.error(f"必須設定セクションが見つかりません: {section}")
            return False
    
    # 数値範囲チェック
    nkat_params = config.get('nkat_parameters', {})
    if nkat_params.get('nkat_dimension', 0) < 8:
        logger.error("nkat_dimension は8以上である必要があります")
        return False
    
    if nkat_params.get('nkat_precision', 0) < 50:
        logger.error("nkat_precision は50以上である必要があります")
        return False
    
    logger.info("設定ファイル検証完了")
    return True

def launch_dashboard(config: Dict[str, Any], logger: logging.Logger) -> subprocess.Popen:
    """ダッシュボード起動"""
    logger.info("Streamlitダッシュボード起動中...")
    
    dashboard_config = config.get('dashboard', {})
    port = dashboard_config.get('port', 8501)
    host = dashboard_config.get('host', 'localhost')
    
    # Streamlit起動コマンド
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        str(project_root / 'src' / 'nkat_riemann_ultimate_precision_system.py'),
        '--server.port', str(port),
        '--server.address', host,
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root,
            env=os.environ.copy()
        )
        
        # 起動確認
        time.sleep(3)
        if process.poll() is None:
            logger.info(f"ダッシュボード起動成功: http://{host}:{port}")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"ダッシュボード起動失敗: {stderr.decode()}")
            return None
    except Exception as e:
        logger.error(f"ダッシュボード起動エラー: {e}")
        return None

def monitor_system_health(config: Dict[str, Any], logger: logging.Logger):
    """システムヘルス監視"""
    monitoring_config = config.get('monitoring', {})
    alert_thresholds = monitoring_config.get('alert_thresholds', {})
    
    while True:
        try:
            # GPU温度チェック
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    if gpu.temperature > alert_thresholds.get('gpu_temperature_celsius', 85):
                        logger.warning(f"GPU温度警告: {gpu.temperature}°C")
            except:
                pass
            
            # メモリ使用量チェック
            memory = psutil.virtual_memory()
            if memory.percent > alert_thresholds.get('memory_usage_percent', 90):
                logger.warning(f"メモリ使用量警告: {memory.percent}%")
            
            time.sleep(monitoring_config.get('monitoring_interval_seconds', 30))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"ヘルス監視エラー: {e}")
            time.sleep(10)

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='NKAT リーマン予想解析システム - 本番ランチャー')
    parser.add_argument('--config', type=str, help='設定ファイルパス')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='ログレベル')
    parser.add_argument('--no-dashboard', action='store_true', help='ダッシュボードを起動しない')
    parser.add_argument('--check-only', action='store_true', help='システムチェックのみ実行')
    
    args = parser.parse_args()
    
    # ログ設定
    logger = setup_production_logging(args.log_level)
    
    try:
        logger.info("🚀 NKAT リーマン予想解析システム - 本番起動開始")
        logger.info("=" * 60)
        
        # 設定ファイル読み込み
        config = load_production_config(args.config)
        logger.info(f"設定ファイル読み込み完了: {config['system_info']['version']}")
        
        # 設定検証
        if not validate_config(config, logger):
            logger.error("設定ファイル検証失敗")
            sys.exit(1)
        
        # システム要件チェック
        if not check_system_requirements(config, logger):
            logger.error("システム要件を満たしていません")
            sys.exit(1)
        
        # 依存関係チェック
        if not check_dependencies(logger):
            logger.error("依存関係チェック失敗")
            sys.exit(1)
        
        if args.check_only:
            logger.info("✅ システムチェック完了")
            return
        
        # 本番用ディレクトリ作成
        create_production_directories(logger)
        
        # シグナルハンドラー設定
        setup_signal_handlers(logger)
        
        # ダッシュボード起動
        dashboard_process = None
        if not args.no_dashboard:
            dashboard_process = launch_dashboard(config, logger)
            if dashboard_process is None:
                logger.error("ダッシュボード起動失敗")
                sys.exit(1)
        
        logger.info("🌌 NKAT システム本番稼働開始")
        logger.info("Ctrl+C で安全にシャットダウンします")
        
        # システムヘルス監視開始
        try:
            monitor_system_health(config, logger)
        except KeyboardInterrupt:
            logger.info("シャットダウン要求を受信")
        
        # クリーンアップ
        if dashboard_process:
            logger.info("ダッシュボードを終了中...")
            dashboard_process.terminate()
            dashboard_process.wait(timeout=10)
        
        logger.info("🛑 NKAT システム正常終了")
        
    except Exception as e:
        logger.error(f"予期しないエラー: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 