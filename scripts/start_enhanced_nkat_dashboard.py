#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Enhanced NKAT Dashboard Launcher
改良版NKATダッシュボード起動スクリプト

機能:
- 依存関係自動チェック・インストール
- GPU環境検証
- ポート管理
- 自動ブラウザ起動
- エラーハンドリング
"""

import os
import sys
import subprocess
import time
import webbrowser
import socket
from pathlib import Path
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDashboardLauncher:
    """改良版ダッシュボード起動クラス"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.src_dir = self.project_root / "src"
        self.dashboard_file = self.src_dir / "enhanced_nkat_dashboard.py"
        self.requirements_file = self.project_root / "requirements.txt"
        self.default_port = 8503  # 新しいポート
        
    def check_python_version(self):
        """Python バージョンチェック"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error("Python 3.8以上が必要です")
            return False
        
        logger.info(f"Python {version.major}.{version.minor}.{version.micro} 確認")
        return True
    
    def check_dependencies(self):
        """依存関係チェック"""
        logger.info("依存関係をチェック中...")
        
        required_packages = [
            'streamlit',
            'numpy',
            'pandas',
            'plotly',
            'psutil'
        ]
        
        optional_packages = [
            'torch',
            'GPUtil',
            'matplotlib',
            'scipy'
        ]
        
        missing_required = []
        missing_optional = []
        
        # 必須パッケージチェック
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} 利用可能")
            except ImportError:
                missing_required.append(package)
                logger.warning(f"❌ {package} 未インストール")
        
        # オプションパッケージチェック
        for package in optional_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} 利用可能")
            except ImportError:
                missing_optional.append(package)
                logger.info(f"ℹ️ {package} 未インストール（オプション）")
        
        return missing_required, missing_optional
    
    def install_dependencies(self, packages):
        """依存関係インストール"""
        if not packages:
            return True
        
        logger.info(f"パッケージをインストール中: {', '.join(packages)}")
        
        try:
            # pip upgrade
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # パッケージインストール
            for package in packages:
                logger.info(f"インストール中: {package}")
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                logger.info(f"✅ {package} インストール完了")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"インストールエラー: {e}")
            return False
    
    def check_gpu_environment(self):
        """GPU環境チェック"""
        logger.info("GPU環境をチェック中...")
        
        gpu_info = {
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_names': [],
            'pytorch_cuda': False
        }
        
        # CUDA チェック
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['gpu_count'] = torch.cuda.device_count()
                gpu_info['pytorch_cuda'] = True
                
                for i in range(gpu_info['gpu_count']):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_info['gpu_names'].append(gpu_name)
                    logger.info(f"✅ GPU {i}: {gpu_name}")
            else:
                logger.info("ℹ️ CUDA利用不可（CPU版PyTorch）")
        except ImportError:
            logger.info("ℹ️ PyTorch未インストール")
        
        # GPUtil チェック
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                logger.info(f"✅ GPUtil検出: {len(gpus)}個のGPU")
                for gpu in gpus:
                    logger.info(f"  - {gpu.name}: {gpu.memoryTotal}MB")
            else:
                logger.info("ℹ️ GPUtil: GPU未検出")
        except ImportError:
            logger.info("ℹ️ GPUtil未インストール")
        
        return gpu_info
    
    def find_available_port(self, start_port=8503, max_attempts=10):
        """利用可能ポート検索"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    logger.info(f"✅ ポート {port} 利用可能")
                    return port
            except OSError:
                logger.info(f"ℹ️ ポート {port} 使用中")
                continue
        
        logger.error("利用可能なポートが見つかりません")
        return None
    
    def kill_existing_streamlit(self):
        """既存のStreamlitプロセス終了"""
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/f', '/im', 'streamlit.exe'], 
                             capture_output=True)
                subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                             capture_output=True)
            else:  # Unix/Linux
                subprocess.run(['pkill', '-f', 'streamlit'], capture_output=True)
            
            logger.info("既存のStreamlitプロセスを終了しました")
            time.sleep(2)  # プロセス終了待機
        except Exception as e:
            logger.warning(f"プロセス終了エラー: {e}")
    
    def create_startup_script(self, port):
        """起動スクリプト作成"""
        script_content = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys
import os

# プロジェクトルートに移動
os.chdir(r"{self.project_root}")

# Streamlit起動
cmd = [
    sys.executable, "-m", "streamlit", "run",
    r"{self.dashboard_file}",
    "--server.port", "{port}",
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false"
]

subprocess.run(cmd)
"""
        
        script_file = self.project_root / "start_enhanced_dashboard.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return script_file
    
    def launch_dashboard(self, port):
        """ダッシュボード起動"""
        logger.info(f"Enhanced NKATダッシュボードを起動中... (ポート: {port})")
        
        # 環境変数設定
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.src_dir)
        
        # Streamlit起動コマンド
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(self.dashboard_file),
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--server.maxUploadSize", "200"
        ]
        
        try:
            # バックグラウンドで起動
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 起動待機
            logger.info("ダッシュボード起動中...")
            time.sleep(5)
            
            # ブラウザ起動
            url = f"http://localhost:{port}"
            logger.info(f"ブラウザを起動: {url}")
            webbrowser.open(url)
            
            return process
            
        except Exception as e:
            logger.error(f"起動エラー: {e}")
            return None
    
    def create_batch_file(self, port):
        """Windows用バッチファイル作成"""
        batch_content = f"""@echo off
chcp 65001 > nul
echo 🌌 Enhanced NKAT Dashboard Launcher
echo =====================================
echo.

cd /d "{self.project_root}"

echo 📦 依存関係チェック中...
py -3 -m pip install --upgrade pip > nul 2>&1
py -3 -m pip install streamlit numpy pandas plotly psutil > nul 2>&1

echo 🚀 ダッシュボード起動中...
py -3 -m streamlit run "{self.dashboard_file}" --server.port {port} --server.headless true

pause
"""
        
        batch_file = self.project_root / "start_enhanced_nkat_dashboard.bat"
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        logger.info(f"バッチファイル作成: {batch_file}")
        return batch_file
    
    def run(self):
        """メイン実行"""
        print("🌌 Enhanced NKAT Dashboard Launcher")
        print("=" * 50)
        
        # Python バージョンチェック
        if not self.check_python_version():
            input("Enterキーを押して終了...")
            return False
        
        # ダッシュボードファイル存在チェック
        if not self.dashboard_file.exists():
            logger.error(f"ダッシュボードファイルが見つかりません: {self.dashboard_file}")
            input("Enterキーを押して終了...")
            return False
        
        # 依存関係チェック
        missing_required, missing_optional = self.check_dependencies()
        
        # 必須パッケージインストール
        if missing_required:
            logger.info("必須パッケージをインストールします...")
            if not self.install_dependencies(missing_required):
                logger.error("必須パッケージのインストールに失敗しました")
                input("Enterキーを押して終了...")
                return False
        
        # オプションパッケージインストール（ユーザー選択）
        if missing_optional:
            print(f"\nオプションパッケージ: {', '.join(missing_optional)}")
            choice = input("オプションパッケージをインストールしますか？ (y/N): ").lower()
            if choice in ['y', 'yes']:
                self.install_dependencies(missing_optional)
        
        # GPU環境チェック
        gpu_info = self.check_gpu_environment()
        
        # 既存プロセス終了
        self.kill_existing_streamlit()
        
        # 利用可能ポート検索
        port = self.find_available_port(self.default_port)
        if not port:
            logger.error("利用可能なポートが見つかりません")
            input("Enterキーを押して終了...")
            return False
        
        # バッチファイル作成
        self.create_batch_file(port)
        
        # ダッシュボード起動
        process = self.launch_dashboard(port)
        if not process:
            logger.error("ダッシュボードの起動に失敗しました")
            input("Enterキーを押して終了...")
            return False
        
        print(f"\n✅ Enhanced NKATダッシュボードが起動しました!")
        print(f"🌐 URL: http://localhost:{port}")
        print(f"🎮 GPU環境: {'✅ 利用可能' if gpu_info['cuda_available'] else '❌ 利用不可'}")
        print("\n📝 使用方法:")
        print("1. ブラウザでダッシュボードにアクセス")
        print("2. サイドバーでパラメータを設定")
        print("3. '解析開始'ボタンで解析実行")
        print("4. リアルタイム監視で状態確認")
        print("\n⏹️ 終了するには Ctrl+C を押してください")
        
        try:
            # プロセス監視
            while True:
                if process.poll() is not None:
                    logger.warning("ダッシュボードプロセスが終了しました")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("ユーザーによる終了要求")
            process.terminate()
            process.wait()
        
        return True

def main():
    """メイン関数"""
    launcher = EnhancedDashboardLauncher()
    success = launcher.run()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 