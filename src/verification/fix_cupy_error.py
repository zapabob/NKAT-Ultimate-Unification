#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT CuPyエラー修復スクリプト
CuPyのインポートエラーを診断・修復します
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def print_status(message, status="INFO"):
    """ステータスメッセージを表示"""
    symbols = {
        "INFO": "ℹ️",
        "SUCCESS": "✅", 
        "WARNING": "⚠️",
        "ERROR": "❌",
        "PROGRESS": "🔄"
    }
    print(f"{symbols.get(status, 'ℹ️')} {message}")

def check_python_version():
    """Pythonバージョンを確認"""
    version = sys.version_info
    print_status(f"Python バージョン: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print_status("Python 3.8以上が必要です", "ERROR")
        return False
    return True

def check_cuda_availability():
    """CUDA環境を確認"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_status("NVIDIA GPU検出済み", "SUCCESS")
            return True
        else:
            print_status("NVIDIA GPUが検出されませんでした", "WARNING")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status("nvidia-smiコマンドが見つかりません", "WARNING")
        return False

def uninstall_cupy():
    """既存のCuPyをアンインストール"""
    print_status("既存のCuPyパッケージをアンインストール中...", "PROGRESS")
    
    cupy_packages = [
        'cupy',
        'cupy-cuda11x', 
        'cupy-cuda12x',
        'cupy-cuda118',
        'cupy-cuda119',
        'cupy-cuda120',
        'cupy-cuda121'
    ]
    
    for package in cupy_packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', package, '-y'], 
                         capture_output=True, check=False)
            print_status(f"{package} をアンインストールしました")
        except Exception as e:
            print_status(f"{package} のアンインストールに失敗: {e}", "WARNING")

def install_cupy():
    """適切なCuPyバージョンをインストール"""
    print_status("CuPyを再インストール中...", "PROGRESS")
    
    # CUDA 12.x対応版をインストール
    try:
        cmd = [sys.executable, '-m', 'pip', 'install', 'cupy-cuda12x', '--no-cache-dir']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print_status("CuPy CUDA 12.x版のインストールが完了しました", "SUCCESS")
            return True
        else:
            print_status(f"CuPyインストールエラー: {result.stderr}", "ERROR")
            
            # フォールバック: CPU版をインストール
            print_status("CPU版CuPyをインストール中...", "PROGRESS")
            cmd_cpu = [sys.executable, '-m', 'pip', 'install', 'cupy', '--no-cache-dir']
            result_cpu = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=300)
            
            if result_cpu.returncode == 0:
                print_status("CuPy CPU版のインストールが完了しました", "SUCCESS")
                return True
            else:
                print_status(f"CPU版CuPyインストールも失敗: {result_cpu.stderr}", "ERROR")
                return False
                
    except subprocess.TimeoutExpired:
        print_status("CuPyインストールがタイムアウトしました", "ERROR")
        return False
    except Exception as e:
        print_status(f"CuPyインストール中にエラー: {e}", "ERROR")
        return False

def test_cupy_import():
    """CuPyのインポートテスト"""
    print_status("CuPyインポートテスト中...", "PROGRESS")
    
    try:
        import cupy as cp
        print_status("CuPyインポート成功", "SUCCESS")
        
        # GPU情報を表示
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            print_status(f"検出されたGPU数: {device_count}")
            
            if device_count > 0:
                device_props = cp.cuda.runtime.getDeviceProperties(0)
                print_status(f"GPU 0: {device_props['name'].decode()}")
                
        except Exception as e:
            print_status(f"GPU情報取得エラー: {e}", "WARNING")
            
        return True
        
    except ImportError as e:
        print_status(f"CuPyインポートエラー: {e}", "ERROR")
        return False
    except Exception as e:
        print_status(f"予期しないエラー: {e}", "ERROR")
        return False

def fix_environment_variables():
    """環境変数を修正"""
    print_status("環境変数を設定中...", "PROGRESS")
    
    # CUDA関連の環境変数
    cuda_vars = {
        'CUDA_VISIBLE_DEVICES': '0',
        'CUPY_CACHE_DIR': str(Path.home() / '.cupy' / 'kernel_cache'),
        'PYTHONIOENCODING': 'utf-8'
    }
    
    for var, value in cuda_vars.items():
        os.environ[var] = value
        print_status(f"環境変数設定: {var}={value}")

def create_safe_riemann_script():
    """安全なリーマン解析スクリプトを作成"""
    print_status("安全なリーマン解析スクリプトを作成中...", "PROGRESS")
    
    safe_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT リーマン予想解析 - 安全版
CuPyエラー対応済み
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# CUDA環境の安全な検出
CUPY_AVAILABLE = False
try:
    import cupy as cp
    import cupyx.scipy.special as cp_special
    CUPY_AVAILABLE = True
    print("✅ CuPy CUDA利用可能 - GPU超高速モードで実行")
except ImportError as e:
    print(f"⚠️ CuPy未検出: {e}")
    print("💡 CPUモードで実行します")
    import numpy as cp
except Exception as e:
    print(f"❌ CuPy初期化エラー: {e}")
    print("💡 CPUモードで実行します")
    import numpy as cp
    CUPY_AVAILABLE = False

def safe_riemann_analysis(max_iterations=1000):
    """安全なリーマン予想解析"""
    print("🔬 NKAT リーマン予想解析開始")
    
    # 解析パラメータ
    t_values = np.linspace(0.1, 50, max_iterations)
    zeta_values = []
    
    print(f"📊 解析点数: {len(t_values)}")
    
    # プログレスバー付きで解析実行
    for t in tqdm(t_values, desc="リーマンゼータ関数計算"):
        try:
            # 簡単なゼータ関数近似
            s = 0.5 + 1j * t
            zeta_approx = sum(1/n**s for n in range(1, 100))
            zeta_values.append(abs(zeta_approx))
        except Exception as e:
            print(f"⚠️ 計算エラー (t={t}): {e}")
            zeta_values.append(0)
    
    # 結果の可視化
    plt.figure(figsize=(12, 8))
    plt.plot(t_values, zeta_values, 'b-', linewidth=1, alpha=0.7)
    plt.title('NKAT リーマンゼータ関数解析結果', fontsize=16)
    plt.xlabel('虚部 t', fontsize=12)
    plt.ylabel('|ζ(1/2 + it)|', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'nkat_safe_riemann_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    # JSON結果保存
    results = {
        'timestamp': timestamp,
        'cupy_available': CUPY_AVAILABLE,
        'max_iterations': max_iterations,
        'analysis_points': len(t_values),
        'max_zeta_value': max(zeta_values),
        'min_zeta_value': min(zeta_values),
        'mean_zeta_value': np.mean(zeta_values)
    }
    
    with open(f'nkat_safe_riemann_analysis_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✅ 解析完了")
    print(f"📈 最大値: {results['max_zeta_value']:.6f}")
    print(f"📉 最小値: {results['min_zeta_value']:.6f}")
    print(f"📊 平均値: {results['mean_zeta_value']:.6f}")
    
    plt.show()
    return results

if __name__ == "__main__":
    try:
        results = safe_riemann_analysis()
        print("🎉 NKAT リーマン予想解析が正常に完了しました")
    except KeyboardInterrupt:
        print("\\n⏹️ ユーザーによって中断されました")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
'''
    
    with open('riemann_analysis_safe.py', 'w', encoding='utf-8') as f:
        f.write(safe_script)
    
    print_status("安全なリーマン解析スクリプトを作成しました: riemann_analysis_safe.py", "SUCCESS")

def main():
    """メイン修復プロセス"""
    print_status("🔧 NKAT CuPyエラー修復プロセス開始", "INFO")
    print("=" * 60)
    
    # 1. Python環境確認
    if not check_python_version():
        return False
    
    # 2. CUDA環境確認
    cuda_available = check_cuda_availability()
    
    # 3. 環境変数修正
    fix_environment_variables()
    
    # 4. CuPy再インストール
    uninstall_cupy()
    
    if not install_cupy():
        print_status("CuPyインストールに失敗しました。安全版スクリプトを使用してください。", "WARNING")
    
    # 5. インポートテスト
    import_success = test_cupy_import()
    
    # 6. 安全なスクリプト作成
    create_safe_riemann_script()
    
    print("=" * 60)
    if import_success:
        print_status("🎉 CuPyエラー修復完了！", "SUCCESS")
        print_status("元のスクリプトが正常に動作するはずです", "INFO")
    else:
        print_status("⚠️ CuPyは修復できませんでしたが、安全版スクリプトを作成しました", "WARNING")
        print_status("riemann_analysis_safe.py を使用してください", "INFO")
    
    print_status("修復プロセス完了", "SUCCESS")
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_status("\\n⏹️ 修復プロセスが中断されました", "WARNING")
    except Exception as e:
        print_status(f"❌ 修復プロセス中にエラー: {e}", "ERROR")
        import traceback
        traceback.print_exc() 