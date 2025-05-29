#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT理論 クイック復旧スクリプト
電源復旧後の即座実行用
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.auto_recovery_startup import AutoRecoverySystem

def main():
    print("🚀 NKAT理論 クイック復旧開始")
    
    recovery_system = AutoRecoverySystem()
    recovery_system.startup_delay = 5  # 短縮待機時間
    
    success = recovery_system.run_auto_recovery()
    
    if success:
        print("✅ 復旧完了！")
        print("📊 ダッシュボード: http://localhost:8501")
    else:
        print("❌ 復旧失敗")
        print("手動で確認してください")

if __name__ == "__main__":
    main()
