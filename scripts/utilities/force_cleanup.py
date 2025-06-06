#!/usr/bin/env python3
"""
🛠️ 緊急クリーンアップスクリプト - Force Cleanup Recovery Data
無限階層のrecovery_dataディレクトリを強制削除します
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
import sys

def force_delete_recovery_data():
    """
    recovery_dataディレクトリを強制削除
    """
    print("🚀 緊急クリーンアップ開始: recovery_data削除")
    
    recovery_path = Path("recovery_data")
    
    if not recovery_path.exists():
        print("✅ recovery_dataディレクトリは既に存在しません")
        return True
    
    try:
        # 方法1: 通常の削除を試行
        print("📁 通常削除を試行中...")
        shutil.rmtree(recovery_path, ignore_errors=True)
        
        if not recovery_path.exists():
            print("✅ 通常削除に成功！")
            return True
            
    except Exception as e:
        print(f"⚠️ 通常削除失敗: {e}")
    
    try:
        # 方法2: 一時ディレクトリを使用した置換削除
        print("🔄 一時ディレクトリによる置換削除を試行中...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_empty = Path(temp_dir) / "empty"
            temp_empty.mkdir()
            
            # robocopyスタイルの置換
            if os.name == 'nt':  # Windows
                import subprocess
                result = subprocess.run([
                    'robocopy', str(temp_empty), str(recovery_path), 
                    '/MIR', '/NFL', '/NDL', '/NJH', '/NJS', '/NC', '/NS', '/NP'
                ], capture_output=True, text=True)
                print(f"robocopy結果: {result.returncode}")
            
            # 削除再試行
            time.sleep(1)
            shutil.rmtree(recovery_path, ignore_errors=True)
            
            if not recovery_path.exists():
                print("✅ 置換削除に成功！")
                return True
                
    except Exception as e:
        print(f"⚠️ 置換削除失敗: {e}")
    
    try:
        # 方法3: 段階的削除（深い階層から削除）
        print("🗂️ 段階的削除を試行中...")
        
        def delete_deep_first(path, max_depth=10):
            """深い階層から削除"""
            if max_depth <= 0:
                return
                
            try:
                for item in path.iterdir():
                    if item.is_dir():
                        delete_deep_first(item, max_depth - 1)
                        try:
                            item.rmdir()
                        except:
                            pass
                    else:
                        try:
                            item.unlink()
                        except:
                            pass
            except:
                pass
        
        delete_deep_first(recovery_path)
        
        # 最終削除試行
        shutil.rmtree(recovery_path, ignore_errors=True)
        
        if not recovery_path.exists():
            print("✅ 段階的削除に成功！")
            return True
            
    except Exception as e:
        print(f"⚠️ 段階的削除失敗: {e}")
    
    # 方法4: OSレベルの強制削除
    try:
        print("💥 OSレベル強制削除を試行中...")
        
        if os.name == 'nt':  # Windows
            # Windowsの場合
            os.system(f'rmdir /s /q "{recovery_path}" 2>nul')
            os.system(f'rd /s /q "{recovery_path}" 2>nul')
        else:
            # Unix系の場合
            os.system(f'rm -rf "{recovery_path}" 2>/dev/null')
        
        time.sleep(2)
        
        if not recovery_path.exists():
            print("✅ OSレベル強制削除に成功！")
            return True
            
    except Exception as e:
        print(f"⚠️ OSレベル削除失敗: {e}")
    
    print("❌ すべての削除方法が失敗しました")
    return False

def main():
    """メイン処理"""
    print("=" * 60)
    print("🛡️ NKAT統合特解理論プロジェクト - 緊急クリーンアップ")
    print("=" * 60)
    
    # 現在のディレクトリ確認
    current_dir = Path.cwd()
    print(f"📍 現在のディレクトリ: {current_dir}")
    
    # recovery_dataの状況確認
    recovery_path = Path("recovery_data")
    if recovery_path.exists():
        print(f"⚠️ recovery_dataディレクトリが存在します")
        try:
            # サイズ情報取得（可能な範囲で）
            file_count = len(list(recovery_path.rglob("*")))
            print(f"📊 ファイル/フォルダ数: {file_count} (概算)")
        except:
            print("📊 ファイル数: 計測不可能（深すぎる階層）")
    else:
        print("✅ recovery_dataディレクトリは存在しません")
        return
    
    # 削除実行
    success = force_delete_recovery_data()
    
    if success:
        print("\n🎉 クリーンアップ完了！")
        print("✨ プロジェクトの整理が完了しました")
    else:
        print("\n❌ クリーンアップ失敗")
        print("🔧 手動での処理が必要かもしれません")
    
    # 最終確認
    if not Path("recovery_data").exists():
        print("🔍 最終確認: recovery_dataディレクトリは削除されています")
    else:
        print("⚠️ 最終確認: recovery_dataディレクトリがまだ存在します")

if __name__ == "__main__":
    main() 