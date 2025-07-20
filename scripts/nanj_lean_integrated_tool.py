#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
なんJ風Lean統合開発ツール
エラー分析、自動修正、段階的開発を統合
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 自作モジュールのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lean_error_analyzer import NanjLeanErrorAnalyzer
from lean_auto_fixer import NanjLeanAutoFixer

class NanjLeanIntegratedTool:
    """なんJ風Lean統合開発ツール"""
    
    def __init__(self, lean_file_path: str):
        self.lean_file_path = lean_file_path
        self.analyzer = NanjLeanErrorAnalyzer(lean_file_path)
        self.fixer = NanjLeanAutoFixer(lean_file_path)
        self.development_log = []
        
    def run_complete_analysis(self):
        """完全な分析を実行"""
        print("🔍 なんJ風Lean完全分析開始...")
        print("Don't hold back. Give it your all deep think!!")
        
        # 1. エラー分析
        errors = self.analyzer.analyze_lean_file()
        if not errors:
            print("✅ エラーは検出されませんでした！")
            return
        
        print(f"🔍 検出されたエラー: {len(errors)}種類")
        for error_type, matches in errors.items():
            print(f"  - {error_type}: {len(matches)}件")
        
        # 2. 仮説生成
        hypotheses = self.analyzer.generate_hypotheses(errors)
        print(f"💡 生成された仮説: {len(hypotheses)}件")
        
        # 3. 修正提案生成
        suggestions = self.analyzer.generate_fix_suggestions(hypotheses)
        print(f"🔧 修正提案: {len(suggestions)}件")
        
        # 4. レポート保存
        report_path = self.analyzer.save_analysis_report(errors, hypotheses, suggestions)
        
        self.development_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'complete_analysis',
            'errors_found': len(errors),
            'hypotheses_generated': len(hypotheses),
            'report_path': report_path
        })
        
        return errors, hypotheses, suggestions
    
    def run_auto_fix(self):
        """自動修正を実行"""
        print("🔧 なんJ風Lean自動修正開始...")
        
        # 自動修正の実行
        self.fixer.auto_fix()
        
        self.development_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'auto_fix',
            'fixes_applied': len(self.fixer.fixes_applied)
        })
        
        return self.fixer.fixes_applied
    
    def run_lean_compilation_check(self):
        """Leanコンパイルチェックを実行"""
        print("🔍 Leanコンパイルチェック開始...")
        
        try:
            # Leanコンパイルの実行（--checkオプションを削除）
            result = subprocess.run(
                ['lean', self.lean_file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ Leanコンパイル成功！")
                compilation_status = 'success'
            else:
                print("❌ Leanコンパイルエラー:")
                print(result.stderr)
                compilation_status = 'error'
            
            self.development_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'lean_compilation_check',
                'status': compilation_status,
                'stdout': result.stdout,
                'stderr': result.stderr
            })
            
            return compilation_status == 'success'
            
        except subprocess.TimeoutExpired:
            print("⏰ Leanコンパイルがタイムアウトしました")
            return False
        except FileNotFoundError:
            print("❌ Leanコマンドが見つかりません")
            return False
        except Exception as e:
            print(f"❌ Leanコンパイルエラー: {e}")
            return False
    
    def run_iterative_development(self, max_iterations: int = 5):
        """反復的開発を実行"""
        print("🔄 なんJ風反復的開発開始...")
        
        for iteration in range(max_iterations):
            print(f"\n🔄 反復 {iteration + 1}/{max_iterations}")
            
            # 1. エラー分析
            errors, hypotheses, suggestions = self.run_complete_analysis()
            
            if not errors:
                print("✅ すべてのエラーが解決されました！")
                break
            
            # 2. 自動修正
            fixes_applied = self.run_auto_fix()
            
            # 3. コンパイルチェック
            compilation_success = self.run_lean_compilation_check()
            
            if compilation_success:
                print("✅ 反復開発成功！")
                break
            
            print(f"🔄 反復 {iteration + 1} 完了、次の反復に進みます...")
            time.sleep(2)  # 少し待機
        
        self.save_development_log()
    
    def save_development_log(self):
        """開発ログを保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"analysis_results/nanj_development_log_{timestamp}.json"
        
        # ディレクトリが存在しない場合は作成
        os.makedirs("analysis_results", exist_ok=True)
        
        log_data = {
            'timestamp': timestamp,
            'lean_file': self.lean_file_path,
            'development_log': self.development_log,
            'summary': {
                'total_actions': len(self.development_log),
                'analysis_count': len([log for log in self.development_log if log['action'] == 'complete_analysis']),
                'fix_count': len([log for log in self.development_log if log['action'] == 'auto_fix']),
                'compilation_count': len([log for log in self.development_log if log['action'] == 'lean_compilation_check'])
            }
        }
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"📊 開発ログを保存しました: {log_path}")
        return log_path
    
    def generate_development_summary(self):
        """開発サマリーを生成"""
        print("\n📊 なんJ風開発サマリー")
        print("=" * 50)
        
        total_actions = len(self.development_log)
        analysis_actions = len([log for log in self.development_log if log['action'] == 'complete_analysis'])
        fix_actions = len([log for log in self.development_log if log['action'] == 'auto_fix'])
        compilation_actions = len([log for log in self.development_log if log['action'] == 'lean_compilation_check'])
        
        print(f"総アクション数: {total_actions}")
        print(f"エラー分析回数: {analysis_actions}")
        print(f"自動修正回数: {fix_actions}")
        print(f"コンパイルチェック回数: {compilation_actions}")
        
        if self.development_log:
            last_action = self.development_log[-1]
            print(f"最終アクション: {last_action['action']}")
            print(f"最終タイムスタンプ: {last_action['timestamp']}")
        
        print("=" * 50)

def main():
    """メイン実行関数"""
    print("🚀 なんJ風Lean統合開発ツール起動")
    print("Don't hold back. Give it your all deep think!!")
    
    # 分析対象のLeanファイル
    lean_file = "lean_nkat/nkat_nanj_final_fix.lean"
    
    if not os.path.exists(lean_file):
        print(f"❌ ファイルが見つかりません: {lean_file}")
        return
    
    # 統合ツールの実行
    tool = NanjLeanIntegratedTool(lean_file)
    
    # 反復的開発の実行
    tool.run_iterative_development(max_iterations=3)
    
    # 開発サマリーの生成
    tool.generate_development_summary()
    
    print("\n🎯 なんJ風統合開発完了！")
    print("次のステップ:")
    print("1. 開発ログの確認")
    print("2. 残存エラーの手動修正")
    print("3. 段階的証明の構築")

if __name__ == "__main__":
    main() 