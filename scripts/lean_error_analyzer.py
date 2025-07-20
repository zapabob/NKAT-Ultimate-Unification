#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
なんJ風Leanエラー分析ツール
段階的エラー解決と仮説検証思考をサポート
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class NanjLeanErrorAnalyzer:
    """なんJ風Leanエラー分析クラス"""
    
    def __init__(self, lean_file_path: str):
        self.lean_file_path = lean_file_path
        self.errors = []
        self.solutions = []
        self.hypotheses = []
        
    def analyze_lean_file(self) -> Dict:
        """Leanファイルを分析してエラーを検出"""
        print("🔍 なんJ風Leanエラー分析開始...")
        
        try:
            with open(self.lean_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ ファイル読み込みエラー: {e}")
            return {}
        
        # エラーパターンの検出
        error_patterns = {
            'OfNat_error': r'OfNat.*\d+',
            'Ring_error': r'Ring.*ℝ',
            'no_goals_error': r'no goals to be solved',
            'synthesis_error': r'failed to synthesize instance',
            'type_error': r'type mismatch',
            'undefined_error': r'undefined identifier'
        }
        
        detected_errors = {}
        for error_type, pattern in error_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                detected_errors[error_type] = matches
        
        return detected_errors
    
    def generate_hypotheses(self, errors: Dict) -> List[Dict]:
        """エラーから仮説を生成"""
        hypotheses = []
        
        for error_type, matches in errors.items():
            hypothesis = {
                'error_type': error_type,
                'matches': matches,
                'hypothesis': self._generate_hypothesis_for_error(error_type, matches),
                'solution_approach': self._generate_solution_approach(error_type),
                'priority': self._assign_priority(error_type)
            }
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_hypothesis_for_error(self, error_type: str, matches: List[str]) -> str:
        """エラータイプに基づいて仮説を生成"""
        hypotheses = {
            'OfNat_error': 'ℕ型の数値リテラルに対するOfNatインスタンスが不足している',
            'Ring_error': 'ℝ型（Float）に対するRingインスタンスの定義に問題がある',
            'no_goals_error': '証明の構造が不完全で、証明すべき目標が存在しない',
            'synthesis_error': '型クラスのインスタンス合成に失敗している',
            'type_error': '型の不一致が発生している',
            'undefined_error': '未定義の識別子が使用されている'
        }
        return hypotheses.get(error_type, '未知のエラータイプ')
    
    def _generate_solution_approach(self, error_type: str) -> str:
        """エラータイプに基づいて解決アプローチを生成"""
        approaches = {
            'OfNat_error': 'Nat.succやNat.zeroを使って明示的に数値を表現',
            'Ring_error': 'Ringインスタンスの定義を確認・修正',
            'no_goals_error': '証明の構造を見直し、適切な目標を設定',
            'synthesis_error': '型クラスのインスタンス定義を確認・追加',
            'type_error': '型の整合性を確認・修正',
            'undefined_error': '未定義識別子の定義を追加'
        }
        return approaches.get(error_type, '一般的なデバッグアプローチ')
    
    def _assign_priority(self, error_type: str) -> int:
        """エラーの優先度を割り当て"""
        priorities = {
            'OfNat_error': 1,
            'Ring_error': 2,
            'synthesis_error': 3,
            'type_error': 4,
            'undefined_error': 5,
            'no_goals_error': 6
        }
        return priorities.get(error_type, 10)
    
    def generate_fix_suggestions(self, hypotheses: List[Dict]) -> List[str]:
        """仮説に基づいて修正提案を生成"""
        suggestions = []
        
        for hypothesis in sorted(hypotheses, key=lambda x: x['priority']):
            error_type = hypothesis['error_type']
            
            if error_type == 'OfNat_error':
                suggestions.append("""
-- 修正提案: Nat.succとNat.zeroを使用
-- 修正前: X 0 + X 1 + X 2
-- 修正後: X (Nat.zero) + X (Nat.succ Nat.zero) + X (Nat.succ (Nat.succ Nat.zero))
""")
            
            elif error_type == 'Ring_error':
                suggestions.append("""
-- 修正提案: Ringインスタンスの確認
instance : Ring Float where
  add := fun a b => a + b
  mul := fun a b => a * b
  zero := 0.0
  one := 1.0
  neg := fun a => -a
""")
            
            elif error_type == 'no_goals_error':
                suggestions.append("""
-- 修正提案: 証明構造の改善
theorem example_theorem :
  ∀ (x : A), x = x := by
  intro x
  rfl  -- 明示的な証明終了
""")
        
        return suggestions
    
    def save_analysis_report(self, errors: Dict, hypotheses: List[Dict], suggestions: List[str]):
        """分析レポートを保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"analysis_results/lean_error_analysis_{timestamp}.json"
        
        # ディレクトリが存在しない場合は作成
        os.makedirs("analysis_results", exist_ok=True)
        
        report = {
            'timestamp': timestamp,
            'lean_file': self.lean_file_path,
            'errors_detected': errors,
            'hypotheses': hypotheses,
            'suggestions': suggestions,
            'analysis_summary': {
                'total_errors': sum(len(matches) for matches in errors.values()),
                'error_types': list(errors.keys()),
                'priority_order': [h['error_type'] for h in sorted(hypotheses, key=lambda x: x['priority'])]
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📊 分析レポートを保存しました: {report_path}")
        return report_path

def main():
    """メイン実行関数"""
    print("🚀 なんJ風Leanエラー分析ツール起動")
    print("Don't hold back. Give it your all deep think!!")
    
    # 分析対象のLeanファイル
    lean_file = "lean_nkat/nkat_nanj_final_fix.lean"
    
    if not os.path.exists(lean_file):
        print(f"❌ ファイルが見つかりません: {lean_file}")
        return
    
    # エラー分析の実行
    analyzer = NanjLeanErrorAnalyzer(lean_file)
    errors = analyzer.analyze_lean_file()
    
    if not errors:
        print("✅ エラーは検出されませんでした！")
        return
    
    print(f"🔍 検出されたエラー: {len(errors)}種類")
    for error_type, matches in errors.items():
        print(f"  - {error_type}: {len(matches)}件")
    
    # 仮説生成
    hypotheses = analyzer.generate_hypotheses(errors)
    print(f"💡 生成された仮説: {len(hypotheses)}件")
    
    # 修正提案生成
    suggestions = analyzer.generate_fix_suggestions(hypotheses)
    print(f"🔧 修正提案: {len(suggestions)}件")
    
    # レポート保存
    report_path = analyzer.save_analysis_report(errors, hypotheses, suggestions)
    
    print("\n🎯 なんJ風分析完了！")
    print("次のステップ:")
    print("1. 優先度順にエラーを解決")
    print("2. 段階的に仮説を検証")
    print("3. 修正提案を適用")
    print("4. テストして結果を確認")

if __name__ == "__main__":
    main() 