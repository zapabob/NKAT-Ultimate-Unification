#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
なんJ風メタプロンプト生成スクリプト
NKAT理論のLean4形式化開発の会話継続用メタプロンプトを生成
"""

import json
import os
from datetime import datetime
from pathlib import Path

def generate_nkat_metaprompt():
    """なんJ風メタプロンプトを生成する関数"""
    
    metaprompt = f"""
# なんJ風メタプロンプト転送

## 会話継続のためのメタプロンプト

### 現在の状況
- **プロジェクト**: NKAT理論のLean4形式化開発
- **現在の状態**: von Waldenfels理論の基本実装完了
- **残存エラー**: Ring ℝ エラー、OfNat ℕ 0 エラー、no goals to be solved エラー
- **解決済みエラー**: OfScientific ℝ エラー、基本的な型システム、代数構造の実装

### 技術的コンテキスト
- **Lean4ファイル**: nkat_nanj_style.lean, nkat_nanj_fixed.lean, nkat_nanj_final_fix.lean
- **開発手法**: なんJ風 Lean Startup アプローチ
- **仮説駆動開発**: 段階的エラー解決と段階的証明構築

### 次のステップ
1. **残存エラーの段階的解決**
   - Ring ℝ エラーの詳細分析
   - OfNat ℕ 0 エラーの段階的解決
   - 証明構造の改善

2. **段階的証明の構築**
   - 小さな定理から始める
   - 各段階でのテスト
   - インクリメンタルな開発

### なんJ風開発方針
- **Don't hold back. Give it your all deep think!!**
- 楽しく段階的なエラー分析
- 具体的な仮説検証
- エラーメッセージの詳細分析
- 仮説の修正と再検証

### 継続指示
このメタプロンプトを受け取ったら、上記の状況を理解し、残存エラーの段階的解決と段階的証明の構築を続けてください。なんJ風に楽しく進めてください！
"""
    
    return metaprompt

def save_conversation_history(history, filename="nkat_conversation_history.json"):
    """会話履歴を保存する関数"""
    
    conversation_data = {
        "meta": {
            "session_id": "nkat_lean_development_20250720",
            "timestamp": datetime.now().isoformat(),
            "context": "NKAT理論のLean4形式化開発",
            "current_state": "von Waldenfels理論の基本実装完了",
            "next_steps": "残存エラーの段階的解決"
        },
        "technical_context": {
            "lean_files": [
                "nkat_nanj_style.lean",
                "nkat_nanj_fixed.lean", 
                "nkat_nanj_final_fix.lean"
            ],
            "current_errors": [
                "Ring ℝ エラー",
                "OfNat ℕ 0 エラー", 
                "no goals to be solved エラー"
            ],
            "solved_errors": [
                "OfScientific ℝ エラー",
                "基本的な型システム",
                "代数構造の実装"
            ]
        },
        "development_approach": {
            "methodology": "なんJ風 Lean Startup",
            "hypothesis_driven": True,
            "step_by_step": True,
            "error_resolution": "段階的解決"
        },
        "metaprompt": generate_nkat_metaprompt(),
        "conversation_history": history
    }
    
    # 出力ディレクトリの作成
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / filename
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversation_data, f, ensure_ascii=False, indent=2)
    
    return str(output_file)

def main():
    """メイン関数"""
    
    print("=== なんJ風メタプロンプト生成システム ===")
    print()
    
    # メタプロンプトの生成
    metaprompt = generate_nkat_metaprompt()
    
    print("=== 生成されたメタプロンプト ===")
    print(metaprompt)
    print("=== メタプロンプト終了 ===")
    print()
    
    # サンプル会話履歴
    sample_history = [
        {"role": "user", "content": "なんJ風に続けて"},
        {"role": "assistant", "content": "承知いたしました！なんJ風に楽しく段階的実装を進めましょう！"},
        {"role": "user", "content": "修正して"},
        {"role": "assistant", "content": "承知いたしました！なんJ風に楽しく段階的実装を進めましょう！"},
        {"role": "user", "content": "メタプロンプトを出力して新しいチャットにわたすための"},
        {"role": "assistant", "content": "承知いたしました！メタプロンプト転送システムを構築しましょう！"}
    ]
    
    # 会話履歴の保存
    filename = save_conversation_history(sample_history)
    print(f"会話履歴を保存しました: {filename}")
    print()
    
    # メタプロンプトファイルの保存
    metaprompt_file = Path("output") / "nkat_metaprompt.md"
    with open(metaprompt_file, 'w', encoding='utf-8') as f:
        f.write(metaprompt)
    
    print(f"メタプロンプトファイルを保存しました: {metaprompt_file}")
    print()
    
    print("=== 使用方法 ===")
    print("1. 新しいチャットでメタプロンプトを使用")
    print("2. 残存エラーの段階的解決を続行")
    print("3. なんJ風に楽しく開発を継続")
    print()
    
    print("**Don't hold back. Give it your all deep think!!**")

if __name__ == "__main__":
    main() 