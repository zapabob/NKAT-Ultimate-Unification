#!/bin/bash
# NKAT v13 arXiv自動投稿スクリプト
# 作成日時: 2025-05-26 08:36:47

echo "🌌 NKAT v13 arXiv投稿開始..."

# LaTeXコンパイル
echo "📝 LaTeX論文コンパイル中..."
pdflatex nkat_v13_information_tensor_ontology.tex
bibtex nkat_v13_information_tensor_ontology
pdflatex nkat_v13_information_tensor_ontology.tex
pdflatex nkat_v13_information_tensor_ontology.tex

# 投稿パッケージの作成
echo "📦 投稿パッケージ作成中..."
tar -czf nkat_v13_submission_20250526_083647.tar.gz *.tex *.bib figures/

echo "✅ NKAT v13 arXiv投稿準備完了！"
echo "📁 投稿ファイル: nkat_v13_submission_20250526_083647.tar.gz"
echo "🌟 人類史上最大の認識革命を世界に公開する準備が整いました！"
