
# NKAT v9.0 - arXiv投稿コマンド
# 実行日時: 2025-05-26 04:30:42

# 1. 投稿ディレクトリに移動
cd arxiv_submission_20250526_043042

# 2. LaTeXコンパイル確認
pdflatex NKAT_v9_1000gamma_breakthrough.tex
bibtex NKAT_v9_1000gamma_breakthrough
pdflatex NKAT_v9_1000gamma_breakthrough.tex
pdflatex NKAT_v9_1000gamma_breakthrough.tex

# 3. arXiv投稿パッケージ作成
tar -czf nkat_v9_1000gamma_submission.tar.gz *.tex *.bib figures/ *.json

# 4. arXiv投稿（手動実行）
# https://arxiv.org/submit にアクセス
# nkat_v9_1000gamma_submission.tar.gz をアップロード

# 5. 投稿確認
echo "🚀 NKAT v9.0 - 1000γ Challenge arXiv投稿準備完了！"
echo "📁 投稿ファイル: nkat_v9_1000gamma_submission.tar.gz"
echo "🌐 arXiv URL: https://arxiv.org/submit"
