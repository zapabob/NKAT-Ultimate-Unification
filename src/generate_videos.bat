@echo off
echo 🎥 NKAT v8.0 Manim動画生成システム
echo ========================================

echo 📦 必要なライブラリをインストール中...
pip install manim matplotlib numpy

echo 🎬 動画生成開始...
manim src/manim_video_generator.py NKATIntroduction -v WARNING --resolution 1080,1920
manim src/manim_video_generator.py RiemannVisualization -v WARNING --resolution 1080,1920  
manim src/manim_video_generator.py GPUPerformance -v WARNING --resolution 1080,1920
manim src/manim_video_generator.py SuccessRateVisualization -v WARNING --resolution 1080,1920
manim src/manim_video_generator.py TheoryExplanation -v WARNING --resolution 1080,1920

echo ✅ 全動画生成完了！
echo 📁 動画ファイルは media/videos/ に保存されました
pause
