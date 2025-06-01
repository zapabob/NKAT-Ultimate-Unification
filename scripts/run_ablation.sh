#!/bin/bash
# NKAT-Transformer Ablation Study - Complete Reproducible Script
# Target: 論文化 & 再現性保証パッケージ
# Seed: 1337 (固定)

echo "🔬 NKAT-Transformer Ablation Study"
echo "Reproducing results with fixed seed 1337"
echo "RTX3080 CUDA optimization enabled"
echo "========================================="

# Create directories
mkdir -p ablation_results
mkdir -p ablation_figures
mkdir -p ablation_logs

# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Run ablation study
echo "Starting ablation study..."
py -3 nkat_ablation_study.py --seed 1337 2>&1 | tee ablation_logs/ablation_run_$(date +%Y%m%d_%H%M%S).log

# Move results to organized folders
mv nkat_ablation_study_*.png ablation_figures/
mv nkat_training_curves_*.png ablation_figures/
mv nkat_ablation_results_*.json ablation_results/

echo "✅ Ablation study completed!"
echo "📊 Results saved in ablation_results/"
echo "📈 Figures saved in ablation_figures/"
echo "📝 Logs saved in ablation_logs/" 