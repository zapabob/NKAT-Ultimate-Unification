#!/usr/bin/env bash
# run.sh -- NKAT理論の再現性検証用スクリプト
# 使用方法: ./run.sh [config_file] [mode]
# 例: ./run.sh configs/quick.yml --minimal

set -e  # エラー時に停止

# デフォルト設定
CFG=${1:-configs/quick.yml}
MODE=${2:---minimal}

echo "=== NKAT Theory Reproducibility Test ==="
echo "Config: $CFG"
echo "Mode: $MODE"
echo "Timestamp: $(date)"
echo ""

# 依存関係チェック
check_dependencies() {
    echo "Checking dependencies..."
    command -v python >/dev/null 2>&1 || { echo "Python is required but not installed. Aborting." >&2; exit 1; }
    command -v pip >/dev/null 2>&1 || { echo "pip is required but not installed. Aborting." >&2; exit 1; }
    
    # 必要なPythonパッケージのチェック
    python -c "import numpy, scipy, matplotlib, tqdm" 2>/dev/null || {
        echo "Installing required packages..."
        pip install numpy scipy matplotlib tqdm
    }
    echo "✓ Dependencies OK"
}

# 設定ファイルの検証
validate_config() {
    if [ ! -f "$CFG" ]; then
        echo "Error: Config file $CFG not found!"
        echo "Available configs:"
        ls -la configs/
        exit 1
    fi
    echo "✓ Config file validated: $CFG"
}

# クイックテスト実行
run_quick_test() {
    echo ""
    echo "=== Running Quick Test (5 min max) ==="
    
    # 出力ディレクトリ作成
    OUTPUT_DIR="./output_quick_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"
    
    echo "Output directory: $OUTPUT_DIR"
    
    # メイン計算実行
    echo "1. Running NKAT core calculation..."
    python src/core/nkat_core.py --cfg "$CFG" --output "$OUTPUT_DIR" --timeout 180
    
    echo "2. Generating energy norm plot..."
    python src/visualization/plot_energy_norm.py --cfg "$CFG" --output "$OUTPUT_DIR"
    
    echo "3. Running ESS diagnosis..."
    python src/analysis/ess_diagnosis.py --cfg "$CFG" --output "$OUTPUT_DIR"
    
    echo "4. Generating convergence analysis..."
    python src/analysis/convergence_analysis.py --cfg "$CFG" --output "$OUTPUT_DIR"
    
    echo ""
    echo "=== Results Summary ==="
    echo "Output files:"
    ls -la "$OUTPUT_DIR"/*.png "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No output files found"
    
    echo ""
    echo "=== Quick Test Completed ==="
    echo "Results saved in: $OUTPUT_DIR"
    echo "Total execution time: $(($(date +%s) - START_TIME)) seconds"
}

# フルテスト実行
run_full_test() {
    echo ""
    echo "=== Running Full Verification Test ==="
    
    OUTPUT_DIR="./output_full_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"
    
    echo "Output directory: $OUTPUT_DIR"
    
    # 包括的検証実行
    echo "1. Running comprehensive NKAT analysis..."
    python src/verification/comprehensive_verification.py --cfg "$CFG" --output "$OUTPUT_DIR"
    
    echo "2. Generating detailed plots..."
    python src/visualization/generate_all_plots.py --cfg "$CFG" --output "$OUTPUT_DIR"
    
    echo "3. Running statistical analysis..."
    python src/analysis/statistical_analysis.py --cfg "$CFG" --output "$OUTPUT_DIR"
    
    echo "4. Generating final report..."
    python src/reporting/generate_report.py --cfg "$CFG" --output "$OUTPUT_DIR"
    
    echo ""
    echo "=== Full Test Completed ==="
    echo "Results saved in: $OUTPUT_DIR"
}

# メイン実行
main() {
    START_TIME=$(date +%s)
    
    echo "Starting NKAT theory verification..."
    echo "Working directory: $(pwd)"
    
    # 依存関係チェック
    check_dependencies
    
    # 設定ファイル検証
    validate_config
    
    # モードに応じて実行
    case "$MODE" in
        --minimal|--quick)
            run_quick_test
            ;;
        --full|--complete)
            run_full_test
            ;;
        *)
            echo "Unknown mode: $MODE"
            echo "Available modes: --minimal, --quick, --full, --complete"
            exit 1
            ;;
    esac
    
    echo ""
    echo "=== NKAT Verification Complete ==="
    echo "Check the output directory for results and plots."
    echo "For detailed analysis, see the generated JSON files."
}

# スクリプト実行
main "$@" 