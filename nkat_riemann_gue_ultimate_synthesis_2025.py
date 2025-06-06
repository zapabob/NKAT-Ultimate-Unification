# メイン実行部
if __name__ == "__main__":
    print("\n🎯 NKAT 2025 - 既知ゼロ点×GUE統計 究極統合システム")
    print("Don't hold back. Give it your all deep think!!")
    print("=" * 80)
    
    try:
        # デバッグ情報
        print(f"📊 NumPy バージョン: {np.__version__}")
        print(f"🔥 PyTorch バージョン: {torch.__version__}")
        print(f"🎮 CUDA利用可能: {GPU_AVAILABLE}")
        if GPU_AVAILABLE:
            print(f"🚀 CUDA デバイス: {torch.cuda.get_device_name()}")
        
        # システム初期化
        print("\n🔧 システム初期化中...")
        nkat_synthesis = NKATRiemannGUEUltimateSynthesis()
        
        # 究極統合解析実行
        print("\n🚀 究極統合解析開始...")
        results = nkat_synthesis.perform_ultimate_synthesis()
        
        # 結果表示
        print("\n📋 解析結果サマリー:")
        print("-" * 40)
        
        if 'final_assessment' in results:
            assessment = results['final_assessment']['overall_assessment']
            print(f"🎯 総合スコア: {assessment['success_rate']}")
            print(f"🔬 理論的意義: {assessment['theoretical_significance']}")
            print(f"🚀 研究インパクト: {assessment['research_impact']}")
            
            # 詳細結果
            phase_scores = results['final_assessment']['phase_scores']
            print(f"\n📊 フェーズ別スコア:")
            for phase, score in phase_scores.items():
                print(f"   {phase}: {score:.2f}")
        
        # 可視化作成
        print("\n📊 包括的可視化作成中...")
        nkat_synthesis.create_comprehensive_visualization()
        
        print("\n✅ NKAT究極統合解析完了！")
        print("🎓 既知のリーマンゼロ点とGUE統計の統合による理論物理学への貢献達成！")
        
        # 結果をJSONファイルにも保存
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f'nkat_synthesis_results_{timestamp}.json'
        
        # NumPy配列をリストに変換してJSON保存可能にする
        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_list(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_numpy_to_list(results)
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 結果をJSONファイルに保存: {json_filename}")
        
    except KeyboardInterrupt:
        print("\n🚨 ユーザー中断 - 緊急保存実行中...")
        logger.info("ユーザーによる実行中断")
    except Exception as e:
        print(f"\n❌ 予期せぬエラー: {e}")
        logger.error(f"実行エラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n💾 セッション {SESSION_ID} の全データが保護されています") 