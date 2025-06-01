#!/usr/bin/env python3
import torch
import json

def analyze_checkpoint(checkpoint_path):
    """チェックポイントの構造を分析"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"=== Checkpoint Analysis: {checkpoint_path} ===")
        print(f"Total keys: {len(checkpoint)}")
        
        # 重要なキーの分析
        if 'cls_token' in checkpoint:
            embed_dim = checkpoint['cls_token'].shape[-1]
            print(f"Embed dimension: {embed_dim}")
        
        if 'pos_embedding' in checkpoint:
            pos_shape = checkpoint['pos_embedding'].shape
            print(f"Position embedding shape: {pos_shape}")
            num_patches = pos_shape[1] - 1  # CLSトークンを除く
            print(f"Number of patches: {num_patches}")
        
        # Transformerレイヤー数の確認
        transformer_layers = [k for k in checkpoint.keys() if 'transformer.layers.' in k]
        if transformer_layers:
            layer_nums = set()
            for k in transformer_layers:
                parts = k.split('.')
                if len(parts) >= 3 and parts[2].isdigit():
                    layer_nums.add(int(parts[2]))
            max_layer = max(layer_nums) if layer_nums else 0
            print(f"Transformer layers: 0-{max_layer} (total: {max_layer + 1})")
        
        # モデル構造の推定
        print("\n=== Estimated Model Configuration ===")
        if 'cls_token' in checkpoint:
            print(f"embed_dim = {checkpoint['cls_token'].shape[-1]}")
        if transformer_layers:
            print(f"depth = {max_layer + 1}")
        
        # 分類器の確認
        classifier_keys = [k for k in checkpoint.keys() if 'classifier.' in k]
        if classifier_keys:
            print(f"Classifier layers: {len(classifier_keys)}")
        
        return {
            'embed_dim': checkpoint['cls_token'].shape[-1] if 'cls_token' in checkpoint else None,
            'depth': max_layer + 1 if transformer_layers else None,
            'num_patches': pos_shape[1] - 1 if 'pos_embedding' in checkpoint else None
        }
        
    except Exception as e:
        print(f"Error analyzing checkpoint: {e}")
        return None

if __name__ == "__main__":
    # 利用可能なチェックポイントを分析
    checkpoints = [
        'checkpoints/nkat_enhanced_v2_best.pth',
        'checkpoints/nkat_final_99_percent.pth'
    ]
    
    for cp in checkpoints:
        try:
            config = analyze_checkpoint(cp)
            if config:
                print(f"\nConfig for {cp}:")
                print(json.dumps(config, indent=2))
        except:
            print(f"Could not analyze {cp}")
        print("-" * 50) 