#!/usr/bin/env python3
import torch

def get_checkpoint_info(path):
    try:
        cp = torch.load(path, map_location='cpu')
        print(f"=== {path} ===")
        print(f"Config: {cp.get('config', 'No config')}")
        print(f"Test accuracy: {cp.get('test_accuracy', 'No accuracy')}")
        print(f"Epoch: {cp.get('epoch', 'No epoch')}")
        
        if 'model_state_dict' in cp:
            model_keys = list(cp['model_state_dict'].keys())
            print(f"Model keys (first 5): {model_keys[:5]}")
            print(f"Total model keys: {len(model_keys)}")
            
            # embed_dimの推定
            for key in model_keys:
                if 'cls_token' in key:
                    shape = cp['model_state_dict'][key].shape
                    print(f"cls_token shape: {shape}")
                    if len(shape) >= 2:
                        print(f"Estimated embed_dim: {shape[-1]}")
                    break
        print("-" * 40)
    except Exception as e:
        print(f"Error loading {path}: {e}")

if __name__ == "__main__":
    get_checkpoint_info('checkpoints/nkat_enhanced_v2_best.pth')
    get_checkpoint_info('checkpoints/nkat_final_99_percent.pth') 