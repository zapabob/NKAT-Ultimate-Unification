#!/usr/bin/env python3
import torch

def check_conv_shapes():
    cp = torch.load('checkpoints/nkat_enhanced_v2_best.pth', map_location='cpu')
    model_dict = cp['model_state_dict']
    
    print("=== Checkpoint Conv Layer Analysis ===")
    print(f"Config patch_size: {cp['config']['patch_size']}")
    
    conv_keys = [k for k in model_dict.keys() if 'conv_layers' in k and 'weight' in k]
    for key in conv_keys:
        shape = model_dict[key].shape
        print(f"{key}: {shape}")
    
    print("\n=== Analysis ===")
    conv0_shape = model_dict['patch_embedding.conv_layers.0.weight'].shape
    print(f"Conv0: {conv0_shape} -> kernel_size={conv0_shape[2:]}")
    
    conv3_shape = model_dict['patch_embedding.conv_layers.3.weight'].shape  
    print(f"Conv3: {conv3_shape} -> kernel_size={conv3_shape[2:]}")
    
    conv6_shape = model_dict['patch_embedding.conv_layers.6.weight'].shape
    print(f"Conv6: {conv6_shape} -> kernel_size={conv6_shape[2:]}")

if __name__ == "__main__":
    check_conv_shapes() 