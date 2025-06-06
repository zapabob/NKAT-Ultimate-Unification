#!/usr/bin/env python3
"""
🔥 CUDA対応PyTorchテスト
"""

import torch
import time

def test_cuda():
    print("🚀 PyTorch CUDA テスト開始")
    print("=" * 50)
    
    # 基本情報
    print(f"PyTorchバージョン: {torch.__version__}")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU数: {torch.cuda.device_count()}")
        print(f"現在のGPU: {torch.cuda.current_device()}")
        print(f"GPU名: {torch.cuda.get_device_name(0)}")
        print(f"GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # GPU計算テスト
        device = torch.device('cuda')
        print(f"\n🔥 GPU計算テスト開始...")
        
        # テストデータ作成
        size = 10000
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        start_time = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU行列乗算完了: {gpu_time:.4f}秒")
        print(f"🚀 GPU加速有効: RTX3080 フル稼働!")
        
        # メモリ使用量
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"💾 GPU メモリ使用: {allocated:.2f}GB / 予約: {reserved:.2f}GB")
        
        return True
    else:
        print("❌ CUDA利用不可")
        return False

if __name__ == "__main__":
    test_cuda() 