#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple CUDA Environment Test for NKAT System
"""

import numpy as np
import sys
import traceback

def test_cuda_environment():
    """CUDA環境のテストを実行"""
    
    print("=" * 50)
    print("CUDA Environment Test for NKAT System")
    print("=" * 50)
    
    # 1. CuPy availability test
    try:
        import cupy as cp
        print("✓ CuPy imported successfully")
        
        # GPU availability check
        if cp.cuda.is_available():
            print("✓ CUDA is available")
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"✓ GPU count: {device_count}")
            
            # GPU memory info
            memory_info = cp.cuda.runtime.memGetInfo()
            total_memory = memory_info[1] / (1024**3)  # GB
            free_memory = memory_info[0] / (1024**3)   # GB
            print(f"✓ GPU Memory: {free_memory:.1f}GB free / {total_memory:.1f}GB total")
            
            # Simple GPU computation test
            print("Testing GPU computation...")
            gpu_array = cp.random.random((1000, 1000))
            result = cp.sum(gpu_array)
            print(f"✓ GPU computation test passed: sum = {float(result):.3f}")
            
        else:
            print("✗ CUDA is not available")
            return False
            
    except ImportError:
        print("✗ CuPy not found - installing CuPy required")
        return False
    except Exception as e:
        print(f"✗ CuPy error: {e}")
        return False
    
    # 2. PyTorch CUDA test
    try:
        import torch
        print("✓ PyTorch imported successfully")
        
        if torch.cuda.is_available():
            print("✓ PyTorch CUDA is available")
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU: {device_name}")
        else:
            print("✗ PyTorch CUDA is not available")
            
    except ImportError:
        print("✗ PyTorch not found")
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
    
    # 3. Required libraries test
    required_libs = [
        'numpy', 'scipy', 'matplotlib', 'tqdm'
    ]
    
    print("\nTesting required libraries:")
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✓ {lib}")
        except ImportError:
            print(f"✗ {lib} not found")
    
    print("\n" + "=" * 50)
    print("CUDA environment test completed")
    print("=" * 50)
    
    return True

def simple_nkat_calculation():
    """簡単なNKAT計算のテスト"""
    
    print("\n" + "=" * 50)
    print("Simple NKAT Calculation Test")
    print("=" * 50)
    
    try:
        # Basic parameters
        gamma = 0.23422
        delta = 0.03511
        Nc = 17.2644
        
        # Test calculation for N=100
        N = 100
        
        # Super convergence factor calculation
        log_term = gamma * np.log(N / Nc) * (1 - np.exp(-delta * (N - Nc)))
        correction = 0.0089 / (N**2) * np.log(N / Nc)**2
        
        S_factor = 1 + log_term + correction
        
        print(f"✓ Test calculation completed")
        print(f"  N = {N}")
        print(f"  Super convergence factor = {S_factor:.6f}")
        print(f"  Log term = {log_term:.6f}")
        print(f"  Correction = {correction:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ NKAT calculation error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        # Test CUDA environment
        cuda_ok = test_cuda_environment()
        
        # Test simple NKAT calculation
        nkat_ok = simple_nkat_calculation()
        
        print(f"\nOverall status:")
        print(f"CUDA Environment: {'✓ OK' if cuda_ok else '✗ FAILED'}")
        print(f"NKAT Calculation: {'✓ OK' if nkat_ok else '✗ FAILED'}")
        
        if cuda_ok and nkat_ok:
            print("\n🚀 System ready for NKAT analysis!")
        else:
            print("\n⚠️ Some issues detected. Please fix before running full analysis.")
            
    except Exception as e:
        print(f"Test script error: {e}")
        traceback.print_exc()
        sys.exit(1) 