#!/usr/bin/env python3
"""
RTX3080 CUDA動作確認テスト
"""

print("🔥 CUDA Test Starting...")

try:
    import cupy as cp
    print("✅ CuPy imported successfully")
    
    # CUDA情報表示
    device_count = cp.cuda.runtime.getDeviceCount()
    print(f"🚀 CUDA Devices: {device_count}")
    
    if device_count > 0:
        device = cp.cuda.Device(0)
        device.use()
        print(f"🎯 Using Device 0: {device.compute_capability}")
        
        # 簡単な計算テスト
        a = cp.arange(1000000, dtype=cp.float32)
        b = cp.arange(1000000, dtype=cp.float32)
        
        import time
        start = time.time()
        c = a + b
        cp.cuda.Stream.null.synchronize()
        end = time.time()
        
        print(f"✅ CUDA computation test passed: {end-start:.4f}s")
        print(f"📊 Result sample: {c[:5]}")
        
except Exception as e:
    print(f"❌ CUDA test failed: {e}")
    import traceback
    traceback.print_exc()

print("🔥 Test completed!") 