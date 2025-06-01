#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Stage ④ Lightweight & Deployment Optimization
RTX3080最適化 + tqdm進捗表示 + 英語グラフ表記
Knowledge Distillation, Pruning, Quantization, ONNX Export
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ONNX対応
try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX not available - install with: pip install onnx onnxruntime")

# 英語表記設定（文字化け防止）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DistilledNKATStudent(nn.Module):
    """Knowledge Distillation用軽量学生モデル"""
    
    def __init__(self, img_size=28, patch_size=4, num_classes=10, 
                 embed_dim=256, depth=4, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # 軽量化されたPatch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels=1 if num_classes <= 47 else 3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 簡素化されたPositional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 軽量Transformer Blocks
        self.blocks = nn.ModuleList([
            DistilledTransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Add class token and positional encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.head(x[:, 0])

class DistilledTransformerBlock(nn.Module):
    """軽量化Transformer Block"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),  # GELUより軽量
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        # 標準的なresidual connection（軽量化）
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class NKATDistillationTrainer:
    """Knowledge Distillation訓練システム"""
    
    def __init__(self, teacher_model, student_model, device, temperature=4.0, alpha=0.3):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher_model.eval()  # 教師は常に評価モード
        
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """蒸留損失計算"""
        
        # Hard target loss (学生と真のラベル)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft target loss (学生と教師)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # 組み合わせ
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss, hard_loss, soft_loss
    
    def train_student(self, train_loader, num_epochs=20):
        """学生モデル訓練"""
        
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        self.student_model.train()
        training_results = {'total_losses': [], 'hard_losses': [], 'soft_losses': []}
        
        print("🎓 Knowledge Distillation Training Starting...")
        
        for epoch in tqdm(range(num_epochs), desc="Distillation Training"):
            epoch_total_loss = 0.0
            epoch_hard_loss = 0.0
            epoch_soft_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # 教師の予測（勾配計算なし）
                with torch.no_grad():
                    teacher_logits = self.teacher_model(data)
                
                # 学生の予測
                student_logits = self.student_model(data)
                
                # 蒸留損失計算
                total_loss, hard_loss, soft_loss = self.distillation_loss(
                    student_logits, teacher_logits, target
                )
                
                total_loss.backward()
                optimizer.step()
                
                epoch_total_loss += total_loss.item()
                epoch_hard_loss += hard_loss.item()
                epoch_soft_loss += soft_loss.item()
                
                progress_bar.set_postfix({
                    'Total': f'{total_loss.item():.4f}',
                    'Hard': f'{hard_loss.item():.4f}',
                    'Soft': f'{soft_loss.item():.4f}'
                })
            
            scheduler.step()
            
            # エポック結果記録
            training_results['total_losses'].append(epoch_total_loss / len(train_loader))
            training_results['hard_losses'].append(epoch_hard_loss / len(train_loader))
            training_results['soft_losses'].append(epoch_soft_loss / len(train_loader))
            
            print(f"Epoch {epoch+1}: Total={epoch_total_loss/len(train_loader):.4f}, "
                  f"Hard={epoch_hard_loss/len(train_loader):.4f}, "
                  f"Soft={epoch_soft_loss/len(train_loader):.4f}")
        
        return training_results

class NKATPruningOptimizer:
    """モデル剪定最適化システム"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def magnitude_pruning(self, pruning_ratio=0.3):
        """マグニチュード剪定実装"""
        
        print(f"✂️ Starting Magnitude Pruning (ratio: {pruning_ratio:.1%})...")
        
        # 全パラメータの重みを収集
        all_weights = []
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                all_weights.extend(module.weight.data.abs().flatten().tolist())
        
        # 閾値計算
        all_weights = torch.tensor(all_weights)
        threshold = torch.quantile(all_weights, pruning_ratio)
        
        # 剪定実行
        pruned_params = 0
        total_params = 0
        
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                mask = module.weight.data.abs() > threshold
                module.weight.data *= mask.float()
                
                pruned_params += (mask == 0).sum().item()
                total_params += mask.numel()
        
        actual_pruning_ratio = pruned_params / total_params
        
        print(f"✅ Pruning completed: {actual_pruning_ratio:.1%} parameters removed")
        
        return actual_pruning_ratio
    
    def structured_pruning(self, channels_to_prune=0.2):
        """構造化剪定（チャンネル剪定）"""
        
        print(f"🔧 Starting Structured Pruning (channels: {channels_to_prune:.1%})...")
        
        # 簡略化：最初のConv2dレイヤーのチャンネル数を削減
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and 'patch_embed' in name:
                original_channels = module.out_channels
                channels_to_keep = int(original_channels * (1 - channels_to_prune))
                
                # チャンネル重要度計算（L1ノルム）
                channel_importance = module.weight.data.abs().sum(dim=(1, 2, 3))
                _, important_channels = torch.topk(channel_importance, channels_to_keep)
                
                # 新しい重みとバイアス
                new_weight = module.weight.data[important_channels]
                new_bias = module.bias.data[important_channels] if module.bias is not None else None
                
                # モジュール更新
                new_conv = nn.Conv2d(
                    module.in_channels, channels_to_keep,
                    module.kernel_size, module.stride, module.padding
                ).to(self.device)
                
                new_conv.weight.data = new_weight
                if new_bias is not None:
                    new_conv.bias.data = new_bias
                
                # モデル内のモジュール置換
                setattr(self.model, name.split('.')[-1], new_conv)
                
                print(f"✅ {name}: {original_channels} → {channels_to_keep} channels")
                break
        
        return channels_to_prune

class NKATQuantizer:
    """INT8量子化システム"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
    
    def quantize_model(self, calibration_loader):
        """INT8量子化実行"""
        
        print("🔢 Starting INT8 Quantization...")
        
        # PyTorchの量子化APIを使用
        self.model.eval()
        
        # 量子化設定
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 量子化準備
        torch.quantization.prepare(self.model, inplace=True)
        
        # キャリブレーション（一部データで実行）
        print("📊 Calibrating with sample data...")
        with torch.no_grad():
            for i, (data, _) in enumerate(tqdm(calibration_loader, desc="Calibration")):
                if i >= 100:  # 100バッチで十分
                    break
                data = data.to(self.device)
                _ = self.model(data)
        
        # 量子化変換
        torch.quantization.convert(self.model, inplace=True)
        
        print("✅ INT8 Quantization completed")
        
        return self.model

def export_to_onnx(model, input_shape, filename, device):
    """ONNX形式でエクスポート"""
    
    if not ONNX_AVAILABLE:
        print("❌ ONNX export skipped - ONNX not available")
        return None
    
    print(f"📦 Exporting to ONNX: {filename}")
    
    model.eval()
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            filename,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # ONNX模型验证
        onnx_model = onnx.load(filename)
        onnx.checker.check_model(onnx_model)
        
        print(f"✅ ONNX export successful: {filename}")
        
        return filename
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None

def benchmark_models(original_model, optimized_models, test_loader, device):
    """モデルベンチマーク比較"""
    
    print("🏁 Starting Model Benchmark...")
    
    results = {}
    
    # オリジナルモデル測定
    results['original'] = benchmark_single_model(original_model, test_loader, device, "Original")
    
    # 最適化モデル測定
    for name, model in optimized_models.items():
        results[name] = benchmark_single_model(model, test_loader, device, name)
    
    return results

def benchmark_single_model(model, test_loader, device, model_name):
    """単一モデルベンチマーク"""
    
    model.eval()
    
    # 精度測定
    correct = 0
    total = 0
    
    # 速度測定
    import time
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"Benchmarking {model_name}", leave=False):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    end_time = time.time()
    
    accuracy = 100. * correct / total
    inference_time = end_time - start_time
    throughput = total / inference_time
    
    # パラメータ数計算
    num_params = sum(p.numel() for p in model.parameters())
    
    # モデルサイズ計算（概算）
    model_size_mb = num_params * 4 / (1024 * 1024)  # 32bit float
    
    results = {
        'accuracy': accuracy,
        'inference_time': inference_time,
        'throughput': throughput,
        'num_params': num_params,
        'model_size_mb': model_size_mb
    }
    
    print(f"📊 {model_name}: Accuracy={accuracy:.2f}%, "
          f"Time={inference_time:.2f}s, "
          f"Throughput={throughput:.1f} samples/s, "
          f"Params={num_params:,}, "
          f"Size={model_size_mb:.1f}MB")
    
    return results

def create_deployment_visualization(benchmark_results, timestamp):
    """デプロイメント最適化結果可視化"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('NKAT Stage IV: Deployment Optimization Results', fontsize=16, fontweight='bold')
    
    model_names = list(benchmark_results.keys())
    
    # 精度比較
    accuracies = [benchmark_results[name]['accuracy'] for name in model_names]
    bars1 = ax1.bar(model_names, accuracies, color=['blue', 'green', 'orange', 'red'][:len(model_names)])
    ax1.set_title('Model Accuracy Comparison', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # スループット比較
    throughputs = [benchmark_results[name]['throughput'] for name in model_names]
    bars2 = ax2.bar(model_names, throughputs, color=['blue', 'green', 'orange', 'red'][:len(model_names)])
    ax2.set_title('Inference Throughput Comparison', fontweight='bold')
    ax2.set_ylabel('Samples/second')
    for bar, thr in zip(bars2, throughputs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{thr:.0f}', ha='center', va='bottom')
    
    # モデルサイズ比較
    sizes = [benchmark_results[name]['model_size_mb'] for name in model_names]
    bars3 = ax3.bar(model_names, sizes, color=['blue', 'green', 'orange', 'red'][:len(model_names)])
    ax3.set_title('Model Size Comparison', fontweight='bold')
    ax3.set_ylabel('Size (MB)')
    for bar, size in zip(bars3, sizes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{size:.1f}MB', ha='center', va='bottom')
    
    # パラメータ数比較
    params = [benchmark_results[name]['num_params'] for name in model_names]
    bars4 = ax4.bar(model_names, params, color=['blue', 'green', 'orange', 'red'][:len(model_names)])
    ax4.set_title('Parameter Count Comparison', fontweight='bold')
    ax4.set_ylabel('Parameters')
    for bar, param in zip(bars4, params):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + param*0.01,
                f'{param/1e6:.1f}M', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 高解像度保存
    filename = f'nkat_stage4_deployment_optimization_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return filename

def main():
    """メイン実行関数"""
    
    # RTX3080 CUDA最適化設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        print(f"🚀 RTX3080 CUDA Optimization Enabled: {torch.cuda.get_device_name()}")
    
    # シード設定
    torch.manual_seed(1337)
    np.random.seed(1337)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("⚡ NKAT Stage IV Ultimate Deployment Optimization Starting...")
    print(f"📅 Timestamp: {timestamp}")
    print(f"🔧 Device: {device}")
    
    # データセット準備（MNIST使用）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # 教師モデル（Stage ②からの継続を仮定）
    from nkat_stage2_ultimate_generalization import NKATTransformerPractical
    
    teacher_model = NKATTransformerPractical(
        img_size=28, patch_size=4, num_classes=10,
        embed_dim=384, depth=6, num_heads=8
    ).to(device)
    
    print(f"📋 Teacher Model: {sum(p.numel() for p in teacher_model.parameters()):,} parameters")
    
    # 最適化モデル辞書
    optimized_models = {}
    
    # 1. Knowledge Distillation
    print("\n" + "="*60)
    print("🎓 Step 1: Knowledge Distillation")
    print("="*60)
    
    student_model = DistilledNKATStudent(
        img_size=28, patch_size=4, num_classes=10,
        embed_dim=256, depth=4, num_heads=4
    ).to(device)
    
    distillation_trainer = NKATDistillationTrainer(teacher_model, student_model, device)
    
    # 軽量化のため5エポックのみ
    distillation_results = distillation_trainer.train_student(train_loader, num_epochs=5)
    
    optimized_models['distilled'] = student_model
    
    # 2. Model Pruning
    print("\n" + "="*60)
    print("✂️ Step 2: Model Pruning")
    print("="*60)
    
    # 教師モデルのコピーを剪定
    import copy
    pruned_model = copy.deepcopy(teacher_model)
    
    pruning_optimizer = NKATPruningOptimizer(pruned_model, device)
    pruning_ratio = pruning_optimizer.magnitude_pruning(pruning_ratio=0.3)
    
    optimized_models['pruned'] = pruned_model
    
    # 3. INT8 Quantization（簡略化）
    print("\n" + "="*60)
    print("🔢 Step 3: INT8 Quantization")
    print("="*60)
    
    try:
        # 量子化のための小さなキャリブレーションデータセット
        calibration_subset = torch.utils.data.Subset(train_dataset, range(0, 1000))
        calibration_loader = DataLoader(calibration_subset, batch_size=32, shuffle=False)
        
        quantized_model = copy.deepcopy(teacher_model)
        quantizer = NKATQuantizer(quantized_model, device)
        quantized_model = quantizer.quantize_model(calibration_loader)
        
        optimized_models['quantized'] = quantized_model
        
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
    
    # 4. ONNX Export
    print("\n" + "="*60)
    print("📦 Step 4: ONNX Export")
    print("="*60)
    
    for model_name, model in optimized_models.items():
        onnx_filename = f'nkat_{model_name}_model_{timestamp}.onnx'
        export_to_onnx(model, (1, 28, 28), onnx_filename, device)
    
    # 5. Benchmark比較
    print("\n" + "="*60)
    print("🏁 Step 5: Performance Benchmark")
    print("="*60)
    
    # テストデータを小さくして高速化
    test_subset = torch.utils.data.Subset(test_dataset, range(0, 1000))
    test_loader_small = DataLoader(test_subset, batch_size=128, shuffle=False)
    
    benchmark_results = benchmark_models(teacher_model, optimized_models, test_loader_small, device)
    
    # 6. 結果可視化
    viz_filename = create_deployment_visualization(benchmark_results, timestamp)
    
    # 7. 結果保存
    final_results = {
        'benchmark_results': benchmark_results,
        'distillation_training': distillation_results,
        'pruning_ratio': pruning_ratio,
        'timestamp': timestamp,
        'device': str(device),
        'visualization': viz_filename
    }
    
    results_filename = f'nkat_stage4_deployment_results_{timestamp}.json'
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # 最終サマリー
    print(f"\n{'='*80}")
    print("⚡ NKAT Stage IV Deployment Optimization Summary")
    print(f"{'='*80}")
    
    original_acc = benchmark_results['original']['accuracy']
    original_size = benchmark_results['original']['model_size_mb']
    original_throughput = benchmark_results['original']['throughput']
    
    print(f"Original Model: {original_acc:.2f}% accuracy, {original_size:.1f}MB, {original_throughput:.1f} samples/s")
    print()
    
    for model_name, results in benchmark_results.items():
        if model_name == 'original':
            continue
        
        acc_drop = original_acc - results['accuracy']
        size_reduction = (1 - results['model_size_mb'] / original_size) * 100
        speed_improvement = (results['throughput'] / original_throughput - 1) * 100
        
        print(f"{model_name.capitalize():12} | Accuracy: {results['accuracy']:.2f}% ({acc_drop:+.1f}%) | "
              f"Size: {size_reduction:+.1f}% | Speed: {speed_improvement:+.1f}%")
    
    print(f"\n📁 Results saved to: {results_filename}")
    print("🚀 Ready for Stage V: Interpretability & Visualization!")

if __name__ == "__main__":
    main() 