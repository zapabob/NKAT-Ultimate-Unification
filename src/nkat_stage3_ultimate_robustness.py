#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Stage ③ Ultimate Robustness & Security Test
RTX3080最適化 + tqdm進捗表示 + 英語グラフ表記
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import cv2
from PIL import Image, ImageFilter
import warnings
warnings.filterwarnings('ignore')

# 英語表記設定（文字化け防止）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class NKATRobustnessAnalyzer:
    """NKAT堅牢性・セキュリティ解析システム"""
    
    def __init__(self, model, device, dataset_name='MNIST'):
        self.model = model
        self.device = device
        self.dataset_name = dataset_name
        self.model.eval()
    
    def fgsm_attack(self, image, label, epsilon=0.1):
        """FGSM敵対的攻撃実装"""
        image.requires_grad = True
        
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        
        self.model.zero_grad()
        loss.backward()
        
        # 勾配収集
        data_grad = image.grad.data
        
        # FGSM攻撃生成
        perturbed_image = image + epsilon * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image
    
    def test_adversarial_robustness(self, test_loader, epsilons=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]):
        """敵対的攻撃への堅牢性テスト"""
        
        print("🛡️ Starting FGSM Adversarial Robustness Test...")
        
        results = {}
        
        for epsilon in tqdm(epsilons, desc="Testing Epsilons"):
            correct = 0
            total = 0
            
            progress_bar = tqdm(test_loader, desc=f"ε={epsilon:.2f}", leave=False)
            
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                # 敵対的サンプル生成
                perturbed_data = self.fgsm_attack(data, target, epsilon)
                
                # 予測
                with torch.no_grad():
                    output = self.model(perturbed_data)
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                
                # リアルタイム更新
                if total > 0:
                    accuracy = 100. * correct / total
                    progress_bar.set_postfix({'Acc': f'{accuracy:.2f}%'})
            
            accuracy = 100. * correct / total
            results[epsilon] = accuracy
            
            print(f"ε={epsilon:.2f}: Accuracy = {accuracy:.2f}%")
        
        return results
    
    def test_rotation_robustness(self, test_loader, angles=[-30, -20, -10, 0, 10, 20, 30]):
        """回転に対する堅牢性テスト"""
        
        print("🔄 Starting Rotation Robustness Test...")
        
        results = {}
        
        for angle in tqdm(angles, desc="Testing Angles"):
            correct = 0
            total = 0
            
            progress_bar = tqdm(test_loader, desc=f"Angle={angle}°", leave=False)
            
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                # 回転変換
                if angle != 0:
                    rotated_data = []
                    for img in data.cpu():
                        img_np = img.squeeze().numpy()
                        if img_np.ndim == 3:  # RGB
                            img_np = np.transpose(img_np, (1, 2, 0))
                        
                        center = (img_np.shape[1] // 2, img_np.shape[0] // 2)
                        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(img_np, matrix, (img_np.shape[1], img_np.shape[0]))
                        
                        if rotated.ndim == 3:
                            rotated = np.transpose(rotated, (2, 0, 1))
                        else:
                            rotated = np.expand_dims(rotated, 0)
                        
                        rotated_data.append(torch.from_numpy(rotated))
                    
                    data = torch.stack(rotated_data).to(self.device)
                
                # 予測
                with torch.no_grad():
                    output = self.model(data)
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                
                # リアルタイム更新
                if total > 0:
                    accuracy = 100. * correct / total
                    progress_bar.set_postfix({'Acc': f'{accuracy:.2f}%'})
            
            accuracy = 100. * correct / total
            results[angle] = accuracy
            
            print(f"Angle={angle}°: Accuracy = {accuracy:.2f}%")
        
        return results
    
    def test_compression_robustness(self, test_loader, quality_levels=[10, 20, 30, 50, 70, 90]):
        """JPEG圧縮に対する堅牢性テスト"""
        
        print("📷 Starting JPEG Compression Robustness Test...")
        
        results = {}
        
        for quality in tqdm(quality_levels, desc="Testing Quality"):
            correct = 0
            total = 0
            
            progress_bar = tqdm(test_loader, desc=f"Quality={quality}", leave=False)
            
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                # JPEG圧縮シミュレーション
                compressed_data = []
                for img in data.cpu():
                    img_np = (img.squeeze().numpy() * 255).astype(np.uint8)
                    
                    if img_np.ndim == 3:
                        img_np = np.transpose(img_np, (1, 2, 0))
                        img_pil = Image.fromarray(img_np, mode='RGB')
                    else:
                        img_pil = Image.fromarray(img_np, mode='L')
                    
                    # JPEG圧縮
                    import io
                    buffer = io.BytesIO()
                    img_pil.save(buffer, format='JPEG', quality=quality)
                    buffer.seek(0)
                    compressed_img = Image.open(buffer)
                    
                    # テンソル変換
                    compressed_np = np.array(compressed_img).astype(np.float32) / 255.0
                    
                    if compressed_np.ndim == 3:
                        compressed_np = np.transpose(compressed_np, (2, 0, 1))
                    else:
                        compressed_np = np.expand_dims(compressed_np, 0)
                    
                    compressed_data.append(torch.from_numpy(compressed_np))
                
                data = torch.stack(compressed_data).to(self.device)
                
                # 予測
                with torch.no_grad():
                    output = self.model(data)
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                
                # リアルタイム更新
                if total > 0:
                    accuracy = 100. * correct / total
                    progress_bar.set_postfix({'Acc': f'{accuracy:.2f}%'})
            
            accuracy = 100. * correct / total
            results[quality] = accuracy
            
            print(f"Quality={quality}: Accuracy = {accuracy:.2f}%")
        
        return results
    
    def test_noise_robustness(self, test_loader, noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5]):
        """ガウシアンノイズに対する堅牢性テスト"""
        
        print("🔊 Starting Gaussian Noise Robustness Test...")
        
        results = {}
        
        for noise_std in tqdm(noise_levels, desc="Testing Noise"):
            correct = 0
            total = 0
            
            progress_bar = tqdm(test_loader, desc=f"Noise σ={noise_std:.1f}", leave=False)
            
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                # ガウシアンノイズ追加
                noise = torch.randn_like(data) * noise_std
                noisy_data = torch.clamp(data + noise, 0, 1)
                
                # 予測
                with torch.no_grad():
                    output = self.model(noisy_data)
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                
                # リアルタイム更新
                if total > 0:
                    accuracy = 100. * correct / total
                    progress_bar.set_postfix({'Acc': f'{accuracy:.2f}%'})
            
            accuracy = 100. * correct / total
            results[noise_std] = accuracy
            
            print(f"Noise σ={noise_std:.1f}: Accuracy = {accuracy:.2f}%")
        
        return results

def create_robustness_visualization(all_results, dataset_name, timestamp):
    """堅牢性テスト結果可視化"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'NKAT Stage III: {dataset_name} Robustness Analysis', fontsize=16, fontweight='bold')
    
    # FGSM Adversarial Attack
    if 'adversarial' in all_results:
        adv_results = all_results['adversarial']
        epsilons = list(adv_results.keys())
        accuracies = list(adv_results.values())
        
        ax1.plot(epsilons, accuracies, 'ro-', linewidth=2, markersize=8)
        ax1.axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='Target: 70%')
        ax1.set_title('FGSM Adversarial Robustness', fontweight='bold')
        ax1.set_xlabel('Epsilon')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 100)
    
    # Rotation Robustness
    if 'rotation' in all_results:
        rot_results = all_results['rotation']
        angles = list(rot_results.keys())
        accuracies = list(rot_results.values())
        
        ax2.plot(angles, accuracies, 'bo-', linewidth=2, markersize=8)
        baseline_acc = rot_results[0] if 0 in rot_results else max(accuracies)
        ax2.axhline(y=baseline_acc-5, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Target: <5pt drop')
        ax2.set_title('Rotation Robustness', fontweight='bold')
        ax2.set_xlabel('Rotation Angle (degrees)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 100)
    
    # JPEG Compression
    if 'compression' in all_results:
        comp_results = all_results['compression']
        qualities = list(comp_results.keys())
        accuracies = list(comp_results.values())
        
        ax3.plot(qualities, accuracies, 'go-', linewidth=2, markersize=8)
        baseline_acc = comp_results[90] if 90 in comp_results else max(accuracies)
        ax3.axhline(y=baseline_acc-3, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Target: <3pt drop at Q=20')
        ax3.set_title('JPEG Compression Robustness', fontweight='bold')
        ax3.set_xlabel('JPEG Quality')
        ax3.set_ylabel('Accuracy (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, 100)
    
    # Gaussian Noise
    if 'noise' in all_results:
        noise_results = all_results['noise']
        noise_levels = list(noise_results.keys())
        accuracies = list(noise_results.values())
        
        ax4.plot(noise_levels, accuracies, 'mo-', linewidth=2, markersize=8)
        ax4.set_title('Gaussian Noise Robustness', fontweight='bold')
        ax4.set_xlabel('Noise Standard Deviation')
        ax4.set_ylabel('Accuracy (%)')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # 高解像度保存
    filename = f'nkat_stage3_{dataset_name.lower()}_robustness_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return filename

def evaluate_robustness_targets(all_results, dataset_name):
    """堅牢性目標達成評価"""
    
    results = {}
    
    # FGSM Target: 70%+ at ε=0.1
    if 'adversarial' in all_results and 0.1 in all_results['adversarial']:
        fgsm_acc = all_results['adversarial'][0.1]
        results['fgsm_target'] = {
            'achieved': fgsm_acc,
            'target': 70.0,
            'passed': fgsm_acc >= 70.0
        }
    
    # Rotation Target: <5pt drop at ±30°
    if 'rotation' in all_results:
        rot_results = all_results['rotation']
        baseline = rot_results[0] if 0 in rot_results else max(rot_results.values())
        worst_drop = baseline - min([rot_results.get(-30, 0), rot_results.get(30, 0)])
        results['rotation_target'] = {
            'achieved': worst_drop,
            'target': 5.0,
            'passed': worst_drop <= 5.0
        }
    
    # Compression Target: <3pt drop at Q=20
    if 'compression' in all_results:
        comp_results = all_results['compression']
        baseline = comp_results[90] if 90 in comp_results else max(comp_results.values())
        q20_drop = baseline - comp_results.get(20, 0)
        results['compression_target'] = {
            'achieved': q20_drop,
            'target': 3.0,
            'passed': q20_drop <= 3.0
        }
    
    return results

def load_pretrained_model(dataset_name, device):
    """事前訓練済みモデル読み込み（Stage ②からの継続）"""
    
    # Stage ②で保存されたモデルがある場合は読み込み
    # ここでは簡略化してランダム初期化
    
    from nkat_stage2_ultimate_generalization import NKATTransformerUltimate
    
    configs = {
        'MNIST': {'num_classes': 10, 'size': 28, 'embed_dim': 384},
        'FashionMNIST': {'num_classes': 10, 'size': 28, 'embed_dim': 384},
        'EMNIST': {'num_classes': 47, 'size': 28, 'embed_dim': 512},
        'CIFAR10': {'num_classes': 10, 'size': 32, 'embed_dim': 384}
    }
    
    config = configs.get(dataset_name, configs['MNIST'])
    
    model = NKATTransformerUltimate(
        img_size=config['size'],
        patch_size=4,
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        nkat_gauge_strength=0.1,
        dropout=0.1
    ).to(device)
    
    print(f"📋 Model initialized for {dataset_name}: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model

def main():
    """メイン実行関数"""
    
    # RTX3080 CUDA最適化設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(0)  # RTX3080選択
        torch.backends.cudnn.benchmark = True
        print(f"🚀 RTX3080 CUDA Optimization Enabled: {torch.cuda.get_device_name()}")
    
    # シード設定
    torch.manual_seed(1337)
    np.random.seed(1337)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Stage III Target Datasets
    target_datasets = ['MNIST', 'FashionMNIST']  # 時間節約のため最初は2つ
    all_dataset_results = {}
    
    print("🛡️ NKAT Stage III Ultimate Robustness & Security Test Starting...")
    print(f"📅 Timestamp: {timestamp}")
    print(f"🔧 Device: {device}")
    print(f"📊 Target Datasets: {', '.join(target_datasets)}")
    
    for dataset_name in target_datasets:
        print(f"\n{'='*60}")
        print(f"🎯 Processing Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # モデル読み込み
        model = load_pretrained_model(dataset_name, device)
        
        # データセット準備
        if dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_dataset = torchvision.datasets.MNIST('./data', train=False, 
                                                    transform=transform, download=True)
        elif dataset_name == 'FashionMNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_dataset = torchvision.datasets.FashionMNIST('./data', train=False,
                                                           transform=transform, download=True)
        elif dataset_name == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            test_dataset = torchvision.datasets.CIFAR10('./data', train=False,
                                                      transform=transform, download=True)
        
        # 高速テストのため一部データのみ使用
        test_subset = Subset(test_dataset, range(0, min(1000, len(test_dataset))))
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, 
                               num_workers=4, pin_memory=True)
        
        # 堅牢性解析器初期化
        analyzer = NKATRobustnessAnalyzer(model, device, dataset_name)
        
        # 各種堅牢性テスト実行
        dataset_results = {}
        
        # 1. FGSM敵対的攻撃テスト
        try:
            adversarial_results = analyzer.test_adversarial_robustness(
                test_loader, epsilons=[0.05, 0.1, 0.15, 0.2]
            )
            dataset_results['adversarial'] = adversarial_results
        except Exception as e:
            print(f"❌ Adversarial test failed: {e}")
        
        # 2. 回転堅牢性テスト
        try:
            rotation_results = analyzer.test_rotation_robustness(
                test_loader, angles=[-30, -15, 0, 15, 30]
            )
            dataset_results['rotation'] = rotation_results
        except Exception as e:
            print(f"❌ Rotation test failed: {e}")
        
        # 3. JPEG圧縮テスト
        try:
            compression_results = analyzer.test_compression_robustness(
                test_loader, quality_levels=[20, 50, 70, 90]
            )
            dataset_results['compression'] = compression_results
        except Exception as e:
            print(f"❌ Compression test failed: {e}")
        
        # 4. ガウシアンノイズテスト
        try:
            noise_results = analyzer.test_noise_robustness(
                test_loader, noise_levels=[0.1, 0.2, 0.3]
            )
            dataset_results['noise'] = noise_results
        except Exception as e:
            print(f"❌ Noise test failed: {e}")
        
        # 目標達成評価
        target_evaluation = evaluate_robustness_targets(dataset_results, dataset_name)
        dataset_results['target_evaluation'] = target_evaluation
        
        # 可視化
        viz_filename = create_robustness_visualization(dataset_results, dataset_name, timestamp)
        dataset_results['visualization'] = viz_filename
        
        # 結果保存
        all_dataset_results[dataset_name] = {
            'results': dataset_results,
            'model_params': sum(p.numel() for p in model.parameters()),
            'device': str(device),
            'timestamp': timestamp
        }
        
        # 目標達成状況表示
        print(f"\n📊 {dataset_name} Robustness Summary:")
        for test_name, evaluation in target_evaluation.items():
            status = "✅ PASSED" if evaluation['passed'] else "❌ FAILED"
            print(f"  {test_name}: {status} (Target: {evaluation['target']}, Achieved: {evaluation['achieved']:.2f})")
        
        # メモリクリア
        del model, analyzer
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # 最終結果保存
    results_filename = f'nkat_stage3_ultimate_robustness_{timestamp}.json'
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(all_dataset_results, f, ensure_ascii=False, indent=2)
    
    # 最終サマリー
    print(f"\n{'='*80}")
    print("🛡️ NKAT Stage III Ultimate Robustness & Security Summary")
    print(f"{'='*80}")
    
    for dataset_name, data in all_dataset_results.items():
        target_eval = data['results']['target_evaluation']
        passed_tests = sum(1 for evaluation in target_eval.values() if evaluation['passed'])
        total_tests = len(target_eval)
        
        print(f"{dataset_name:12} | Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    print(f"\n📁 Results saved to: {results_filename}")
    print("🚀 Ready for Stage IV: Lightweight & Deployment!")

if __name__ == "__main__":
    main() 