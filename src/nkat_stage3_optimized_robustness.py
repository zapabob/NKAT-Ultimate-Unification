#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ NKAT Stage3 最適化ロバストネステスト
NKATTransformerPractical + RTX3080 CUDA + 電源断リカバリー対応
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
import os
from datetime import datetime
from tqdm import tqdm
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class NKATPatchEmbedding(nn.Module):
    """段階的パッチ埋め込み"""
    
    def __init__(self, img_size=28, patch_size=4, channels=1, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 段階的畳み込み
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, embed_dim // 4, kernel_size=patch_size//2, stride=patch_size//2, padding=0),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(embed_dim),
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)
        return x

class NKATTransformerPractical(nn.Module):
    """NKAT Transformer Practical Edition"""
    
    def __init__(self, img_size=28, patch_size=4, num_classes=10, 
                 embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, 
                 nkat_strength=0.0, dropout=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.nkat_strength = nkat_strength
        
        channels = 1 if num_classes <= 47 else 3
        
        self.patch_embedding = NKATPatchEmbedding(img_size, patch_size, channels, embed_dim)
        
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.input_norm = nn.LayerNorm(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        self.apply(self._init_weights)
        self.use_amp = torch.cuda.is_available()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embedding(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embedding
        x = self.input_norm(x)
        
        if self.nkat_strength > 0:
            mean_activation = x.mean(dim=-1, keepdim=True)
            nkat_factor = 1.0 + self.nkat_strength * 0.01 * torch.tanh(mean_activation)
            x = x * nkat_factor
        
        x = self.transformer(x)
        
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        
        return logits

class NKATRobustnessAnalyzer:
    """NKAT堅牢性・セキュリティ解析システム"""
    
    def __init__(self, model, device, dataset_name='MNIST'):
        self.model = model
        self.device = device
        self.dataset_name = dataset_name
        self.model.eval()
        
    def fgsm_attack(self, image, label, epsilon=0.1):
        """FGSM敵対的攻撃"""
        image.requires_grad = True
        
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        
        self.model.zero_grad()
        loss.backward()
        
        data_grad = image.grad.data
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
                
                perturbed_data = self.fgsm_attack(data, target, epsilon)
                
                with torch.no_grad():
                    output = self.model(perturbed_data)
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                
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
                
                if angle != 0:
                    rotated_data = []
                    for img in data.cpu():
                        img_np = img.squeeze().numpy()
                        if img_np.ndim == 3:
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
                
                with torch.no_grad():
                    output = self.model(data)
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                
                if total > 0:
                    accuracy = 100. * correct / total
                    progress_bar.set_postfix({'Acc': f'{accuracy:.2f}%'})
            
            accuracy = 100. * correct / total
            results[angle] = accuracy
            
            print(f"Angle={angle}°: Accuracy = {accuracy:.2f}%")
        
        return results
    
    def test_noise_robustness(self, test_loader, noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5]):
        """ガウシアンノイズに対する堅牢性テスト"""
        
        print("🔊 Starting Gaussian Noise Robustness Test...")
        
        results = {}
        
        for noise_level in tqdm(noise_levels, desc="Testing Noise Levels"):
            correct = 0
            total = 0
            
            progress_bar = tqdm(test_loader, desc=f"Noise={noise_level:.1f}", leave=False)
            
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                noise = torch.randn_like(data) * noise_level
                noisy_data = torch.clamp(data + noise, 0, 1)
                
                with torch.no_grad():
                    output = self.model(noisy_data)
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                
                if total > 0:
                    accuracy = 100. * correct / total
                    progress_bar.set_postfix({'Acc': f'{accuracy:.2f}%'})
            
            accuracy = 100. * correct / total
            results[noise_level] = accuracy
            
            print(f"Noise={noise_level:.1f}: Accuracy = {accuracy:.2f}%")
        
        return results

def create_robustness_visualization(all_results, dataset_name, timestamp):
    """堅牢性結果の可視化"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 敵対的攻撃結果
    if 'adversarial' in all_results:
        epsilons = list(all_results['adversarial'].keys())
        accuracies = list(all_results['adversarial'].values())
        
        ax1.plot(epsilons, accuracies, 'ro-', linewidth=2, markersize=8)
        ax1.set_xlabel('Epsilon (Attack Strength)')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('FGSM Adversarial Attack Robustness')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
    
    # 回転堅牢性
    if 'rotation' in all_results:
        angles = list(all_results['rotation'].keys())
        accuracies = list(all_results['rotation'].values())
        
        ax2.plot(angles, accuracies, 'bo-', linewidth=2, markersize=8)
        ax2.set_xlabel('Rotation Angle (degrees)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Rotation Robustness')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
    
    # ノイズ堅牢性
    if 'noise' in all_results:
        noise_levels = list(all_results['noise'].keys())
        accuracies = list(all_results['noise'].values())
        
        ax3.plot(noise_levels, accuracies, 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Noise Level (std)')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Gaussian Noise Robustness')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
    
    # 堅牢性サマリー
    summary_metrics = []
    summary_values = []
    
    if 'adversarial' in all_results:
        min_adv_acc = min(all_results['adversarial'].values())
        summary_metrics.append('Min Adversarial Acc')
        summary_values.append(min_adv_acc)
    
    if 'rotation' in all_results:
        min_rot_acc = min(all_results['rotation'].values())
        summary_metrics.append('Min Rotation Acc')
        summary_values.append(min_rot_acc)
    
    if 'noise' in all_results:
        min_noise_acc = min(all_results['noise'].values())
        summary_metrics.append('Min Noise Acc')
        summary_values.append(min_noise_acc)
    
    if summary_metrics:
        ax4.barh(summary_metrics, summary_values, color=['red', 'blue', 'green'][:len(summary_metrics)])
        ax4.set_xlabel('Accuracy (%)')
        ax4.set_title('Robustness Summary')
        ax4.set_xlim(0, 100)
    
    plt.tight_layout()
    
    output_path = f"logs/nkat_stage3_robustness_{dataset_name}_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def calculate_robustness_score(all_results):
    """総合堅牢性スコア計算"""
    
    scores = []
    
    if 'adversarial' in all_results:
        adv_scores = list(all_results['adversarial'].values())
        min_adv = min(adv_scores)
        avg_adv = np.mean(adv_scores)
        scores.append(min_adv * 0.6 + avg_adv * 0.4)  # 最悪ケース重視
    
    if 'rotation' in all_results:
        rot_scores = list(all_results['rotation'].values())
        min_rot = min(rot_scores)
        avg_rot = np.mean(rot_scores)
        scores.append(min_rot * 0.4 + avg_rot * 0.6)  # 平均重視
    
    if 'noise' in all_results:
        noise_scores = list(all_results['noise'].values())
        min_noise = min(noise_scores)
        avg_noise = np.mean(noise_scores)
        scores.append(min_noise * 0.5 + avg_noise * 0.5)  # バランス
    
    if scores:
        return np.mean(scores)
    else:
        return 0.0

def main():
    """メイン実行関数"""
    
    print("🛡️ NKAT Stage3 最適化ロバストネステスト 開始")
    print("="*60)
    
    # RTX3080 CUDA設定確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"🚀 RTX3080 CUDA Optimization Enabled: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
    else:
        print("⚠️ CUDA not available, using CPU")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # データセット設定
    dataset_name = 'MNIST'
    
    # テスト用データローダー
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # サブセット作成（時間短縮）
    test_indices = torch.randperm(len(test_dataset))[:1000]
    test_subset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0)
    
    # モデル読み込み
    model = NKATTransformerPractical(
        img_size=28,
        patch_size=4,
        num_classes=10,
        embed_dim=512,
        depth=8,
        num_heads=8,
        nkat_strength=0.015
    ).to(device)
    
    # ベストチェックポイント読み込み
    checkpoint_path = "checkpoints/nkat_enhanced_v2_best.pth"
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"✅ ベストチェックポイント読み込み完了: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ チェックポイント読み込みエラー: {e}")
    else:
        print(f"⚠️ チェックポイントが見つかりません: {checkpoint_path}")
    
    # 堅牢性解析器作成
    analyzer = NKATRobustnessAnalyzer(model, device, dataset_name)
    
    # 各種堅牢性テスト実行
    all_results = {}
    
    try:
        # 1. 敵対的攻撃テスト
        print("\n🛡️ 敵対的攻撃テスト開始...")
        all_results['adversarial'] = analyzer.test_adversarial_robustness(test_loader)
        
        # 2. 回転堅牢性テスト
        print("\n🔄 回転堅牢性テスト開始...")
        all_results['rotation'] = analyzer.test_rotation_robustness(test_loader)
        
        # 3. ノイズ堅牢性テスト
        print("\n🔊 ノイズ堅牢性テスト開始...")
        all_results['noise'] = analyzer.test_noise_robustness(test_loader)
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
    
    # 結果分析
    robustness_score = calculate_robustness_score(all_results)
    
    # 結果保存
    results_data = {
        'timestamp': timestamp,
        'dataset': dataset_name,
        'device': str(device),
        'robustness_score': robustness_score,
        'test_results': all_results,
        'checkpoint_path': checkpoint_path
    }
    
    results_path = f"logs/nkat_stage3_robustness_results_{timestamp}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # 可視化作成
    viz_path = create_robustness_visualization(all_results, dataset_name, timestamp)
    
    # 結果サマリー表示
    print("\n" + "="*60)
    print("🛡️ STAGE3 堅牢性テスト結果サマリー")
    print("="*60)
    print(f"🎯 総合堅牢性スコア: {robustness_score:.2f}%")
    
    if 'adversarial' in all_results:
        min_adv = min(all_results['adversarial'].values())
        print(f"⚔️  最小敵対的精度: {min_adv:.2f}%")
    
    if 'rotation' in all_results:
        min_rot = min(all_results['rotation'].values())
        print(f"🔄 最小回転精度: {min_rot:.2f}%")
    
    if 'noise' in all_results:
        min_noise = min(all_results['noise'].values())
        print(f"🔊 最小ノイズ精度: {min_noise:.2f}%")
    
    print(f"📊 結果保存: {results_path}")
    print(f"📈 可視化保存: {viz_path}")
    print("="*60)

if __name__ == "__main__":
    main() 