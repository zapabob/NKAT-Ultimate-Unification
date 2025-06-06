#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer Robustness & Security Test
敵対的攻撃、回転、画像破損に対する頑健性検証

Tests:
1. Adversarial FGSM (ε=0.1) → target: 70%+ accuracy
2. Rotation (±30°) → target: <5pt accuracy drop
3. JPEG compression (Q=20) → target: <3pt accuracy drop
4. Gaussian noise → robustness evaluation

Target: ③ フェーズの Robustness & Security
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from tqdm import tqdm
from PIL import Image, ImageFilter
import io
import warnings
warnings.filterwarnings('ignore')

# 英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

class RobustnessConfig:
    """Configuration for robustness testing"""
    
    def __init__(self):
        # Model architecture (matching NKAT)
        self.image_size = 28
        self.patch_size = 7
        self.input_channels = 1
        self.hidden_dim = 384
        self.num_layers = 6
        self.num_heads = 6
        self.mlp_ratio = 4.0
        self.num_classes = 10
        self.dropout = 0.1
        
        # Robustness test settings
        self.batch_size = 256  # Larger for efficiency
        self.test_subset_size = 2000  # Subset for faster testing
        
        # Attack parameters
        self.fgsm_epsilon = 0.1  # Target: 70%+ accuracy under attack
        self.rotation_angles = [-30, -15, 0, 15, 30]  # Target: <5pt drop
        self.jpeg_quality = 20  # Target: <3pt drop
        self.noise_std = [0.05, 0.1, 0.15, 0.2]  # Various noise levels
        
        # For reproducibility
        self.seed = 1337

# FGSM Attack Implementation
def fgsm_attack(model, data, target, epsilon):
    """Fast Gradient Sign Method attack"""
    data.requires_grad = True
    
    output = model(data)
    loss = F.cross_entropy(output, target)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = data.grad.data
    sign_data_grad = data_grad.sign()
    
    perturbed_data = data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data

# Image corruption functions
class ImageCorruptions:
    """Image corruption utilities"""
    
    @staticmethod
    def apply_rotation(images, angle):
        """Apply rotation to batch of images"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=[angle, angle], fill=0),
            transforms.ToTensor()
        ])
        
        rotated_images = torch.stack([
            transform(img.cpu()) for img in images
        ])
        
        return rotated_images.to(images.device)
    
    @staticmethod
    def apply_jpeg_compression(images, quality):
        """Apply JPEG compression to batch of images"""
        compressed_images = []
        
        for img in images:
            # Convert to PIL Image
            img_np = (img.cpu().numpy().squeeze() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode='L')
            
            # Apply JPEG compression
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            compressed_pil = Image.open(buffer)
            
            # Convert back to tensor
            img_tensor = transforms.ToTensor()(compressed_pil)
            compressed_images.append(img_tensor)
        
        return torch.stack(compressed_images).to(images.device)
    
    @staticmethod
    def add_gaussian_noise(images, std):
        """Add Gaussian noise to images"""
        noise = torch.randn_like(images) * std
        noisy_images = torch.clamp(images + noise, 0, 1)
        return noisy_images

class RobustNKATModel(nn.Module):
    """NKAT model with robustness enhancements"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Patch embedding with robustness
        self.patch_embed = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Calculate number of patches
        num_patches = (config.image_size // config.patch_size) ** 2
        
        # Enhanced positional encoding (gauge equivariant)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_dim) * 0.02
        )
        self.gauge_rotation = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Class token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02
        )
        
        # Gauge parameter for robustness
        self.gauge_theta = nn.Parameter(torch.zeros(config.hidden_dim))
        
        # Robust transformer blocks
        self.transformer = nn.ModuleList([
            RobustNKATBlock(config) for _ in range(config.num_layers)
        ])
        
        # Classification head with dropout for robustness
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.num_classes)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Robustness-specific components
        self.input_noise_layer = nn.Parameter(torch.zeros(1))  # Learnable noise resistance
        
    def forward(self, x, training=True):
        B, C, H, W = x.shape
        
        # Input robustness layer
        if training:
            x = x + self.input_noise_layer * torch.randn_like(x) * 0.01
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Gauge-equivariant positional encoding (rotation robust)
        pos_enhanced = self.gauge_rotation(self.pos_embed)
        x = x + pos_enhanced
        x = self.dropout(x)
        
        # Robust transformer layers
        for layer in self.transformer:
            x = layer(x, self.gauge_theta)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits

class RobustNKATBlock(nn.Module):
    """Robust NKAT transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = RobustAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = RobustMLP(config)
        
        # Robust residual connections
        self.residual_alpha = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x, gauge_theta):
        # Robust attention
        attn_out = self.attn(self.norm1(x), gauge_theta)
        x = x + self.residual_alpha * attn_out
        
        # Robust MLP
        mlp_out = self.mlp(self.norm2(x), gauge_theta)
        x = x + self.residual_alpha * mlp_out
        
        return x

class RobustAttention(nn.Module):
    """Robust multi-head attention with gauge equivariance"""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.qkv = nn.Linear(config.hidden_dim, config.hidden_dim * 3)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Gauge transformation for robustness
        self.gauge_transform = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Attention noise resistance
        self.attention_noise_scale = nn.Parameter(torch.ones(1) * 0.01)
        
    def forward(self, x, gauge_theta):
        B, N, C = x.shape
        
        # Apply gauge transformation (rotation equivariance)
        x_gauge = self.gauge_transform(x) * gauge_theta.unsqueeze(0).unsqueeze(0)
        x = x + 0.1 * x_gauge
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Robust attention computation
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        # Add small noise to attention for robustness during training
        if self.training:
            attn = attn + self.attention_noise_scale * torch.randn_like(attn)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class RobustMLP(nn.Module):
    """Robust MLP with non-commutative operations"""
    
    def __init__(self, config):
        super().__init__()
        hidden_features = int(config.hidden_dim * config.mlp_ratio)
        
        self.fc1 = nn.Linear(config.hidden_dim, hidden_features)
        self.fc2 = nn.Linear(hidden_features, config.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        
        # Non-commutative weights for robustness
        self.nc_weight1 = nn.Parameter(torch.ones(hidden_features) * 0.1)
        self.nc_weight2 = nn.Parameter(torch.ones(config.hidden_dim) * 0.1)
        
    def forward(self, x, gauge_theta):
        x = self.fc1(x)
        x = x * self.nc_weight1.unsqueeze(0).unsqueeze(0)
        x = self.act(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x * self.nc_weight2.unsqueeze(0).unsqueeze(0)
        x = self.dropout(x)
        
        return x

class BaselineRobustModel(nn.Module):
    """Baseline model for comparison"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Standard ViT components
        self.patch_embed = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_dim) * 0.02
        )
        
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02
        )
        
        # Standard transformer blocks
        self.transformer = nn.ModuleList([
            BaselineBlock(config) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, training=True):
        B, C, H, W = x.shape
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for layer in self.transformer:
            x = layer(x)
        
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits

class BaselineBlock(nn.Module):
    """Standard transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = BaselineAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = BaselineMLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class BaselineAttention(nn.Module):
    """Standard attention"""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.qkv = nn.Linear(config.hidden_dim, config.hidden_dim * 3)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class BaselineMLP(nn.Module):
    """Standard MLP"""
    
    def __init__(self, config):
        super().__init__()
        hidden_features = int(config.hidden_dim * config.mlp_ratio)
        
        self.fc1 = nn.Linear(config.hidden_dim, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class RobustnessTester:
    """Comprehensive robustness testing framework"""
    
    def __init__(self):
        self.config = RobustnessConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        print(f"🛡️ NKAT Robustness & Security Testing")
        print(f"Device: {self.device}")
        print(f"Test subset size: {self.config.test_subset_size}")
        
        # Prepare test data
        self.test_loader = self._prepare_test_data()
        
        # Initialize corruptions
        self.corruptions = ImageCorruptions()
        
    def _prepare_test_data(self):
        """Prepare test dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = torchvision.datasets.MNIST(
            root="data", train=False, download=True, transform=transform
        )
        
        # Use subset for faster testing
        subset_indices = torch.randperm(len(test_dataset))[:self.config.test_subset_size]
        test_subset = Subset(test_dataset, subset_indices)
        
        test_loader = DataLoader(
            test_subset, batch_size=self.config.batch_size, shuffle=False
        )
        
        return test_loader
    
    def load_pretrained_models(self):
        """Load or create pretrained models"""
        # For demonstration, we'll create new models
        # In practice, load from checkpoints
        
        print("📦 Loading models...")
        
        # NKAT model
        nkat_model = RobustNKATModel(self.config).to(self.device)
        nkat_model.eval()
        
        # Baseline model
        baseline_model = BaselineRobustModel(self.config).to(self.device)
        baseline_model.eval()
        
        # Quick training on clean data for demonstration
        print("🔧 Quick training for demonstration...")
        self._quick_train(nkat_model, "NKAT")
        self._quick_train(baseline_model, "Baseline")
        
        return nkat_model, baseline_model
    
    def _quick_train(self, model, model_name):
        """Quick training for demonstration"""
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root="data", train=True, download=True, transform=train_transform
        )
        
        # Small subset for quick training
        train_subset = Subset(train_dataset, torch.randperm(len(train_dataset))[:5000])
        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(3):  # Very quick training
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx > 20:  # Limit batches
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        model.eval()
        print(f"✅ {model_name} quick training completed")
    
    def test_clean_accuracy(self, model, model_name):
        """Test clean accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data, training=False)
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100.0 * correct / total
        print(f"🧪 {model_name} clean accuracy: {accuracy:.2f}%")
        return accuracy
    
    def test_fgsm_robustness(self, model, model_name):
        """Test FGSM adversarial robustness"""
        print(f"\n⚔️ Testing FGSM robustness for {model_name} (ε={self.config.fgsm_epsilon})")
        
        model.eval()
        correct = 0
        total = 0
        
        for data, target in tqdm(self.test_loader, desc="FGSM Attack"):
            data, target = data.to(self.device), target.to(self.device)
            
            # Generate adversarial examples
            perturbed_data = fgsm_attack(model, data, target, self.config.fgsm_epsilon)
            
            # Test on adversarial examples
            with torch.no_grad():
                output = model(perturbed_data, training=False)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100.0 * correct / total
        print(f"🎯 {model_name} FGSM accuracy: {accuracy:.2f}%")
        return accuracy
    
    def test_rotation_robustness(self, model, model_name):
        """Test rotation robustness"""
        print(f"\n🔄 Testing rotation robustness for {model_name}")
        
        rotation_results = {}
        
        for angle in self.config.rotation_angles:
            correct = 0
            total = 0
            
            for data, target in tqdm(self.test_loader, desc=f"Rotation {angle}°"):
                data, target = data.to(self.device), target.to(self.device)
                
                # Apply rotation
                if angle != 0:
                    rotated_data = self.corruptions.apply_rotation(data, angle)
                else:
                    rotated_data = data
                
                # Test on rotated data
                with torch.no_grad():
                    output = model(rotated_data, training=False)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            accuracy = 100.0 * correct / total
            rotation_results[angle] = accuracy
            print(f"📐 {model_name} at {angle}°: {accuracy:.2f}%")
        
        return rotation_results
    
    def test_jpeg_robustness(self, model, model_name):
        """Test JPEG compression robustness"""
        print(f"\n📷 Testing JPEG compression robustness for {model_name} (Q={self.config.jpeg_quality})")
        
        correct = 0
        total = 0
        
        for data, target in tqdm(self.test_loader, desc="JPEG Compression"):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply JPEG compression
            compressed_data = self.corruptions.apply_jpeg_compression(data, self.config.jpeg_quality)
            
            # Test on compressed data
            with torch.no_grad():
                output = model(compressed_data, training=False)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100.0 * correct / total
        print(f"🗜️ {model_name} JPEG Q{self.config.jpeg_quality}: {accuracy:.2f}%")
        return accuracy
    
    def test_noise_robustness(self, model, model_name):
        """Test Gaussian noise robustness"""
        print(f"\n🔊 Testing noise robustness for {model_name}")
        
        noise_results = {}
        
        for std in self.config.noise_std:
            correct = 0
            total = 0
            
            for data, target in tqdm(self.test_loader, desc=f"Noise σ={std}"):
                data, target = data.to(self.device), target.to(self.device)
                
                # Add Gaussian noise
                noisy_data = self.corruptions.add_gaussian_noise(data, std)
                
                # Test on noisy data
                with torch.no_grad():
                    output = model(noisy_data, training=False)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            accuracy = 100.0 * correct / total
            noise_results[std] = accuracy
            print(f"📡 {model_name} noise σ={std}: {accuracy:.2f}%")
        
        return noise_results
    
    def run_robustness_study(self):
        """Run comprehensive robustness study"""
        print("🚀 Starting Robustness Study")
        print("=" * 60)
        
        # Load models
        nkat_model, baseline_model = self.load_pretrained_models()
        
        models = {
            'NKAT-Transformer': nkat_model,
            'Baseline ViT': baseline_model
        }
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"🔬 Testing {model_name}")
            print(f"{'='*60}")
            
            # Clean accuracy
            clean_acc = self.test_clean_accuracy(model, model_name)
            
            # FGSM robustness
            fgsm_acc = self.test_fgsm_robustness(model, model_name)
            
            # Rotation robustness
            rotation_results = self.test_rotation_robustness(model, model_name)
            
            # JPEG robustness
            jpeg_acc = self.test_jpeg_robustness(model, model_name)
            
            # Noise robustness
            noise_results = self.test_noise_robustness(model, model_name)
            
            # Store results
            self.results[model_name] = {
                'clean_accuracy': clean_acc,
                'fgsm_accuracy': fgsm_acc,
                'fgsm_drop': clean_acc - fgsm_acc,
                'rotation_results': rotation_results,
                'rotation_drop': clean_acc - min(rotation_results.values()),
                'jpeg_accuracy': jpeg_acc,
                'jpeg_drop': clean_acc - jpeg_acc,
                'noise_results': noise_results
            }
        
        self.analyze_and_visualize()
    
    def analyze_and_visualize(self):
        """Analyze and visualize robustness results"""
        print(f"\n{'='*60}")
        print("📊 ROBUSTNESS STUDY SUMMARY")
        print(f"{'='*60}")
        
        # Summary table
        print(f"{'Model':<18} {'Clean':<8} {'FGSM':<8} {'Rot Drop':<10} {'JPEG Drop':<12}")
        print("-" * 60)
        
        for model_name, results in self.results.items():
            print(f"{model_name:<18} {results['clean_accuracy']:<8.2f} "
                  f"{results['fgsm_accuracy']:<8.2f} {results['rotation_drop']:<10.2f} "
                  f"{results['jpeg_drop']:<12.2f}")
        
        # Create visualizations
        self.create_visualizations()
    
    def create_visualizations(self):
        """Create robustness visualizations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.results.keys())
        
        # 1. Clean vs FGSM accuracy
        clean_accs = [self.results[m]['clean_accuracy'] for m in models]
        fgsm_accs = [self.results[m]['fgsm_accuracy'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, clean_accs, width, label='Clean', alpha=0.8)
        ax1.bar(x + width/2, fgsm_accs, width, label='FGSM Attack', alpha=0.8)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Clean vs FGSM Adversarial Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rotation robustness
        for model_name in models:
            angles = list(self.results[model_name]['rotation_results'].keys())
            accs = list(self.results[model_name]['rotation_results'].values())
            ax2.plot(angles, accs, marker='o', label=model_name)
        
        ax2.set_xlabel('Rotation Angle (degrees)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Rotation Robustness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy drops comparison
        fgsm_drops = [self.results[m]['fgsm_drop'] for m in models]
        rotation_drops = [self.results[m]['rotation_drop'] for m in models]
        jpeg_drops = [self.results[m]['jpeg_drop'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        ax3.bar(x - width, fgsm_drops, width, label='FGSM Drop', alpha=0.8)
        ax3.bar(x, rotation_drops, width, label='Rotation Drop', alpha=0.8)
        ax3.bar(x + width, jpeg_drops, width, label='JPEG Drop', alpha=0.8)
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Accuracy Drop (%)')
        ax3.set_title('Robustness: Accuracy Drops')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Noise robustness
        for model_name in models:
            stds = list(self.results[model_name]['noise_results'].keys())
            accs = list(self.results[model_name]['noise_results'].values())
            ax4.plot(stds, accs, marker='s', label=model_name)
        
        ax4.set_xlabel('Noise Standard Deviation')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Gaussian Noise Robustness')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'nkat_robustness_study_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved as {filename}")
        
        # Save results
        with open(f'nkat_robustness_results_{timestamp}.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Results saved as nkat_robustness_results_{timestamp}.json")

def main():
    """Run robustness study"""
    tester = RobustnessTester()
    tester.run_robustness_study()

if __name__ == "__main__":
    main() 