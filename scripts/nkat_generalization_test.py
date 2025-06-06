#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer Generalization Test
拡張データセットでの汎化性能検証

Datasets:
- EMNIST (Balanced, 47 classes)
- Fashion-MNIST (10 classes)
- CIFAR-10 (10 classes, RGB)

Target: ② フェーズの汎化性テスト
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from tqdm import tqdm
from torchvision.datasets import EMNIST
import warnings
warnings.filterwarnings('ignore')

# 英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

class GeneralizationConfig:
    """Configuration for generalization testing"""
    
    def __init__(self, dataset_name='mnist'):
        self.dataset_name = dataset_name
        
        # Dataset-specific settings
        if dataset_name == 'emnist':
            self.image_size = 28
            self.input_channels = 1
            self.num_classes = 47
            self.patch_size = 7
        elif dataset_name == 'fashion_mnist':
            self.image_size = 28
            self.input_channels = 1
            self.num_classes = 10
            self.patch_size = 7
        elif dataset_name == 'cifar10':
            self.image_size = 32
            self.input_channels = 3
            self.num_classes = 10
            self.patch_size = 4  # Smaller patches for 32x32
        else:  # mnist
            self.image_size = 28
            self.input_channels = 1
            self.num_classes = 10
            self.patch_size = 7
        
        # Model architecture
        self.hidden_dim = 384
        self.num_layers = 6
        self.num_heads = 6
        self.mlp_ratio = 4.0
        self.dropout = 0.1
        
        # Training settings
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.num_epochs = 50  # Moderate for generalization test
        self.weight_decay = 1e-4
        
        # For reproducibility
        self.seed = 1337

class NKATGeneralizationModel(nn.Module):
    """NKAT model adapted for different datasets"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Adaptive patch embedding for different input channels
        self.patch_embed = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Calculate number of patches
        num_patches = (config.image_size // config.patch_size) ** 2
        
        # NKAT Enhancement 1: Gauge-equivariant positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_dim) * 0.02
        )
        self.gauge_rotation = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Class token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02
        )
        
        # NKAT Enhancement 2: Learnable gauge parameters
        self.gauge_theta = nn.Parameter(torch.zeros(config.hidden_dim))
        
        # Enhanced transformer blocks
        self.transformer = nn.ModuleList([
            NKATGeneralizationBlock(config) for _ in range(config.num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.num_classes)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Channel adaptation for RGB inputs
        if config.input_channels == 3:
            self.channel_mixer = nn.Conv2d(3, 3, kernel_size=1)
            print("✅ RGB channel adaptation enabled")
        else:
            self.channel_mixer = None
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # RGB channel adaptation if needed
        if self.channel_mixer is not None:
            x = self.channel_mixer(x)
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, hidden_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # NKAT: Gauge-equivariant positional encoding
        pos_enhanced = self.gauge_rotation(self.pos_embed)
        x = x + pos_enhanced
        x = self.dropout(x)
        
        # Enhanced transformer layers
        for layer in self.transformer:
            x = layer(x, self.gauge_theta)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits

class NKATGeneralizationBlock(nn.Module):
    """Enhanced transformer block with NKAT features"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = GaugeEquivariantAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = NonCommutativeMLP(config)
        
        # NKAT Enhancement 4: Enhanced residual connections
        self.residual_alpha = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x, gauge_theta):
        # Enhanced attention with gauge parameter
        attn_out = self.attn(self.norm1(x), gauge_theta)
        x = x + self.residual_alpha * attn_out
        
        # Non-commutative MLP
        mlp_out = self.mlp(self.norm2(x), gauge_theta)
        x = x + self.residual_alpha * mlp_out
        
        return x

class GaugeEquivariantAttention(nn.Module):
    """Gauge equivariant multi-head attention"""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.qkv = nn.Linear(config.hidden_dim, config.hidden_dim * 3)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Gauge transformation
        self.gauge_transform = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def forward(self, x, gauge_theta):
        B, N, C = x.shape
        
        # Apply gauge transformation
        x_gauge = self.gauge_transform(x) * gauge_theta.unsqueeze(0).unsqueeze(0)
        x = x + 0.1 * x_gauge  # Small gauge correction
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard attention computation
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class NonCommutativeMLP(nn.Module):
    """Non-commutative MLP with gauge parameter"""
    
    def __init__(self, config):
        super().__init__()
        hidden_features = int(config.hidden_dim * config.mlp_ratio)
        
        self.fc1 = nn.Linear(config.hidden_dim, hidden_features)
        self.fc2 = nn.Linear(hidden_features, config.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        
        # Non-commutative weights
        self.nc_weight1 = nn.Parameter(torch.ones(hidden_features) * 0.1)
        self.nc_weight2 = nn.Parameter(torch.ones(config.hidden_dim) * 0.1)
        
    def forward(self, x, gauge_theta):
        # First transformation with non-commutativity
        x = self.fc1(x)
        x = x * self.nc_weight1.unsqueeze(0).unsqueeze(0)
        x = self.act(x)
        x = self.dropout(x)
        
        # Second transformation
        x = self.fc2(x)
        x = x * self.nc_weight2.unsqueeze(0).unsqueeze(0)
        x = self.dropout(x)
        
        return x

class BaselineViTGeneralization(nn.Module):
    """Baseline ViT for comparison"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Standard patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Standard positional embedding
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_dim) * 0.02
        )
        
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02
        )
        
        # Standard transformer blocks
        self.transformer = nn.ModuleList([
            StandardBlock(config) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer layers
        for layer in self.transformer:
            x = layer(x)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits

class StandardBlock(nn.Module):
    """Standard transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = StandardAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = StandardMLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class StandardAttention(nn.Module):
    """Standard multi-head attention"""
    
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

class StandardMLP(nn.Module):
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

class GeneralizationTester:
    """Comprehensive generalization testing framework"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        print(f"🚀 NKAT Generalization Testing")
        print(f"Device: {self.device}")
        print(f"Testing on multiple datasets for generalization")
        
        # Datasets to test
        self.datasets = ['mnist', 'fashion_mnist', 'emnist', 'cifar10']
        
    def get_dataloaders(self, dataset_name, config):
        """Get dataloaders for specified dataset"""
        
        if dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = torchvision.datasets.MNIST(
                root="data", train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root="data", train=False, download=True, transform=transform
            )
            
        elif dataset_name == 'fashion_mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
            
            train_dataset = torchvision.datasets.FashionMNIST(
                root="data", train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.FashionMNIST(
                root="data", train=False, download=True, transform=transform
            )
            
        elif dataset_name == 'emnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1751,), (0.3332,))
            ])
            
            train_dataset = EMNIST(
                root="data", split='balanced', train=True, download=True, transform=transform
            )
            test_dataset = EMNIST(
                root="data", split='balanced', train=False, download=True, transform=transform
            )
            
        elif dataset_name == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            train_dataset = torchvision.datasets.CIFAR10(
                root="data", train=True, download=True, transform=transform_train
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root="data", train=False, download=True, transform=transform_test
            )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False
        )
        
        return train_loader, test_loader
    
    def train_model(self, model, train_loader, config, model_name, dataset_name):
        """Train a model on the given dataset"""
        print(f"\n📈 Training {model_name} on {dataset_name.upper()}...")
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs
        )
        
        model.train()
        for epoch in tqdm(range(config.num_epochs), desc=f"Training {model_name}"):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                train_acc = 100.0 * correct / total
                print(f"Epoch {epoch+1}: Acc={train_acc:.2f}%")
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    def test_on_dataset(self, dataset_name):
        """Test both models on a specific dataset"""
        print(f"\n{'='*60}")
        print(f"🔬 Testing on {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Configuration for this dataset
        config = GeneralizationConfig(dataset_name)
        
        # Get data loaders
        train_loader, test_loader = self.get_dataloaders(dataset_name, config)
        
        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Test baseline ViT
        baseline_model = BaselineViTGeneralization(config).to(self.device)
        self.train_model(baseline_model, train_loader, config, "Baseline ViT", dataset_name)
        baseline_acc = self.evaluate_model(baseline_model, test_loader)
        
        # Test NKAT-Transformer
        nkat_model = NKATGeneralizationModel(config).to(self.device)
        self.train_model(nkat_model, train_loader, config, "NKAT-Transformer", dataset_name)
        nkat_acc = self.evaluate_model(nkat_model, test_loader)
        
        # Store results
        self.results[dataset_name] = {
            'baseline_vit_acc': baseline_acc,
            'nkat_acc': nkat_acc,
            'improvement': nkat_acc - baseline_acc,
            'num_classes': config.num_classes,
            'image_size': config.image_size,
            'input_channels': config.input_channels
        }
        
        print(f"\n📊 Results on {dataset_name.upper()}:")
        print(f"Baseline ViT: {baseline_acc:.2f}%")
        print(f"NKAT-Transformer: {nkat_acc:.2f}%")
        print(f"Improvement: {nkat_acc - baseline_acc:+.2f}%")
        
        return baseline_acc, nkat_acc
    
    def run_generalization_study(self):
        """Run comprehensive generalization study"""
        print("🚀 Starting Generalization Study")
        print("Testing NKAT-Transformer on multiple datasets")
        
        for dataset_name in self.datasets:
            try:
                self.test_on_dataset(dataset_name)
            except Exception as e:
                print(f"❌ Error testing {dataset_name}: {str(e)}")
                continue
        
        self.analyze_and_visualize()
    
    def analyze_and_visualize(self):
        """Analyze results and create visualization"""
        if not self.results:
            return
        
        print(f"\n{'='*60}")
        print("📊 GENERALIZATION STUDY SUMMARY")
        print(f"{'='*60}")
        
        # Summary table
        print(f"{'Dataset':<15} {'Baseline':<10} {'NKAT':<10} {'Improvement':<12} {'Classes':<8}")
        print("-" * 60)
        
        total_improvement = 0
        for dataset, result in self.results.items():
            print(f"{dataset.upper():<15} {result['baseline_vit_acc']:<10.2f} "
                  f"{result['nkat_acc']:<10.2f} {result['improvement']:<12.2f} "
                  f"{result['num_classes']:<8}")
            total_improvement += result['improvement']
        
        avg_improvement = total_improvement / len(self.results)
        print(f"\nAverage improvement: {avg_improvement:.2f}%")
        
        # Create visualization
        self.create_visualization()
    
    def create_visualization(self):
        """Create generalization visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        datasets = list(self.results.keys())
        baseline_accs = [self.results[d]['baseline_vit_acc'] for d in datasets]
        nkat_accs = [self.results[d]['nkat_acc'] for d in datasets]
        improvements = [self.results[d]['improvement'] for d in datasets]
        
        # Accuracy comparison
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_accs, width, label='Baseline ViT', alpha=0.8)
        bars2 = ax1.bar(x + width/2, nkat_accs, width, label='NKAT-Transformer', alpha=0.8)
        
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('Generalization Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.upper() for d in datasets])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars1, baseline_accs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
        
        for bar, acc in zip(bars2, nkat_accs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Improvement bar chart
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars3 = ax2.bar(datasets, improvements, color=colors, alpha=0.7)
        
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Accuracy Improvement (%)')
        ax2.set_title('NKAT-Transformer Improvement over Baseline')
        ax2.set_xticklabels([d.upper() for d in datasets])
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars3, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{imp:+.2f}%', ha='center', va='bottom' if imp > 0 else 'top')
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'nkat_generalization_study_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved as {filename}")
        
        # Save results
        with open(f'nkat_generalization_results_{timestamp}.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Results saved as nkat_generalization_results_{timestamp}.json")

def main():
    """Run generalization study"""
    tester = GeneralizationTester()
    tester.run_generalization_study()

if __name__ == "__main__":
    main() 