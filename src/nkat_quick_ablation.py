#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer Quick Ablation Study
99%達成の各要素寄与度を高速検証

Target: 論文Figure 2生成 (短時間版)
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
import warnings
warnings.filterwarnings('ignore')

# 英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

class QuickConfig:
    """Quick ablation configuration"""
    
    def __init__(self):
        # Model architecture
        self.image_size = 28
        self.patch_size = 7
        self.input_channels = 1
        self.hidden_dim = 192  # Reduced for speed
        self.num_layers = 4    # Reduced for speed
        self.num_heads = 4     # Reduced for speed
        self.mlp_ratio = 4.0
        self.num_classes = 10
        self.dropout = 0.1
        
        # Training (快速版)
        self.batch_size = 128
        self.learning_rate = 2e-4
        self.num_epochs = 20   # 大幅短縮
        self.weight_decay = 1e-4
        
        # For reproducibility
        self.seed = 1337

class BaselineViTQuick(nn.Module):
    """Lightweight baseline ViT for quick testing"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Positional embedding
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_dim) * 0.02
        )
        
        # Class token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02
        )
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            QuickTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Classification head
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

class QuickTransformerBlock(nn.Module):
    """Quick transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = QuickMultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = QuickMLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class QuickMultiHeadAttention(nn.Module):
    """Quick multi-head attention"""
    
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

class QuickMLP(nn.Module):
    """Quick MLP"""
    
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

class EnhancedViTQuick(nn.Module):
    """NKAT拡張版（簡略実装）"""
    
    def __init__(self, config, enhancements=None):
        super().__init__()
        self.config = config
        self.enhancements = enhancements or ['all']
        
        # Basic ViT structure
        self.patch_embed = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        num_patches = (config.image_size // config.patch_size) ** 2
        
        # Enhancement 1: Gauge parameter
        if 'gauge_param' in self.enhancements or 'all' in self.enhancements:
            self.gauge_theta = nn.Parameter(torch.zeros(config.hidden_dim))
            print("✅ Gauge parameter enabled")
        else:
            self.gauge_theta = None
            print("❌ Gauge parameter disabled")
        
        # Enhancement 2: Gauge equivariant positional encoding
        if 'gauge_pos' in self.enhancements or 'all' in self.enhancements:
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches + 1, config.hidden_dim) * 0.02
            )
            self.pos_rotation = nn.Linear(config.hidden_dim, config.hidden_dim)
            print("✅ Gauge equivariant positional encoding enabled")
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches + 1, config.hidden_dim) * 0.02
            )
            self.pos_rotation = None
            print("❌ Standard positional encoding")
        
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02
        )
        
        # Transformer blocks with enhancements
        self.transformer = nn.ModuleList([
            EnhancedQuickBlock(config, self.enhancements) 
            for _ in range(config.num_layers)
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
        
        # Enhanced positional encoding
        if self.pos_rotation is not None:
            pos_enhanced = self.pos_rotation(self.pos_embed)
            x = x + pos_enhanced
        else:
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

class EnhancedQuickBlock(nn.Module):
    """Enhanced transformer block with NKAT features"""
    
    def __init__(self, config, enhancements):
        super().__init__()
        self.enhancements = enhancements
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = QuickMultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        # Enhancement 3: Non-commutative MLP
        if 'non_commutative' in enhancements or 'all' in enhancements:
            self.mlp = NonCommutativeQuickMLP(config)
        else:
            self.mlp = QuickMLP(config)
        
        # Enhancement 4: Enhanced residual connections
        if 'enhanced_residual' in enhancements or 'all' in enhancements:
            self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)
        else:
            self.residual_weight = None
        
    def forward(self, x):
        # Standard attention
        attn_out = self.attn(self.norm1(x))
        
        # Enhanced residual for attention
        if self.residual_weight is not None:
            x = x + self.residual_weight * attn_out
        else:
            x = x + attn_out
        
        # MLP with residual
        mlp_out = self.mlp(self.norm2(x))
        
        if self.residual_weight is not None:
            x = x + self.residual_weight * mlp_out
        else:
            x = x + mlp_out
        
        return x

class NonCommutativeQuickMLP(nn.Module):
    """Non-commutative MLP (simplified)"""
    
    def __init__(self, config):
        super().__init__()
        hidden_features = int(config.hidden_dim * config.mlp_ratio)
        
        self.fc1 = nn.Linear(config.hidden_dim, hidden_features)
        self.fc2 = nn.Linear(hidden_features, config.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        
        # Non-commutative parameter
        self.nc_weight = nn.Parameter(torch.ones(hidden_features) * 0.1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        
        # Apply non-commutative transformation
        x = x * self.nc_weight.unsqueeze(0).unsqueeze(0)
        
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class QuickAblationStudy:
    """Quick ablation study for demonstration"""
    
    def __init__(self):
        self.config = QuickConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        print(f"🔬 NKAT Quick Ablation Study")
        print(f"Device: {self.device}")
        print(f"Configuration: Lightweight for fast demonstration")
        
        # Prepare data
        self.train_loader, self.test_loader = self._prepare_data()
        
        # Define model variants
        self.model_variants = {
            'Baseline ViT': (BaselineViTQuick, None),
            'NKAT - Gauge Param': (EnhancedViTQuick, ['gauge_pos', 'non_commutative', 'enhanced_residual']),
            'NKAT - Gauge Pos': (EnhancedViTQuick, ['gauge_param', 'non_commutative', 'enhanced_residual']),
            'NKAT - Non-Commutative': (EnhancedViTQuick, ['gauge_param', 'gauge_pos', 'enhanced_residual']),
            'NKAT - Enhanced Residual': (EnhancedViTQuick, ['gauge_param', 'gauge_pos', 'non_commutative']),
            'Full NKAT': (EnhancedViTQuick, ['all'])
        }
        
        self.results = {}
    
    def _prepare_data(self):
        """Prepare MNIST data"""
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
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        return train_loader, test_loader
    
    def train_model(self, model_class, enhancements, model_name):
        """Train a single model variant"""
        print(f"\n📈 Training {model_name}...")
        
        # Initialize model
        if enhancements is None:
            model = model_class(self.config).to(self.device)
        else:
            model = model_class(self.config, enhancements).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
        
        # Optimizer and criterion
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )
        
        # Training loop
        model.train()
        for epoch in tqdm(range(self.config.num_epochs), desc=f"Training {model_name}"):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
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
            
            if (epoch + 1) % 5 == 0:
                train_acc = 100.0 * correct / total
                print(f"Epoch {epoch+1}: Acc={train_acc:.2f}%")
        
        # Evaluation
        test_acc = self.evaluate_model(model)
        
        return {
            'model_name': model_name,
            'total_params': total_params,
            'test_acc': test_acc
        }
    
    def evaluate_model(self, model):
        """Evaluate model on test set"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_acc = 100.0 * correct / total
        return test_acc
    
    def run_study(self):
        """Run the quick ablation study"""
        print("🚀 Starting Quick Ablation Study")
        print("=" * 50)
        
        for model_name, (model_class, enhancements) in self.model_variants.items():
            try:
                result = self.train_model(model_class, enhancements, model_name)
                self.results[model_name] = result
                
                print(f"✅ {model_name}: {result['test_acc']:.2f}% accuracy")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        self.analyze_and_visualize()
    
    def analyze_and_visualize(self):
        """Analyze results and create visualization"""
        print("\n" + "=" * 50)
        print("📊 QUICK ABLATION RESULTS")
        print("=" * 50)
        
        # Sort by accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['test_acc'],
            reverse=True
        )
        
        print(f"{'Model':<25} {'Test Acc':<10} {'Params':<10}")
        print("-" * 45)
        
        for name, result in sorted_results:
            print(f"{name:<25} {result['test_acc']:<10.2f}% {result['total_params']:<10,}")
        
        # Calculate contributions
        if 'Full NKAT' in self.results and 'Baseline ViT' in self.results:
            baseline_acc = self.results['Baseline ViT']['test_acc']
            full_nkat_acc = self.results['Full NKAT']['test_acc']
            
            print(f"\n🎯 Component Contributions (estimated):")
            print("-" * 40)
            
            for name, result in self.results.items():
                if 'NKAT -' in name:
                    component = name.replace('NKAT - ', '')
                    # Contribution = Full NKAT - (NKAT without component)
                    contribution = full_nkat_acc - result['test_acc']
                    print(f"{component:<20}: {contribution:>+.2f}%")
            
            total_improvement = full_nkat_acc - baseline_acc
            print(f"\nTotal NKAT improvement: +{total_improvement:.2f}%")
        
        # Create visualization
        self.create_visualization()
    
    def create_visualization(self):
        """Create quick visualization"""
        if not self.results:
            return
        
        # Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        names = list(self.results.keys())
        accs = [result['test_acc'] for result in self.results.values()]
        
        bars = ax1.bar(range(len(names)), accs, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(names))))
        ax1.set_xlabel('Model Variant')
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('NKAT Quick Ablation: Accuracy Comparison')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # Contribution analysis (if available)
        if 'Full NKAT' in self.results and 'Baseline ViT' in self.results:
            full_nkat_acc = self.results['Full NKAT']['test_acc']
            
            contributions = {}
            for name, result in self.results.items():
                if 'NKAT -' in name:
                    component = name.replace('NKAT - ', '')
                    contribution = full_nkat_acc - result['test_acc']
                    contributions[component] = contribution
            
            if contributions:
                comp_names = list(contributions.keys())
                comp_values = list(contributions.values())
                
                bars2 = ax2.bar(range(len(comp_names)), comp_values,
                               color=plt.cm.plasma(np.linspace(0, 1, len(comp_names))))
                ax2.set_xlabel('NKAT Component')
                ax2.set_ylabel('Contribution to Accuracy (%)')
                ax2.set_title('Individual Component Contributions')
                ax2.set_xticks(range(len(comp_names)))
                ax2.set_xticklabels(comp_names, rotation=45, ha='right')
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars2, comp_values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{val:+.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'nkat_quick_ablation_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved as {filename}")
        
        # Save results
        with open(f'nkat_quick_ablation_results_{timestamp}.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Results saved as nkat_quick_ablation_results_{timestamp}.json")

def main():
    """Run quick ablation study"""
    study = QuickAblationStudy()
    study.run_study()

if __name__ == "__main__":
    main() 