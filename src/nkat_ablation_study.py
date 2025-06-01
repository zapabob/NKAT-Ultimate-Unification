#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer Ablation Study
各拡張要素の寄与度定量化

Ablations:
1. Baseline ViT (no NKAT enhancements)
2. -θ(x) learnable gauge parameter  
3. -Non-commutative algebra
4. -Gauge equivariant positional encoding
5. -Residual connection modifications
6. Full NKAT (all enhancements)

Target: Figure 2 for paper submission
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
import seaborn as sns
import json
import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

class BaselineViT(nn.Module):
    """Standard Vision Transformer (no NKAT enhancements)"""
    
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
        
        # Class token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02
        )
        
        # Standard transformer blocks
        self.transformer = nn.ModuleList([
            StandardTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.num_classes)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, hidden_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]
        
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

class StandardTransformerBlock(nn.Module):
    """Standard transformer block (no NKAT modifications)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = StandardMultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = StandardMLP(config)
        
    def forward(self, x):
        # Standard residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class StandardMultiHeadAttention(nn.Module):
    """Standard multi-head attention (no gauge equivariance)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.qkv = nn.Linear(config.hidden_dim, config.hidden_dim * 3)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class StandardMLP(nn.Module):
    """Standard MLP (no non-commutative algebra)"""
    
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

# Partial NKAT implementations for ablation
class NKATMinusGaugeParam(nn.Module):
    """NKAT without learnable gauge parameter θ(x)"""
    
    def __init__(self, config):
        super().__init__()
        # Import from full NKAT but disable gauge parameter
        from nkat_enhanced_transformer_v2 import EnhancedNKATVisionTransformer
        self.full_nkat = EnhancedNKATVisionTransformer(config)
        
        # Disable gauge parameter learning
        for module in self.full_nkat.modules():
            if hasattr(module, 'gauge_learnable'):
                module.gauge_learnable = False
            if hasattr(module, 'theta_param'):
                module.theta_param.requires_grad = False
                
    def forward(self, x):
        return self.full_nkat(x)

class NKATMinusNonCommutative(nn.Module):
    """NKAT without non-commutative algebra"""
    
    def __init__(self, config):
        super().__init__()
        from nkat_enhanced_transformer_v2 import EnhancedNKATVisionTransformer
        self.full_nkat = EnhancedNKATVisionTransformer(config)
        
        # Replace non-commutative operations with standard ones
        for module in self.full_nkat.modules():
            if hasattr(module, 'use_non_commutative'):
                module.use_non_commutative = False
                
    def forward(self, x):
        return self.full_nkat(x)

class NKATMinusGaugeEquivariant(nn.Module):
    """NKAT without gauge equivariant positional encoding"""
    
    def __init__(self, config):
        super().__init__()
        from nkat_enhanced_transformer_v2 import EnhancedNKATVisionTransformer
        self.full_nkat = EnhancedNKATVisionTransformer(config)
        
        # Replace gauge equivariant pos encoding with standard
        num_patches = (config.image_size // config.patch_size) ** 2
        self.full_nkat.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_dim) * 0.02
        )
        
    def forward(self, x):
        return self.full_nkat(x)

class NKATMinusResidualMod(nn.Module):
    """NKAT without residual connection modifications"""
    
    def __init__(self, config):
        super().__init__()
        from nkat_enhanced_transformer_v2 import EnhancedNKATVisionTransformer
        self.full_nkat = EnhancedNKATVisionTransformer(config)
        
        # Replace enhanced residuals with standard ones
        for module in self.full_nkat.modules():
            if hasattr(module, 'use_enhanced_residual'):
                module.use_enhanced_residual = False
                
    def forward(self, x):
        return self.full_nkat(x)

class AblationConfig:
    """Configuration for ablation study"""
    
    def __init__(self):
        # Model architecture
        self.image_size = 28
        self.patch_size = 7
        self.input_channels = 1
        self.hidden_dim = 384
        self.num_layers = 6
        self.num_heads = 6
        self.mlp_ratio = 4.0
        self.num_classes = 10
        self.dropout = 0.1
        
        # Training
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.num_epochs = 50  # Reduced for ablation
        self.weight_decay = 1e-4
        
        # For reproducibility
        self.seed = 1337

class AblationStudy:
    """Comprehensive ablation study executor"""
    
    def __init__(self):
        self.config = AblationConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set seed for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        print(f"🔬 NKAT Ablation Study")
        print(f"Device: {self.device}")
        print(f"Seed: {self.config.seed}")
        
        # Prepare data
        self.train_loader, self.test_loader = self._prepare_data()
        
        # Models to test
        self.models = {
            'Baseline ViT': BaselineViT,
            'NKAT - Gauge Param': NKATMinusGaugeParam,
            'NKAT - Non-Commutative': NKATMinusNonCommutative,
            'NKAT - Gauge Equivariant': NKATMinusGaugeEquivariant,
            'NKAT - Residual Mod': NKATMinusResidualMod
        }
        
        # Add full NKAT
        from nkat_enhanced_transformer_v2 import EnhancedNKATVisionTransformer
        self.models['Full NKAT'] = EnhancedNKATVisionTransformer
        
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
    
    def train_model(self, model_class, model_name):
        """Train a single model variant"""
        print(f"\n📈 Training {model_name}...")
        
        # Initialize model
        model = model_class(self.config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        
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
        train_losses = []
        train_accs = []
        
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
            
            avg_loss = epoch_loss / len(self.train_loader)
            train_acc = 100.0 * correct / total
            
            train_losses.append(avg_loss)
            train_accs.append(train_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"Loss={avg_loss:.4f}, Acc={train_acc:.2f}%")
        
        # Evaluation
        test_acc, test_loss = self.evaluate_model(model)
        
        return {
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'final_train_acc': train_accs[-1],
            'test_acc': test_acc,
            'test_loss': test_loss,
            'train_losses': train_losses,
            'train_accs': train_accs
        }
    
    def evaluate_model(self, model):
        """Evaluate model on test set"""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_acc = 100.0 * correct / total
        avg_test_loss = test_loss / len(self.test_loader)
        
        return test_acc, avg_test_loss
    
    def run_full_ablation(self):
        """Run complete ablation study"""
        print("🚀 Starting Comprehensive Ablation Study")
        print("=" * 60)
        
        for model_name, model_class in self.models.items():
            try:
                result = self.train_model(model_class, model_name)
                self.results[model_name] = result
                
                print(f"✅ {model_name}: {result['test_acc']:.2f}% accuracy")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        self.analyze_results()
        self.create_visualizations()
        self.save_results()
    
    def analyze_results(self):
        """Analyze ablation results"""
        print("\n" + "=" * 60)
        print("📊 ABLATION STUDY RESULTS")
        print("=" * 60)
        
        # Sort by accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['test_acc'],
            reverse=True
        )
        
        print(f"{'Model':<25} {'Test Acc':<10} {'Δ from Baseline':<15} {'Params':<10}")
        print("-" * 70)
        
        baseline_acc = next(
            result['test_acc'] for name, result in self.results.items() 
            if 'Baseline' in name
        )
        
        for name, result in sorted_results:
            delta = result['test_acc'] - baseline_acc
            print(f"{name:<25} {result['test_acc']:<10.2f}% "
                  f"{delta:<15.2f}% {result['total_params']:<10,}")
        
        # Calculate contributions
        full_nkat_acc = next(
            result['test_acc'] for name, result in self.results.items() 
            if name == 'Full NKAT'
        )
        
        print(f"\n🎯 NKAT Enhancement Contributions:")
        print("-" * 40)
        
        for name, result in self.results.items():
            if 'NKAT -' in name:
                component = name.replace('NKAT - ', '')
                # Contribution = Full NKAT - (NKAT without component)
                contribution = full_nkat_acc - result['test_acc']
                print(f"{component:<20}: {contribution:>+.2f}%")
    
    def create_visualizations(self):
        """Create ablation study visualizations"""
        # Figure 1: Accuracy comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of accuracies
        names = list(self.results.keys())
        accs = [result['test_acc'] for result in self.results.values()]
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        
        bars = ax1.bar(range(len(names)), accs, color=colors)
        ax1.set_xlabel('Model Variant')
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('NKAT Ablation Study: Accuracy Comparison')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{acc:.2f}%', ha='center', va='bottom')
        
        # Contribution analysis
        baseline_acc = next(
            result['test_acc'] for name, result in self.results.items() 
            if 'Baseline' in name
        )
        
        full_nkat_acc = next(
            result['test_acc'] for name, result in self.results.items() 
            if name == 'Full NKAT'
        )
        
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
        plt.savefig(f'nkat_ablation_study_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Training curves comparison
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training curves for all models"""
        plt.figure(figsize=(12, 8))
        
        for name, result in self.results.items():
            if 'train_accs' in result:
                epochs = range(1, len(result['train_accs']) + 1)
                plt.plot(epochs, result['train_accs'], label=f"{name}", linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy (%)')
        plt.title('Training Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'nkat_training_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results for JSON serialization
        json_results = {}
        for name, result in self.results.items():
            json_results[name] = {
                k: v for k, v in result.items() 
                if k not in ['train_losses', 'train_accs']  # Skip large arrays
            }
            
            # Add summary statistics
            if 'train_accs' in result:
                json_results[name]['final_train_acc'] = result['train_accs'][-1]
                json_results[name]['max_train_acc'] = max(result['train_accs'])
        
        # Save results
        with open(f'nkat_ablation_results_{timestamp}.json', 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'config': vars(self.config),
                'results': json_results,
                'summary': {
                    'best_model': max(self.results.items(), key=lambda x: x[1]['test_acc']),
                    'total_models_tested': len(self.results)
                }
            }, f, indent=2)
        
        print(f"✅ Results saved to nkat_ablation_results_{timestamp}.json")

def main():
    """Run ablation study"""
    study = AblationStudy()
    study.run_full_ablation()

if __name__ == "__main__":
    main() 