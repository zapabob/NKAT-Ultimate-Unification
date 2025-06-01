#!/usr/bin/env python3
# nkat_stage2_generalization_runner.py
"""
🌍 NKAT-Transformer Stage2 汎化テスト基盤
複数データセットでの転移学習 + Global TPE指標計算

対象データセット:
- MNIST (基準)
- FashionMNIST
- EMNIST
- CIFAR10 (グレースケール変換)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

from nkat_transformer.model import NKATVisionTransformer
from utils.metrics import tpe_metric, count_nkat_parameters, comprehensive_model_analysis

# 英語グラフ設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

class Stage2GeneralizationTester:
    """🌍 Stage2汎化テスター"""
    
    def __init__(self, model_config: Dict[str, Any] = None, device: str = 'cuda'):
        self.device = device
        self.model_config = model_config or self._get_default_config()
        self.datasets = ['MNIST', 'FashionMNIST', 'EMNIST', 'CIFAR10']
        self.results = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルトモデル設定"""
        return {
            'embed_dim': 512,
            'depth': 6,
            'num_heads': 8,
            'temperature': 1.0,
            'top_k': 10,
            'top_p': 0.9,
            'nkat_strength': 0.015,
            'nkat_decay': 0.95,
            'dropout_attn': 0.1,
            'dropout_embed': 0.1,
            'label_smoothing': 0.05
        }
    
    def _get_dataset_loaders(self, dataset_name: str, batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
        """データセット別ローダー取得"""
        
        if dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )
            
        elif dataset_name == 'FashionMNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
            train_dataset = torchvision.datasets.FashionMNIST(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.FashionMNIST(
                root='./data', train=False, download=True, transform=transform
            )
            
        elif dataset_name == 'EMNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1751,), (0.3332,))
            ])
            train_dataset = torchvision.datasets.EMNIST(
                root='./data', split='letters', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.EMNIST(
                root='./data', split='letters', train=False, download=True, transform=transform
            )
            
        elif dataset_name == 'CIFAR10':
            # CIFAR10をグレースケール28x28に変換
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((0.4734,), (0.2516,))
            ])
            train_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # クイックテスト用にサイズを制限
        quick_train_size = min(5000, len(train_dataset))
        quick_test_size = min(1000, len(test_dataset))
        
        train_indices = torch.randperm(len(train_dataset))[:quick_train_size]
        test_indices = torch.randperm(len(test_dataset))[:quick_test_size]
        
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    
    def create_model(self, num_classes: int = 10) -> nn.Module:
        """設定に基づいてモデル作成"""
        model = NKATVisionTransformer(
            img_size=28,
            patch_size=4,
            num_classes=num_classes,
            **self.model_config
        ).to(self.device)
        return model
    
    def finetune_on_dataset(self, 
                           model_path: str, 
                           dataset_name: str, 
                           epochs: int = 5,
                           lr: float = 1e-4) -> Dict[str, float]:
        """データセット上でのファインチューニング"""
        
        print(f"\n🔧 Fine-tuning on {dataset_name}...")
        
        # データセット固有のクラス数設定
        num_classes_map = {
            'MNIST': 10,
            'FashionMNIST': 10,
            'EMNIST': 27,  # letters split
            'CIFAR10': 10
        }
        num_classes = num_classes_map[dataset_name]
        
        # モデル作成（事前訓練重みがある場合はロード）
        model = self.create_model(num_classes)
        
        if model_path and model_path != "scratch":
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint, strict=False)
                print(f"✅ Loaded pretrained weights from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load weights: {e}. Training from scratch.")
        
        # データローダー
        train_loader, test_loader = self._get_dataset_loaders(dataset_name)
        
        # オプティマイザー
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 訓練ループ
        model.train()
        train_losses = []
        
        start_time = time.time()
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"{dataset_name} Epoch {epoch+1}/{epochs}", leave=False)
            for data, targets in progress_bar:
                data, targets = data.to(self.device), targets.to(self.device).long()
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = model.compute_loss(outputs, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            scheduler.step()
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
        
        training_time = time.time() - start_time
        
        # 評価
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device).long()
                outputs = model(data)
                loss = model.compute_loss(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        avg_test_loss = test_loss / len(test_loader)
        
        # パラメータ分析
        param_analysis = count_nkat_parameters(model)
        
        # TPE計算
        tpe_score = tpe_metric(accuracy, param_analysis['nkat_params'])
        
        result = {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'test_loss': avg_test_loss,
            'training_time': training_time,
            'tpe_score': tpe_score,
            'lambda_theory': param_analysis['nkat_params'],
            'nkat_ratio': param_analysis['nkat_ratio'],
            'final_train_loss': train_losses[-1] if train_losses else 0.0
        }
        
        print(f"✅ {dataset_name}: Acc={accuracy:.4f}, TPE={tpe_score:.6f}")
        return result
    
    def run_generalization_test(self, 
                               model_path: str = "scratch",
                               epochs_per_dataset: int = 5) -> Dict[str, Any]:
        """汎化テスト実行"""
        
        print("🌍 Starting Stage2 Generalization Test")
        print("="*60)
        
        all_results = []
        tpe_scores = []
        
        for dataset in self.datasets:
            try:
                result = self.finetune_on_dataset(model_path, dataset, epochs_per_dataset)
                all_results.append(result)
                tpe_scores.append(result['tpe_score'])
            except Exception as e:
                print(f"❌ Failed on {dataset}: {e}")
                all_results.append({
                    'dataset': dataset,
                    'accuracy': 0.0,
                    'tpe_score': 0.0,
                    'error': str(e)
                })
                tpe_scores.append(0.0)
        
        # Global TPE指標計算
        global_tpe = np.mean(tpe_scores) if tpe_scores else 0.0
        global_accuracy = np.mean([r.get('accuracy', 0) for r in all_results])
        
        # 結果サマリー
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_config': self.model_config,
            'global_tpe': global_tpe,
            'global_accuracy': global_accuracy,
            'dataset_results': all_results,
            'tpe_scores': tpe_scores,
            'generalization_score': min(tpe_scores) if tpe_scores else 0.0,  # 最悪ケース
            'consistency_score': 1.0 - np.std(tpe_scores) if len(tpe_scores) > 1 else 1.0
        }
        
        self.results = summary
        return summary
    
    def save_results(self, output_path: str = None):
        """結果保存"""
        if not self.results:
            print("⚠️ No results to save")
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"logs/stage2_generalization_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to {output_path}")
    
    def create_visualization(self, output_path: str = None):
        """結果可視化"""
        if not self.results:
            print("⚠️ No results to visualize")
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"logs/stage2_visualization_{timestamp}.png"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # データセット別精度
        datasets = [r['dataset'] for r in self.results['dataset_results']]
        accuracies = [r.get('accuracy', 0) for r in self.results['dataset_results']]
        
        ax1.bar(datasets, accuracies, color='skyblue')
        ax1.set_title('Accuracy by Dataset')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # データセット別TPE
        tpe_scores = [r.get('tpe_score', 0) for r in self.results['dataset_results']]
        
        ax2.bar(datasets, tpe_scores, color='lightcoral')
        ax2.set_title('TPE Score by Dataset')
        ax2.set_ylabel('TPE Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 精度 vs TPE散布図
        ax3.scatter(accuracies, tpe_scores, color='green', alpha=0.7)
        ax3.set_xlabel('Accuracy')
        ax3.set_ylabel('TPE Score')
        ax3.set_title('Accuracy vs TPE Score')
        ax3.grid(True, alpha=0.3)
        
        # 汎化性サマリー
        metrics = ['Global TPE', 'Global Accuracy', 'Generalization Score', 'Consistency Score']
        values = [
            self.results['global_tpe'],
            self.results['global_accuracy'],
            self.results['generalization_score'],
            self.results['consistency_score']
        ]
        
        ax4.barh(metrics, values, color='gold')
        ax4.set_title('Generalization Metrics')
        ax4.set_xlabel('Score')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved to {output_path}")
    
    def print_summary(self):
        """結果サマリー表示"""
        if not self.results:
            print("⚠️ No results available")
            return
        
        print("\n" + "="*60)
        print("🌍 STAGE2 GENERALIZATION TEST SUMMARY")
        print("="*60)
        print(f"🎯 Global TPE Score: {self.results['global_tpe']:.6f}")
        print(f"📊 Global Accuracy: {self.results['global_accuracy']:.4f}")
        print(f"🔄 Generalization Score: {self.results['generalization_score']:.6f}")
        print(f"📏 Consistency Score: {self.results['consistency_score']:.6f}")
        
        print("\n📋 DATASET BREAKDOWN:")
        for result in self.results['dataset_results']:
            if 'error' in result:
                print(f"  {result['dataset']:>12}: ❌ {result['error']}")
            else:
                print(f"  {result['dataset']:>12}: Acc={result['accuracy']:.4f}, TPE={result['tpe_score']:.6f}")
        
        print("="*60)


def load_best_optuna_config(optuna_results_path: str) -> Dict[str, Any]:
    """Optuna結果からベスト設定をロード"""
    try:
        with open(optuna_results_path, 'r', encoding='utf-8') as f:
            optuna_data = json.load(f)
        
        best_params = optuna_data.get('best_params', {})
        
        # Stage2用に設定変換
        config = {
            'embed_dim': best_params.get('embed_dim', 512),
            'depth': best_params.get('depth', 6),
            'num_heads': 8,
            'temperature': best_params.get('temperature', 1.0),
            'top_k': best_params.get('top_k') if best_params.get('top_k', 0) > 0 else None,
            'top_p': best_params.get('top_p', 0.9),
            'nkat_strength': best_params.get('nkat_strength', 0.015),
            'nkat_decay': best_params.get('nkat_decay', 0.95),
            'dropout_attn': best_params.get('dropout_attn', 0.1),
            'dropout_embed': best_params.get('dropout_embed', 0.1),
            'label_smoothing': best_params.get('label_smoothing', 0.05)
        }
        
        print(f"✅ Loaded Optuna config from {optuna_results_path}")
        return config
        
    except Exception as e:
        print(f"⚠️ Could not load Optuna config: {e}")
        return None


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NKAT Stage2 Generalization Test")
    parser.add_argument("--model_path", type=str, default="scratch",
                      help="Path to pretrained model weights")
    parser.add_argument("--optuna_config", type=str, default=None,
                      help="Path to Optuna results JSON for configuration")
    parser.add_argument("--epochs", type=int, default=5,
                      help="Epochs per dataset")
    parser.add_argument("--output_dir", type=str, default="logs",
                      help="Output directory")
    
    args = parser.parse_args()
    
    # 設定ロード
    config = None
    if args.optuna_config:
        config = load_best_optuna_config(args.optuna_config)
    
    # テスター作成・実行
    tester = Stage2GeneralizationTester(model_config=config)
    results = tester.run_generalization_test(args.model_path, args.epochs)
    
    # 結果表示・保存
    tester.print_summary()
    tester.save_results()
    tester.create_visualization()


if __name__ == "__main__":
    main() 