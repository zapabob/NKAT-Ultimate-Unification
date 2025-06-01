#!/usr/bin/env python3
# nkat_stage2_optimized_generalization.py
"""
NKAT-Transformer Stage2 最適化汎化テスト
TPE=0.7113ベストパラメータでの複数データセット転移学習

対象データセット:
- MNIST (基準)
- FashionMNIST
- EMNIST
- CIFAR10 (グレースケール変換)

目標:
- Global TPE ≥ 0.70
- 全データセットで安定した性能
- 汎化性能の定量評価
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
import os
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 英語グラフ設定（文字化け防止）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# CUDA最適化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

from nkat_transformer.model import NKATVisionTransformer
from utils.metrics import tpe_metric, count_nkat_parameters


class OptimizedStage2Tester:
    """最適化Stage2汎化テスター"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # ベストパラメータ（TPE=0.7113）
        self.best_params = {
            'temperature': 0.5469995537778101,
            'top_k': 6,
            'top_p': 0.7554317362491326,
            'nkat_strength': 0.0023912273756024186,
            'nkat_decay': 0.992297354758566,
            'lr': 0.00023485677334823308,
            'label_smoothing': 0.06691314688896288,
            'dropout_attn': 0.17479367920079422,
            'dropout_embed': 0.07883647026878872,
            'embed_dim': 384,
            'depth': 5
        }
        
        # データセット設定
        self.datasets_config = {
            'MNIST': {
                'num_classes': 10,
                'normalize': (0.1307, 0.3081),
                'dataset_class': torchvision.datasets.MNIST,
                'kwargs': {}
            },
            'FashionMNIST': {
                'num_classes': 10,
                'normalize': (0.2860, 0.3530),
                'dataset_class': torchvision.datasets.FashionMNIST,
                'kwargs': {}
            },
            'EMNIST': {
                'num_classes': 27,  # letters split
                'normalize': (0.1751, 0.3332),
                'dataset_class': torchvision.datasets.EMNIST,
                'kwargs': {'split': 'letters'}
            },
            'CIFAR10': {
                'num_classes': 10,
                'normalize': (0.4734, 0.2516),  # グレースケール変換後
                'dataset_class': torchvision.datasets.CIFAR10,
                'kwargs': {},
                'special_transform': True
            }
        }
        
        # 訓練設定
        self.training_config = {
            'epochs_per_dataset': 8,  # 効率的な転移学習
            'batch_size': 128,
            'train_samples': 8000,    # 高速化のためサンプル制限
            'test_samples': 2000,
            'early_stopping_patience': 3,
            'grad_clip_norm': 1.0
        }
        
        self.results = {}
    
    def create_model(self, num_classes: int) -> nn.Module:
        """最適化モデル作成"""
        model = NKATVisionTransformer(
            img_size=28,
            patch_size=4,
            num_classes=num_classes,
            embed_dim=self.best_params['embed_dim'],
            depth=self.best_params['depth'],
            num_heads=8,
            temperature=self.best_params['temperature'],
            top_k=self.best_params['top_k'],
            top_p=self.best_params['top_p'],
            nkat_strength=self.best_params['nkat_strength'],
            nkat_decay=self.best_params['nkat_decay'],
            dropout_attn=self.best_params['dropout_attn'],
            dropout_embed=self.best_params['dropout_embed']
        ).to(self.device)
        
        return model
    
    def get_dataset_loaders(self, dataset_name: str) -> Tuple[DataLoader, DataLoader]:
        """データセット別ローダー取得"""
        config = self.datasets_config[dataset_name]
        
        # 基本変換
        if dataset_name == 'CIFAR10':
            # CIFAR10をグレースケール28x28に変換
            train_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(28),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize((config['normalize'][0],), (config['normalize'][1],))
            ])
            test_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((config['normalize'][0],), (config['normalize'][1],))
            ])
        else:
            # 28x28グレースケール
            train_transform = transforms.Compose([
                transforms.RandomRotation(5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize((config['normalize'][0],), (config['normalize'][1],))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((config['normalize'][0],), (config['normalize'][1],))
            ])
        
        # データセット作成
        train_dataset = config['dataset_class'](
            root='./data', train=True, download=True, 
            transform=train_transform, **config['kwargs']
        )
        test_dataset = config['dataset_class'](
            root='./data', train=False, download=True, 
            transform=test_transform, **config['kwargs']
        )
        
        # サンプル制限（高速化）
        train_size = min(self.training_config['train_samples'], len(train_dataset))
        test_size = min(self.training_config['test_samples'], len(test_dataset))
        
        train_indices = torch.randperm(len(train_dataset))[:train_size]
        test_indices = torch.randperm(len(test_dataset))[:test_size]
        
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        
        train_loader = DataLoader(
            train_subset, 
            batch_size=self.training_config['batch_size'], 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_subset, 
            batch_size=self.training_config['batch_size'], 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def finetune_on_dataset(self, dataset_name: str, 
                           pretrained_model_path: str = None) -> Dict[str, Any]:
        """データセット上でのファインチューニング"""
        
        print(f"\nFine-tuning on {dataset_name}...")
        
        config = self.datasets_config[dataset_name]
        num_classes = config['num_classes']
        
        # モデル作成
        model = self.create_model(num_classes)
        
        # 事前訓練重みロード（可能な場合）
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            try:
                checkpoint = torch.load(pretrained_model_path, map_location=self.device)
                # 分類層以外をロード
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint.items() 
                                 if k in model_dict and 'head' not in k}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print(f"Loaded pretrained weights (excluding head)")
            except Exception as e:
                print(f"WARNING: Could not load pretrained weights: {e}")
        
        # データローダー
        train_loader, test_loader = self.get_dataset_loaders(dataset_name)
        
        # オプティマイザー・スケジューラー
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.best_params['lr'] * 2.0,  # 転移学習用に少し高め
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.training_config['epochs_per_dataset'],
            eta_min=1e-6
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.best_params['label_smoothing']
        )
        
        # 訓練ループ
        model.train()
        train_losses = []
        val_accuracies = []
        best_acc = 0.0
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.training_config['epochs_per_dataset']):
            # 訓練
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, 
                              desc=f"{dataset_name} Epoch {epoch+1}/{self.training_config['epochs_per_dataset']}", 
                              leave=False)
            
            for data, targets in progress_bar:
                data, targets = data.to(self.device), targets.to(self.device).long()
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.training_config['grad_clip_norm']
                )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            scheduler.step()
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # 検証
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(self.device), targets.to(self.device).long()
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)
            avg_val_loss = val_loss / len(test_loader)
            
            # ベスト更新チェック
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            model.train()
            
            print(f"  Epoch {epoch+1}: Loss={avg_train_loss:.4f}, ValAcc={val_accuracy:.4f}, Best={best_acc:.4f}")
            
            # Early stopping
            if patience_counter >= self.training_config['early_stopping_patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        
        # パラメータ分析
        param_analysis = count_nkat_parameters(model)
        
        # TPE計算
        tpe_score = tpe_metric(best_acc, param_analysis['nkat_params'])
        
        result = {
            'dataset': dataset_name,
            'num_classes': num_classes,
            'best_accuracy': best_acc,
            'final_val_loss': avg_val_loss,
            'training_time': training_time,
            'epochs_completed': len(train_losses),
            'tpe_score': tpe_score,
            'lambda_theory': param_analysis['nkat_params'],
            'nkat_ratio': param_analysis['nkat_ratio'],
            'total_params': param_analysis['total_params'],
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }
        
        print(f"{dataset_name}: Acc={best_acc:.4f}, TPE={tpe_score:.6f}, Time={training_time:.1f}s")
        return result
    
    def run_generalization_test(self, pretrained_model_path: str = None) -> Dict[str, Any]:
        """汎化テスト実行"""
        
        print("Starting Optimized Stage2 Generalization Test")
        print("="*70)
        print(f"Using best params: TPE=0.7113 configuration")
        print(f"Target: Global TPE >= 0.70, consistent performance")
        print("="*70)
        
        all_results = []
        tpe_scores = []
        accuracies = []
        
        # 各データセットでテスト
        for dataset_name in self.datasets_config.keys():
            try:
                result = self.finetune_on_dataset(dataset_name, pretrained_model_path)
                all_results.append(result)
                tpe_scores.append(result['tpe_score'])
                accuracies.append(result['best_accuracy'])
                
            except Exception as e:
                print(f"ERROR: Failed on {dataset_name}: {e}")
                error_result = {
                    'dataset': dataset_name,
                    'best_accuracy': 0.0,
                    'tpe_score': 0.0,
                    'error': str(e)
                }
                all_results.append(error_result)
                tpe_scores.append(0.0)
                accuracies.append(0.0)
        
        # Global指標計算
        valid_tpe_scores = [score for score in tpe_scores if score > 0]
        valid_accuracies = [acc for acc in accuracies if acc > 0]
        
        global_tpe = np.mean(valid_tpe_scores) if valid_tpe_scores else 0.0
        global_accuracy = np.mean(valid_accuracies) if valid_accuracies else 0.0
        
        # 汎化性指標
        generalization_score = min(valid_tpe_scores) if valid_tpe_scores else 0.0  # 最悪ケース
        consistency_score = 1.0 - np.std(valid_tpe_scores) if len(valid_tpe_scores) > 1 else 1.0
        robustness_score = len(valid_tpe_scores) / len(self.datasets_config)  # 成功率
        
        # 結果サマリー
        summary = {
            'timestamp': datetime.now().isoformat(),
            'best_params': self.best_params,
            'training_config': self.training_config,
            'global_metrics': {
                'global_tpe': global_tpe,
                'global_accuracy': global_accuracy,
                'generalization_score': generalization_score,
                'consistency_score': consistency_score,
                'robustness_score': robustness_score
            },
            'dataset_results': all_results,
            'tpe_scores': tpe_scores,
            'accuracies': accuracies,
            'performance_analysis': {
                'best_dataset': max(all_results, key=lambda x: x.get('tpe_score', 0))['dataset'] if all_results else None,
                'worst_dataset': min(all_results, key=lambda x: x.get('tpe_score', 0))['dataset'] if all_results else None,
                'tpe_range': max(valid_tpe_scores) - min(valid_tpe_scores) if len(valid_tpe_scores) > 1 else 0.0,
                'accuracy_range': max(valid_accuracies) - min(valid_accuracies) if len(valid_accuracies) > 1 else 0.0
            }
        }
        
        self.results = summary
        return summary
    
    def save_results(self, output_path: str = None):
        """結果保存"""
        if not self.results:
            print("WARNING: No results to save")
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"logs/stage2_optimized_generalization_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")
    
    def create_visualization(self, output_path: str = None):
        """結果可視化"""
        if not self.results:
            print("WARNING: No results to visualize")
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"logs/stage2_optimized_viz_{timestamp}.png"
        
        fig = plt.figure(figsize=(16, 12))
        
        # データセット別性能比較
        ax1 = plt.subplot(2, 3, 1)
        datasets = [r['dataset'] for r in self.results['dataset_results'] if 'error' not in r]
        accuracies = [r['best_accuracy'] for r in self.results['dataset_results'] if 'error' not in r]
        
        bars1 = ax1.bar(datasets, [acc*100 for acc in accuracies], color='skyblue', alpha=0.7)
        ax1.set_title('Accuracy by Dataset (%)', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # データセット別TPE
        ax2 = plt.subplot(2, 3, 2)
        tpe_scores = [r['tpe_score'] for r in self.results['dataset_results'] if 'error' not in r]
        
        bars2 = ax2.bar(datasets, tpe_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('TPE Score by Dataset', fontweight='bold')
        ax2.set_ylabel('TPE Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, tpe in zip(bars2, tpe_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{tpe:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 精度 vs TPE散布図
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(accuracies, tpe_scores, c='green', alpha=0.7, s=100)
        ax3.set_xlabel('Accuracy')
        ax3.set_ylabel('TPE Score')
        ax3.set_title('Accuracy vs TPE Score', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # データセット名をラベル
        for i, dataset in enumerate(datasets):
            ax3.annotate(dataset, (accuracies[i], tpe_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Global指標
        ax4 = plt.subplot(2, 3, 4)
        global_metrics = self.results['global_metrics']
        metrics = ['Global TPE', 'Global Accuracy', 'Generalization', 'Consistency', 'Robustness']
        values = [
            global_metrics['global_tpe'],
            global_metrics['global_accuracy'],
            global_metrics['generalization_score'],
            global_metrics['consistency_score'],
            global_metrics['robustness_score']
        ]
        colors = ['purple', 'blue', 'green', 'orange', 'red']
        
        bars4 = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_title('Global Generalization Metrics', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, value in zip(bars4, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 訓練時間比較
        ax5 = plt.subplot(2, 3, 5)
        training_times = [r['training_time'] for r in self.results['dataset_results'] if 'error' not in r]
        
        bars5 = ax5.bar(datasets, training_times, color='gold', alpha=0.7)
        ax5.set_title('Training Time by Dataset (s)', fontweight='bold')
        ax5.set_ylabel('Time (seconds)')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # パフォーマンス分析
        ax6 = plt.subplot(2, 3, 6)
        analysis = self.results['performance_analysis']
        
        # レーダーチャート風の表示
        categories = ['TPE Range', 'Accuracy Range', 'Consistency', 'Robustness']
        values_norm = [
            1.0 - min(analysis['tpe_range'], 0.5) / 0.5,  # 範囲が小さいほど良い
            1.0 - min(analysis['accuracy_range'], 0.3) / 0.3,
            global_metrics['consistency_score'],
            global_metrics['robustness_score']
        ]
        
        bars6 = ax6.bar(categories, values_norm, color='lightgreen', alpha=0.7)
        ax6.set_title('Performance Analysis', fontweight='bold')
        ax6.set_ylabel('Normalized Score')
        ax6.tick_params(axis='x', rotation=45)
        ax6.set_ylim([0, 1])
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    def print_summary(self):
        """結果サマリー表示"""
        if not self.results:
            print("WARNING: No results available")
            return
        
        print("\n" + "="*70)
        print("OPTIMIZED STAGE2 GENERALIZATION TEST SUMMARY")
        print("="*70)
        
        global_metrics = self.results['global_metrics']
        print(f"Global TPE Score: {global_metrics['global_tpe']:.6f}")
        print(f"Global Accuracy: {global_metrics['global_accuracy']:.4f}")
        print(f"Generalization Score: {global_metrics['generalization_score']:.6f}")
        print(f"Consistency Score: {global_metrics['consistency_score']:.6f}")
        print(f"Robustness Score: {global_metrics['robustness_score']:.6f}")
        
        print(f"\nDATASET BREAKDOWN:")
        for result in self.results['dataset_results']:
            if 'error' in result:
                print(f"  {result['dataset']:>12}: ERROR {result['error']}")
            else:
                print(f"  {result['dataset']:>12}: Acc={result['best_accuracy']:.4f}, "
                      f"TPE={result['tpe_score']:.6f}, Time={result['training_time']:.1f}s")
        
        # パフォーマンス分析
        analysis = self.results['performance_analysis']
        print(f"\nPERFORMANCE ANALYSIS:")
        print(f"  Best Dataset: {analysis['best_dataset']}")
        print(f"  Worst Dataset: {analysis['worst_dataset']}")
        print(f"  TPE Range: {analysis['tpe_range']:.4f}")
        print(f"  Accuracy Range: {analysis['accuracy_range']:.4f}")
        
        # 目標達成評価
        target_achieved = global_metrics['global_tpe'] >= 0.70
        consistency_good = global_metrics['consistency_score'] >= 0.8
        robustness_good = global_metrics['robustness_score'] >= 0.75
        
        print(f"\nTARGET ACHIEVEMENT:")
        print(f"  Global TPE >= 0.70: {'ACHIEVED' if target_achieved else 'NOT ACHIEVED'}")
        print(f"  Consistency >= 0.80: {'GOOD' if consistency_good else 'NEEDS IMPROVEMENT'}")
        print(f"  Robustness >= 0.75: {'GOOD' if robustness_good else 'NEEDS IMPROVEMENT'}")
        
        if target_achieved and consistency_good and robustness_good:
            print(f"\nEXCELLENT GENERALIZATION ACHIEVED!")
        elif target_achieved:
            print(f"\nTARGET ACHIEVED! Some aspects need improvement")
        else:
            print(f"\nProgress made, continue optimization")
        
        print("="*70)


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NKAT Optimized Stage2 Generalization Test")
    parser.add_argument("--pretrained_model", type=str, default=None,
                      help="Path to pretrained model weights")
    parser.add_argument("--epochs", type=int, default=8,
                      help="Epochs per dataset")
    parser.add_argument("--batch_size", type=int, default=128,
                      help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device")
    parser.add_argument("--output_dir", type=str, default="logs",
                      help="Output directory")
    
    args = parser.parse_args()
    
    # テスター作成
    tester = OptimizedStage2Tester(device=args.device)
    
    # 設定上書き
    if args.epochs != 8:
        tester.training_config['epochs_per_dataset'] = args.epochs
    if args.batch_size != 128:
        tester.training_config['batch_size'] = args.batch_size
    
    try:
        # テスト実行
        results = tester.run_generalization_test(args.pretrained_model)
        
        # 結果表示・保存
        tester.print_summary()
        tester.save_results()
        tester.create_visualization()
        
        print(f"\nStage2 generalization test completed!")
        
    except KeyboardInterrupt:
        print(f"\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 