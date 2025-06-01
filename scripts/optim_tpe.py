# optim_tpe.py
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import gc
import time
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm
import torch.nn.functional as F

from nkat_transformer.model import NKATVisionTransformer
from utils.metrics import tpe_metric, count_nkat_parameters, comprehensive_model_analysis

# 英語グラフ設定（文字化け防止）
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

# CUDA最適化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"🚀 RTX3080 CUDA Optimization: {torch.cuda.get_device_name(0)}")


def get_dataloaders(batch_size: int = 128, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    📦 データローダー準備（高速化版）
    Windows対応：num_workers=0でマルチプロセッシング回避
    
    Args:
        batch_size: バッチサイズ
        num_workers: ワーカー数（Windows環境では0推奨）
        
    Returns:
        (train_loader, val_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # MNIST データセット
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    val_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 高速化：クイック訓練用にサブセット作成
    quick_train_size = min(10000, len(train_dataset))  # 1万サンプル
    quick_val_size = min(2000, len(val_dataset))       # 2千サンプル
    
    train_indices = torch.randperm(len(train_dataset))[:quick_train_size]
    val_indices = torch.randperm(len(val_dataset))[:quick_val_size]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


def quick_train_and_eval(model: nn.Module, 
                        lr: float,
                        label_smoothing: float = 0.0,
                        epochs: int = 3,
                        device: str = 'cuda') -> Dict[str, float]:
    """
    ⚡ クイック訓練・評価（Optuna用）
    
    Args:
        model: 訓練対象モデル
        lr: 学習率
        label_smoothing: ラベルスムージング
        epochs: エポック数
        device: デバイス
        
    Returns:
        評価結果辞書
    """
    model = model.to(device)
    model.train()
    
    # データローダー準備（Windows対応）
    train_loader, val_loader = get_dataloaders(batch_size=64, num_workers=0)
    
    # オプティマイザー・スケジューラー
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 訓練ループ
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(device), targets.to(device)
            targets = targets.long()  # 確実にlong型にキャスト
            
            optimizer.zero_grad()
            outputs = model(data)
            
            # カスタム損失計算（ラベルスムージング対応）
            if hasattr(model, 'compute_loss'):
                loss = model.compute_loss(outputs, targets)
            else:
                # 手動でラベルスムージング適用
                if label_smoothing > 0:
                    num_classes = outputs.size(-1)
                    one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
                    smoothed_targets = one_hot * (1 - label_smoothing) + label_smoothing / num_classes
                    log_probs = F.log_softmax(outputs, dim=-1)
                    loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()
                else:
                    loss = nn.CrossEntropyLoss()(outputs, targets)
            
            loss.backward()
            
            # 勾配クリッピング（安定化）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
    
    # 評価
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            targets = targets.long()  # 確実にlong型にキャスト
            outputs = model(data)
            
            if hasattr(model, 'compute_loss'):
                loss = model.compute_loss(outputs, targets)
            else:
                loss = nn.CrossEntropyLoss()(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()
    
    val_accuracy = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    
    return {
        'val_accuracy': val_accuracy,
        'val_loss': avg_val_loss,
        'train_losses': train_losses,
        'final_train_loss': train_losses[-1] if train_losses else 0.0
    }


def objective(trial: optuna.Trial) -> float:
    """
    🎯 Optuna目的関数：TPE指標最大化
    
    Args:
        trial: Optunaトライアル
        
    Returns:
        TPEスコア（最大化目標）
    """
    try:
        # ----------- 🎛️ ハイパーパラメータ探索 -----------
        
        # LLMスタイルパラメータ
        temperature = trial.suggest_float("temperature", 0.5, 1.5)
        top_k = trial.suggest_int("top_k", 0, 20)
        top_p = trial.suggest_float("top_p", 0.7, 1.0)
        
        # NKAT理論パラメータ
        nkat_strength = trial.suggest_float("nkat_strength", 0.001, 0.05, log=True)
        nkat_decay = trial.suggest_float("nkat_decay", 0.85, 1.0)
        
        # 訓練パラメータ
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15)
        
        # Dropout系
        dropout_attn = trial.suggest_float("dropout_attn", 0.05, 0.2)
        dropout_embed = trial.suggest_float("dropout_embed", 0.05, 0.2)
        
        # アーキテクチャ調整
        embed_dim_choice = trial.suggest_categorical("embed_dim", [256, 384, 512])
        depth_choice = trial.suggest_int("depth", 4, 8)
        
        # ----------- 🏗️ モデル構築 -----------
        model = NKATVisionTransformer(
            img_size=28,
            patch_size=4,
            num_classes=10,
            embed_dim=embed_dim_choice,
            depth=depth_choice,
            num_heads=8,
            # LLMスタイルパラメータ
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            # NKAT理論パラメータ
            nkat_strength=nkat_strength,
            nkat_decay=nkat_decay,
            # 正則化
            dropout_embed=dropout_embed,
            dropout_attn=dropout_attn,
            label_smoothing=label_smoothing
        ).cuda()
        
        # ----------- ⚡ クイック訓練・評価 -----------
        start_time = time.time()
        results = quick_train_and_eval(model, lr, label_smoothing, epochs=3)
        train_time = time.time() - start_time
        
        val_accuracy = results['val_accuracy']
        
        # ----------- 📊 TPE指標計算 -----------
        param_analysis = count_nkat_parameters(model)
        lambda_theory = param_analysis['nkat_params']
        
        # 基本TPE計算
        basic_tpe = tpe_metric(val_accuracy, lambda_theory)
        
        # 効率性ペナルティ
        efficiency_penalty = 0.0
        if train_time > 60:  # 1分以上の場合ペナルティ
            efficiency_penalty = np.log10(train_time / 60)
        
        # 最終TPEスコア
        final_tpe = basic_tpe / (1.0 + efficiency_penalty)
        
        # ----------- 📈 中間値報告（Pruning用） -----------
        trial.report(final_tpe, step=0)
        
        # ----------- 🗑️ メモリクリーンアップ -----------
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # ----------- 📝 追加属性保存 -----------
        trial.set_user_attr("val_accuracy", val_accuracy)
        trial.set_user_attr("lambda_theory", lambda_theory)
        trial.set_user_attr("train_time", train_time)
        trial.set_user_attr("basic_tpe", basic_tpe)
        trial.set_user_attr("nkat_ratio", param_analysis['nkat_ratio'])
        
        return final_tpe
    
    except Exception as e:
        print(f"❌ Trial failed: {e}")
        return 0.0


def enhanced_objective(trial: optuna.Trial) -> float:
    """
    🚀 強化版目的関数：複数データセット対応
    
    Args:
        trial: Optunaトライアル
        
    Returns:
        汎化TPEスコア
    """
    # 基本目的関数を実行
    base_tpe = objective(trial)
    
    # より厳密な評価が必要な場合のみ実行
    if base_tpe > 0.18:  # 閾値以上の場合のみ
        try:
            # 他のデータセットでの評価も追加可能
            # 例：FashionMNIST、EMNIST等での汎化性テスト
            pass
        except:
            pass
    
    return base_tpe


def print_trial_summary(study: optuna.Study):
    """
    📊 トライアル結果サマリー表示
    
    Args:
        study: 完了したOptuna Study
    """
    if len(study.trials) == 0:
        print("No trials completed.")
        return
    
    best_trial = study.best_trial
    
    print("\n" + "="*60)
    print("🏆 BEST TRIAL SUMMARY")
    print("="*60)
    print(f"🎯 Best TPE Score: {best_trial.value:.6f}")
    
    # N/A処理を追加
    val_acc = best_trial.user_attrs.get('val_accuracy', None)
    if val_acc is not None:
        print(f"📊 Validation Accuracy: {val_acc:.4f}")
    else:
        print(f"📊 Validation Accuracy: N/A")
    
    lambda_theory = best_trial.user_attrs.get('lambda_theory', None)
    if lambda_theory is not None:
        print(f"🧠 Lambda Theory: {lambda_theory:,}")
    else:
        print(f"🧠 Lambda Theory: N/A")
    
    train_time = best_trial.user_attrs.get('train_time', None)
    if train_time is not None:
        print(f"⏱️ Training Time: {train_time:.2f}s")
    else:
        print(f"⏱️ Training Time: N/A")
    
    nkat_ratio = best_trial.user_attrs.get('nkat_ratio', None)
    if nkat_ratio is not None:
        print(f"🔬 NKAT Ratio: {nkat_ratio:.6f}")
    else:
        print(f"🔬 NKAT Ratio: N/A")
    
    print("\n🎛️ BEST HYPERPARAMETERS:")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.6f}")
        else:
            print(f"  {key:20s}: {value}")
    
    print("\n📈 TOP 5 TRIALS:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]
    for i, trial in enumerate(sorted_trials):
        val_acc = trial.user_attrs.get('val_accuracy', 0)
        tpe_score = trial.value or 0
        print(f"  #{i+1}: TPE={tpe_score:.4f}, Acc={val_acc:.4f}")
    
    print("="*60) 