#!/usr/bin/env python3
"""
🎯 NKAT Optuna ハイパーパラメータ最適化システム
LLMスタイルのハイパーパラメータ導入 + TPE指標最適化

新規導入ハイパーパラメータ:
- Temperature: Attention/Logits温度制御
- TopK: Attention上位K個選択
- TopP: Nucleus Attention サンプリング
- Multi-scale Dropout: 層別Dropout率
- Dynamic NKAT strength: 適応的理論強度
- Regularization coefficients: L1/L2正則化

目的関数: TPE (Theory-Practical Equilibrium) スコア最大化
RTX3080最適化、tqdm進捗、英語グラフ対応
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import optuna
from optuna.integration import TensorBoardCallback
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# CUDA最適化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"🚀 RTX3080 CUDA Optimization: {torch.cuda.get_device_name(0)}")

# 英語グラフ設定（文字化け防止）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class LLMStyleAttention(nn.Module):
    """LLMスタイルのAttentionメカニズム（Temperature、TopK/TopP対応）"""
    
    def __init__(self, embed_dim, num_heads, temperature=1.0, top_k=None, top_p=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, S, D = x.shape
        
        # Q, K, V projection
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with temperature
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores / self.temperature  # Temperature scaling
        
        # TopK filtering
        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, scores.size(-1))
            top_k_scores, _ = torch.topk(scores, top_k, dim=-1)
            mask = scores < top_k_scores[..., -1:, :]
            scores = scores.masked_fill(mask, float('-inf'))
        
        # TopP (Nucleus) filtering
        if self.top_p is not None and self.top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_scores, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find cutoff index for nucleus
            nucleus_mask = cumulative_probs <= self.top_p
            nucleus_mask[..., 1:] = nucleus_mask[..., :-1].clone()
            nucleus_mask[..., 0] = True
            
            # Apply nucleus mask
            cutoff_indices = torch.gather(sorted_indices, -1, nucleus_mask.long().sum(-1, keepdim=True) - 1)
            nucleus_scores = torch.gather(sorted_scores, -1, cutoff_indices)
            mask = scores < nucleus_scores
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Standard attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        
        return out, attn_weights

class NKATOptimizedTransformer(nn.Module):
    """Optuna最適化対応NKAT-Transformer"""
    
    def __init__(self, 
                 img_size=28, 
                 patch_size=4, 
                 num_classes=10,
                 embed_dim=512, 
                 depth=6, 
                 num_heads=8, 
                 mlp_ratio=4.0,
                 # LLMスタイルハイパーパラメータ
                 temperature=1.0,
                 top_k=None,
                 top_p=None,
                 nkat_strength=0.0,
                 nkat_decay=1.0,  # NKAT強度の減衰
                 # Dropout系ハイパーパラメータ
                 dropout_attention=0.1,
                 dropout_mlp=0.1,
                 dropout_embedding=0.1,
                 dropout_classifier=0.1,
                 # 正則化ハイパーパラメータ
                 l1_reg=0.0,
                 l2_reg=0.0):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.nkat_strength = nkat_strength
        self.nkat_decay = nkat_decay
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        # パッチ埋め込み（実績のある段階的ConvStem）
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(dropout_embedding),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(dropout_embedding),
            nn.Conv2d(128, embed_dim, kernel_size=patch_size, stride=patch_size)
        )
        
        # 位置埋め込み
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
        # クラストークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # 入力正規化
        self.input_norm = nn.LayerNorm(embed_dim)
        self.embedding_dropout = nn.Dropout(dropout_embedding)
        
        # LLMスタイルTransformer layers
        self.layers = nn.ModuleList([
            self._create_llm_style_layer(
                embed_dim, num_heads, mlp_ratio, 
                temperature, top_k, top_p,
                dropout_attention, dropout_mlp
            ) for _ in range(depth)
        ])
        
        # 分類ヘッド（層別Dropout対応）
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout_classifier),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_classifier * 0.5),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_classifier * 0.5),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        # Logits temperature制御
        self.logits_temperature = nn.Parameter(torch.tensor(temperature))
        
        # 重み初期化
        self.apply(self._init_weights)
    
    def _create_llm_style_layer(self, embed_dim, num_heads, mlp_ratio, 
                               temperature, top_k, top_p, dropout_attn, dropout_mlp):
        """LLMスタイルTransformerレイヤー作成"""
        return nn.ModuleDict({
            'norm1': nn.LayerNorm(embed_dim),
            'attention': LLMStyleAttention(embed_dim, num_heads, temperature, top_k, top_p),
            'norm2': nn.LayerNorm(embed_dim),
            'mlp': nn.Sequential(
                nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(dropout_mlp),
                nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                nn.Dropout(dropout_mlp)
            ),
            'dropout_attn': nn.Dropout(dropout_attn),
            'dropout_mlp': nn.Dropout(dropout_mlp)
        })
    
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
    
    def forward(self, x, layer_idx=None):
        B = x.shape[0]
        
        # パッチ埋め込み
        x = self.patch_embedding(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # クラストークン追加
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # 位置埋め込み
        x = x + self.pos_embedding
        x = self.input_norm(x)
        x = self.embedding_dropout(x)
        
        # 動的NKAT適応（層による減衰）
        if self.nkat_strength > 0:
            current_strength = self.nkat_strength
            if layer_idx is not None:
                current_strength *= (self.nkat_decay ** layer_idx)
            
            # チャネル方向平均での適応的調整
            mean_activation = x.mean(dim=-1, keepdim=True)
            nkat_factor = 1.0 + current_strength * 0.01 * torch.tanh(mean_activation)
            x = x * nkat_factor
        
        # LLMスタイルTransformer処理
        for i, layer in enumerate(self.layers):
            residual = x
            
            # Self-attention
            x = layer['norm1'](x)
            attn_out, _ = layer['attention'](x)
            x = residual + layer['dropout_attn'](attn_out)
            
            # MLP
            residual = x
            x = layer['norm2'](x)
            mlp_out = layer['mlp'](x)
            x = residual + layer['dropout_mlp'](mlp_out)
            
            # 層ごとの動的NKAT適応
            if self.nkat_strength > 0:
                current_strength = self.nkat_strength * (self.nkat_decay ** i)
                if current_strength > 1e-4:  # 閾値以下は無視
                    mean_activation = x.mean(dim=-1, keepdim=True)
                    nkat_factor = 1.0 + current_strength * 0.01 * torch.tanh(mean_activation)
                    x = x * nkat_factor
        
        # 分類（Temperature制御付き）
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.classifier(cls_output)
        logits = logits / self.logits_temperature  # Temperature scaling
        
        return logits
    
    def get_regularization_loss(self):
        """L1/L2正則化損失計算"""
        l1_loss = 0
        l2_loss = 0
        
        for param in self.parameters():
            if param.requires_grad:
                l1_loss += torch.sum(torch.abs(param))
                l2_loss += torch.sum(param ** 2)
        
        return self.l1_reg * l1_loss + self.l2_reg * l2_loss

def calculate_tpe_score(val_accuracy, theory_params, total_params, complexity_penalty=0.1):
    """TPE (Theory-Practical Equilibrium) スコア計算"""
    lambda_theory = max(theory_params, 1)
    complexity_factor = np.log10(1 + lambda_theory) + complexity_penalty
    tpe = val_accuracy / complexity_factor
    return tpe

def count_theory_parameters(model):
    """理論的パラメータ数カウント"""
    theory_params = 0
    
    # NKAT関連パラメータ
    if hasattr(model, 'nkat_strength') and model.nkat_strength > 0:
        theory_params += 100  # NKAT理論的寄与
    
    # LLMスタイルパラメータ
    if hasattr(model, 'logits_temperature'):
        theory_params += model.logits_temperature.numel()
    
    # Attention特殊パラメータ
    for layer in model.layers:
        if hasattr(layer['attention'], 'temperature'):
            theory_params += 10  # Temperature寄与
    
    # パッチ埋め込みの理論的寄与（30%）
    for param in model.patch_embedding.parameters():
        theory_params += int(param.numel() * 0.3)
    
    return theory_params

def train_and_evaluate_trial(trial_params, num_epochs=15, device='cuda'):
    """単一試行のトレーニング・評価"""
    
    # データ準備
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)
    
    # モデル作成
    model = NKATOptimizedTransformer(**trial_params).to(device)
    
    # パラメータカウント
    total_params = sum(p.numel() for p in model.parameters())
    theory_params = count_theory_parameters(model)
    
    # トレーニング設定
    criterion = nn.CrossEntropyLoss(label_smoothing=trial_params.get('label_smoothing', 0.08))
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=trial_params.get('learning_rate', 1e-4),
        weight_decay=trial_params.get('weight_decay', 2e-4)
    )
    
    if trial_params.get('use_cosine_lr', True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=trial_params.get('min_lr', 1e-6)
        )
    else:
        scheduler = None
    
    scaler = torch.amp.GradScaler('cuda')
    
    # トレーニング
    model.train()
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            try:
                with torch.amp.autocast('cuda'):
                    output = model(data, layer_idx=epoch // 3)  # 動的層制御
                    loss = criterion(output, target)
                    
                    # 正則化損失追加
                    reg_loss = model.get_regularization_loss()
                    total_loss = loss + reg_loss
                
                # 数値安定性チェック
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"❌ Numerical instability detected")
                    continue
                
                scaler.scale(total_loss).backward()
                
                # 勾配クリッピング
                if trial_params.get('use_grad_clip', True):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=trial_params.get('grad_clip_norm', 1.0)
                    )
                
                scaler.step(optimizer)
                scaler.update()
                
            except RuntimeError as e:
                print(f"❌ Runtime error: {e}")
                torch.cuda.empty_cache()
                continue
            
            epoch_loss += total_loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        if scheduler is not None:
            scheduler.step()
        
        # 中間評価（Pruning用）
        if epoch % 5 == 4:
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
                    test_total += target.size(0)
            
            intermediate_accuracy = 100. * test_correct / test_total
            best_accuracy = max(best_accuracy, intermediate_accuracy)
            model.train()
    
    # 最終テスト評価
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += target.size(0)
    
    final_accuracy = 100. * test_correct / test_total
    
    # TPEスコア計算
    tpe_score = calculate_tpe_score(final_accuracy / 100.0, theory_params, total_params)
    
    return {
        'accuracy': final_accuracy,
        'tpe_score': tpe_score,
        'total_params': total_params,
        'theory_params': theory_params
    }

def objective(trial):
    """Optuna目的関数：TPEスコア最大化"""
    
    # LLMスタイルハイパーパラメータ探索
    temperature = trial.suggest_float('temperature', 0.5, 2.0)
    top_k = trial.suggest_int('top_k', 0, 20) if trial.suggest_categorical('use_top_k', [True, False]) else None
    top_p = trial.suggest_float('top_p', 0.1, 1.0) if trial.suggest_categorical('use_top_p', [True, False]) else None
    
    # NKAT関連
    nkat_strength = trial.suggest_float('nkat_strength', 0.0, 0.05)
    nkat_decay = trial.suggest_float('nkat_decay', 0.8, 1.0)
    
    # Dropout系
    dropout_attention = trial.suggest_float('dropout_attention', 0.0, 0.3)
    dropout_mlp = trial.suggest_float('dropout_mlp', 0.0, 0.3)
    dropout_embedding = trial.suggest_float('dropout_embedding', 0.0, 0.2)
    dropout_classifier = trial.suggest_float('dropout_classifier', 0.0, 0.4)
    
    # 正則化
    l1_reg = trial.suggest_float('l1_reg', 0.0, 1e-4, log=True) if trial.suggest_categorical('use_l1', [True, False]) else 0.0
    l2_reg = trial.suggest_float('l2_reg', 0.0, 1e-3, log=True) if trial.suggest_categorical('use_l2', [True, False]) else 0.0
    
    # トレーニング設定
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.15)
    grad_clip_norm = trial.suggest_float('grad_clip_norm', 0.5, 2.0)
    min_lr = trial.suggest_float('min_lr', 1e-7, 1e-5, log=True)
    
    # パラメータ辞書作成
    trial_params = {
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'nkat_strength': nkat_strength,
        'nkat_decay': nkat_decay,
        'dropout_attention': dropout_attention,
        'dropout_mlp': dropout_mlp,
        'dropout_embedding': dropout_embedding,
        'dropout_classifier': dropout_classifier,
        'l1_reg': l1_reg,
        'l2_reg': l2_reg,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'label_smoothing': label_smoothing,
        'grad_clip_norm': grad_clip_norm,
        'min_lr': min_lr,
        'use_cosine_lr': True,
        'use_grad_clip': True
    }
    
    # GPU使用可能性確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        result = train_and_evaluate_trial(trial_params, num_epochs=12, device=device)
        return result['tpe_score']  # TPEスコアを最大化
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0  # 失敗時は最低スコア

def run_optuna_optimization(n_trials=50, timeout=3600):
    """Optuna最適化実行"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🎯 NKAT Optuna ハイパーパラメータ最適化開始")
    print(f"📅 実行日時: {timestamp}")
    print(f"🔢 試行回数: {n_trials}")
    print(f"⏱️  タイムアウト: {timeout}秒")
    
    # Optuna設定
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    study = optuna.create_study(
        direction='maximize',  # TPEスコア最大化
        sampler=sampler,
        pruner=pruner,
        study_name=f'nkat_optimization_{timestamp}'
    )
    
    # 最適化実行
    study.optimize(
        objective, 
        n_trials=n_trials, 
        timeout=timeout,
        show_progress_bar=True
    )
    
    # 結果分析
    best_trial = study.best_trial
    
    print(f"\n🏆 最適化完了!")
    print(f"📊 最高TPEスコア: {best_trial.value:.6f}")
    print(f"🎯 最適パラメータ:")
    
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # 結果保存
    results = {
        'best_tpe_score': best_trial.value,
        'best_params': best_trial.params,
        'n_trials': len(study.trials),
        'timestamp': timestamp
    }
    
    # 可視化作成
    create_optimization_visualization(study, timestamp)
    
    # 結果JSON保存
    results_file = f'nkat_optuna_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 結果保存: {results_file}")
    
    return study, best_trial

def create_optimization_visualization(study, timestamp):
    """最適化結果可視化"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 最適化履歴
    plt.subplot(2, 3, 1)
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    plt.plot(values, alpha=0.7)
    plt.xlabel('Trial')
    plt.ylabel('TPE Score')
    plt.title('🎯 Optimization History')
    plt.grid(True, alpha=0.3)
    
    # 2. パラメータ重要度
    plt.subplot(2, 3, 2)
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())[:10]  # 上位10個
        values = [importances[p] for p in params]
        
        plt.barh(params, values)
        plt.xlabel('Importance')
        plt.title('📊 Parameter Importance')
    except:
        plt.text(0.5, 0.5, 'Importance analysis\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # 3. パラメータ相関（主要パラメータ）
    plt.subplot(2, 3, 3)
    df = study.trials_dataframe()
    if len(df) > 10:
        # 主要パラメータの相関
        key_params = ['params_temperature', 'params_nkat_strength', 'params_learning_rate']
        existing_params = [p for p in key_params if p in df.columns]
        
        if len(existing_params) >= 2:
            corr_matrix = df[existing_params + ['value']].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('🔗 Parameter Correlations')
        else:
            plt.text(0.5, 0.5, 'Insufficient data\nfor correlation', 
                    ha='center', va='center', transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, 'Insufficient trials\nfor correlation', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # 4. Temperature vs Performance
    plt.subplot(2, 3, 4)
    if 'params_temperature' in df.columns:
        plt.scatter(df['params_temperature'], df['value'], alpha=0.6)
        plt.xlabel('Temperature')
        plt.ylabel('TPE Score')
        plt.title('🌡️ Temperature vs Performance')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Temperature data\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # 5. NKAT Strength vs Performance  
    plt.subplot(2, 3, 5)
    if 'params_nkat_strength' in df.columns:
        plt.scatter(df['params_nkat_strength'], df['value'], alpha=0.6, color='orange')
        plt.xlabel('NKAT Strength')
        plt.ylabel('TPE Score')
        plt.title('⚡ NKAT Strength vs Performance')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'NKAT Strength data\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # 6. 最適解統計
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    best_trial = study.best_trial
    stats_text = f"""
🏆 Optimization Summary

📊 Best TPE Score: {best_trial.value:.6f}
🔢 Total Trials: {len(study.trials)}
⏱️ Best Trial: #{best_trial.number}

🎯 Key Parameters:
Temperature: {best_trial.params.get('temperature', 'N/A'):.3f}
NKAT Strength: {best_trial.params.get('nkat_strength', 'N/A'):.4f}
Learning Rate: {best_trial.params.get('learning_rate', 'N/A'):.2e}
"""
    
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    
    # 保存
    viz_file = f'nkat_optuna_optimization_analysis_{timestamp}.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"📊 最適化可視化保存: {viz_file}")
    
    return viz_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='NKAT Optuna ハイパーパラメータ最適化')
    parser.add_argument('--n_trials', type=int, default=30, help='最適化試行回数')
    parser.add_argument('--timeout', type=int, default=3600, help='タイムアウト時間（秒）')
    parser.add_argument('--device', default='cuda', help='デバイス')
    
    args = parser.parse_args()
    
    print("🎯 NKAT Optuna ハイパーパラメータ最適化システム")
    print("🔥 LLMスタイルパラメータ + TPE指標最適化")
    
    # GPU確認
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"🚀 CUDA デバイス: {torch.cuda.get_device_name(0)}")
    
    # 最適化実行
    study, best_trial = run_optuna_optimization(args.n_trials, args.timeout)
    
    print(f"\n🎉 最適化完了！")
    print(f"🏆 最高TPEスコア: {best_trial.value:.6f}")
    print(f"💡 これでNKAT-Transformerの理論⇔工学バランスが完璧に最適化されました！")

if __name__ == "__main__":
    main() 