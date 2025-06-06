#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT機械学習クイック分析システム
38,832個のゼロ点データ迅速評価
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

def quick_ml_analysis():
    """迅速機械学習分析"""
    print("🚀 NKAT機械学習クイック分析開始")
    
    # データ読み込み
    try:
        with open("nkat_production_results_nkat_prod_20250604_102015.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        zeros_raw = data['results']['zeros_data']
        df = pd.DataFrame(zeros_raw)
        print(f"✅ データ読み込み: {len(df):,}個のゼロ点")
        
        # 基本統計
        print(f"📊 基本統計:")
        print(f"   t値範囲: {df['t'].min():.3f} - {df['t'].max():.3f}")
        print(f"   平均信頼度: {df['confidence'].mean():.6f}")
        print(f"   平均残差: {df['residual'].mean():.2e}")
        
        # 既知ゼロ点マッチング
        if 'known_match' in df.columns:
            known_matches = df['known_match'].fillna(0).sum()
            print(f"   既知ゼロ点一致: {known_matches}/{len(df)} ({known_matches/len(df)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return None
    
    # 簡易特徴量作成
    features = {
        't_value': df['t'].values,
        'residual': df['residual'].values,
        'confidence': df['confidence'].values,
        'superconv_factor': df['superconv_factor'].values
    }
    
    # 追加特徴量
    t_diffs = np.diff(df['t'].values)
    features['zero_spacing'] = np.concatenate([[t_diffs[0]], t_diffs])
    features['residual_log'] = np.log10(df['residual'] + 1e-16)
    features['confidence_log'] = np.log10(df['confidence'] + 1e-16)
    
    # フィーチャー行列
    X = np.column_stack([features['t_value'], features['residual'], 
                        features['confidence'], features['superconv_factor'],
                        features['zero_spacing'], features['residual_log'],
                        features['confidence_log']])
    
    feature_names = ['t_value', 'residual', 'confidence', 'superconv_factor',
                    'zero_spacing', 'residual_log', 'confidence_log']
    
    print(f"🔬 特徴量: {len(feature_names)}個")
    
    # 複数の分類タスク
    targets = {}
    
    # 1. 高信頼度ゼロ点（上位25%）
    confidence_threshold = np.percentile(df['confidence'], 75)
    targets['high_confidence'] = (df['confidence'] > confidence_threshold).astype(int)
    
    # 2. 低残差ゼロ点（下位25%）
    residual_threshold = np.percentile(df['residual'], 25)
    targets['low_residual'] = (df['residual'] < residual_threshold).astype(int)
    
    # 3. 既知ゼロ点マッチング
    if 'known_match' in df.columns:
        targets['known_match'] = df['known_match'].fillna(0).astype(int)
    
    print(f"🎯 分類タスク: {len(targets)}個")
    
    # 各ターゲットで評価
    results = {}
    
    for target_name, y in targets.items():
        print(f"\n🔍 {target_name} 分析開始...")
        
        # データ準備
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # RandomForest 訓練
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # 予測
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        
        # 評価メトリクス
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.5
        
        accuracy = (y_pred == y_test).mean()
        
        # 混同行列
        cm = confusion_matrix(y_test, y_pred)
        
        # 特徴量重要度
        importances = rf.feature_importances_
        
        results[target_name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'feature_importance': dict(zip(feature_names, importances)),
            'positive_rate': y.mean()
        }
        
        print(f"   精度: {accuracy:.3f}")
        print(f"   ROC AUC: {roc_auc:.3f}")
        print(f"   正例率: {y.mean():.3f}")
        
        # Top 3 重要特徴量
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:3]
        print(f"   重要特徴量:")
        for i, (name, importance) in enumerate(top_features):
            print(f"     {i+1}. {name}: {importance:.3f}")
    
    # 可視化
    plt.figure(figsize=(15, 10))
    
    # ROC AUC比較
    plt.subplot(2, 3, 1)
    target_names = list(results.keys())
    roc_values = [results[name]['roc_auc'] for name in target_names]
    plt.bar(target_names, roc_values)
    plt.title('ROC AUC Comparison')
    plt.ylabel('ROC AUC')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 精度比較
    plt.subplot(2, 3, 2)
    accuracy_values = [results[name]['accuracy'] for name in target_names]
    plt.bar(target_names, accuracy_values)
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 特徴量重要度（最初のタスク）
    if target_names:
        first_target = target_names[0]
        importances = results[first_target]['feature_importance']
        plt.subplot(2, 3, 3)
        plt.bar(importances.keys(), importances.values())
        plt.title(f'Feature Importance: {first_target}')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 混同行列（最初のタスク）
    if target_names:
        cm = np.array(results[first_target]['confusion_matrix'])
        plt.subplot(2, 3, 4)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {first_target}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    # データ分布
    plt.subplot(2, 3, 5)
    plt.hist(df['confidence'], bins=50, alpha=0.7, label='Confidence')
    plt.hist(np.log10(df['residual']), bins=50, alpha=0.7, label='Log Residual')
    plt.title('Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ゼロ点分布
    plt.subplot(2, 3, 6)
    plt.scatter(df['t'], df['confidence'], alpha=0.1, s=1)
    plt.title('Zero Distribution')
    plt.xlabel('t (imaginary part)')
    plt.ylabel('Confidence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nkat_ml_quick_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 結果サマリー
    print(f"\n🎊 NKAT機械学習クイック分析完了!")
    print(f"📊 検出ゼロ点: {len(df):,}個")
    print(f"🔬 特徴量数: {len(feature_names)}")
    print(f"🎯 分類タスク数: {len(targets)}")
    print(f"📈 可視化保存: nkat_ml_quick_analysis.png")
    
    print(f"\n🏆 ベスト性能:")
    for target_name in target_names:
        roc_auc = results[target_name]['roc_auc']
        accuracy = results[target_name]['accuracy']
        print(f"   {target_name}: ROC AUC={roc_auc:.3f}, Accuracy={accuracy:.3f}")
    
    return results

if __name__ == "__main__":
    results = quick_ml_analysis() 