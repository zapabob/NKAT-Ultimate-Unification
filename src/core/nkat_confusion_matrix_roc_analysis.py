#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 NKAT混同行列・ROC AUC詳細分析システム
38,832個のゼロ点データでの深層性能評価
参考: Medium記事 "Predicting Riemann Zeta Function Zeros: A Machine Learning Odyssey"
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# matplotlib日本語設定
plt.rcParams['font.family'] = ['DejaVu Sans']
sns.set_style("whitegrid")

def load_nkat_data():
    """NKATデータ読み込み"""
    print("📊 38,832個のNKATゼロ点データ読み込み...")
    
    try:
        with open("nkat_production_results_nkat_prod_20250604_102015.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        zeros_raw = data['results']['zeros_data']
        df = pd.DataFrame(zeros_raw)
        print(f"✅ 読み込み完了: {len(df):,}個のゼロ点")
        
        return df
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return None

def extract_advanced_features(df):
    """高度特徴量抽出（GitHub参考）"""
    print("🔬 GitHub avysogorets/riemann-zeta参考特徴量抽出...")
    
    # ソート
    df_sorted = df.sort_values('t').reset_index(drop=True)
    
    features = {}
    
    # 基本特徴量
    features['t_value'] = df_sorted['t'].values
    features['residual'] = df_sorted['residual'].values
    features['confidence'] = df_sorted['confidence'].values
    features['superconv_factor'] = df_sorted['superconv_factor'].values
    
    # ゼロ点間隔特徴量
    t_diffs = np.diff(features['t_value'])
    features['zero_spacing'] = np.concatenate([[t_diffs[0]], t_diffs])
    
    # 統計的特徴量
    features['residual_log'] = np.log10(features['residual'] + 1e-16)
    features['confidence_log'] = np.log10(features['confidence'] + 1e-16)
    
    # Gram点近似（Google Sites参考）
    gram_approx = 2 * np.pi * features['t_value'] / np.log(features['t_value'] / (2 * np.pi))
    features['gram_deviation'] = features['t_value'] - gram_approx
    
    # Hardy Z関数近似
    z_approx = np.cos(features['t_value'] * np.log(features['t_value']) / 2)
    features['z_function_approx'] = z_approx
    
    # 局所密度
    window = 5
    local_density = []
    for i in range(len(features['t_value'])):
        start = max(0, i - window//2)
        end = min(len(features['t_value']), i + window//2 + 1)
        if end - start > 1:
            local_spacings = np.diff(features['t_value'][start:end])
            local_density.append(np.mean(local_spacings))
        else:
            local_density.append(features['zero_spacing'][i])
    features['local_density'] = np.array(local_density)
    
    # 正規化特徴量
    features['t_normalized'] = (features['t_value'] - features['t_value'].mean()) / features['t_value'].std()
    features['spacing_normalized'] = (features['zero_spacing'] - np.mean(features['zero_spacing'])) / np.std(features['zero_spacing'])
    
    feature_names = ['t_value', 'residual', 'confidence', 'superconv_factor',
                    'zero_spacing', 'residual_log', 'confidence_log', 
                    'gram_deviation', 'z_function_approx', 'local_density',
                    't_normalized', 'spacing_normalized']
    
    X = np.column_stack([features[name] for name in feature_names])
    
    print(f"✅ 抽出完了: {len(feature_names)}特徴量 × {X.shape[0]}サンプル")
    
    return X, feature_names, df_sorted

def create_classification_targets(df):
    """分類ターゲット作成"""
    print("🎯 分類ターゲット作成...")
    
    targets = {}
    
    # 1. 高信頼度ゼロ点（上位20%）
    confidence_threshold = np.percentile(df['confidence'], 80)
    targets['high_confidence'] = (df['confidence'] > confidence_threshold).astype(int)
    
    # 2. 極低残差ゼロ点（下位10%）
    residual_threshold = np.percentile(df['residual'], 10)
    targets['ultra_low_residual'] = (df['residual'] < residual_threshold).astype(int)
    
    # 3. 既知ゼロ点マッチング
    if 'known_match' in df.columns:
        targets['known_match'] = df['known_match'].fillna(0).astype(int)
    
    # 4. 超収束異常値（上位5%）
    superconv_threshold = np.percentile(df['superconv_factor'], 95)
    targets['superconv_outlier'] = (df['superconv_factor'] > superconv_threshold).astype(int)
    
    # 5. 密集ゼロ点ペア
    t_diffs = np.diff(df['t'].values)
    avg_spacing = np.mean(t_diffs)
    close_threshold = avg_spacing * 0.3  # 平均の30%以下
    close_pairs = t_diffs < close_threshold
    targets['close_pairs'] = np.concatenate([[0], close_pairs.astype(int)])
    
    print(f"✅ 作成完了: {len(targets)}個の分類タスク")
    for name, target in targets.items():
        positive_rate = np.mean(target)
        print(f"   {name}: 正例率 {positive_rate:.3f} ({np.sum(target):,}/{len(target):,})")
    
    return targets

def detailed_confusion_matrix_analysis(X, targets, feature_names):
    """詳細混同行列・ROC AUC分析"""
    print("\n📊 詳細混同行列・ROC AUC分析開始...")
    
    # 複数モデルでの評価
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    all_results = {}
    
    # データ標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for target_name, y in targets.items():
        print(f"\n🎯 {target_name} 詳細分析...")
        
        target_results = {}
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 各モデルで評価
        for model_name, model in models.items():
            print(f"   🔍 {model_name} 評価中...")
            
            try:
                # 訓練
                model.fit(X_train, y_train)
                
                # 予測
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # メトリクス計算
                accuracy = (y_pred == y_test).mean()
                
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                    avg_precision = average_precision_score(y_test, y_pred_proba)
                except:
                    roc_auc = 0.5
                    fpr, tpr = [0, 1], [0, 1]
                    precision, recall = [1, 0], [0, 1]
                    avg_precision = np.mean(y_test)
                
                # 混同行列
                cm = confusion_matrix(y_test, y_pred)
                
                # 分類レポート
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                # クロスバリデーション
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                
                target_results[model_name] = {
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'avg_precision': avg_precision,
                    'confusion_matrix': cm,
                    'classification_report': class_report,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'fpr': fpr,
                    'tpr': tpr,
                    'precision_curve': precision,
                    'recall_curve': recall
                }
                
                print(f"     Accuracy: {accuracy:.3f}")
                print(f"     ROC AUC: {roc_auc:.3f} (CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f})")
                print(f"     Avg Precision: {avg_precision:.3f}")
                
            except Exception as e:
                print(f"     ❌ {model_name} エラー: {e}")
        
        all_results[target_name] = target_results
    
    return all_results

def create_comprehensive_visualization(results, feature_names):
    """包括的可視化作成"""
    print("\n📈 包括的可視化作成中...")
    
    target_names = list(results.keys())
    model_names = ['RandomForest', 'SVM', 'LogisticRegression']
    
    # 大きなフィギュア作成
    fig = plt.figure(figsize=(20, 24))
    
    plot_idx = 1
    
    for i, target_name in enumerate(target_names):
        target_results = results[target_name]
        
        # ROC曲線
        plt.subplot(6, 4, plot_idx)
        for model_name in model_names:
            if model_name in target_results:
                res = target_results[model_name]
                plt.plot(res['fpr'], res['tpr'], 
                        label=f"{model_name} (AUC={res['roc_auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {target_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_idx += 1
        
        # Precision-Recall曲線
        plt.subplot(6, 4, plot_idx)
        for model_name in model_names:
            if model_name in target_results:
                res = target_results[model_name]
                plt.plot(res['recall_curve'], res['precision_curve'],
                        label=f"{model_name} (AP={res['avg_precision']:.3f})")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall: {target_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_idx += 1
        
        # 混同行列（RandomForest）
        plt.subplot(6, 4, plot_idx)
        if 'RandomForest' in target_results:
            cm = target_results['RandomForest']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix (RF): {target_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        plot_idx += 1
        
        # クロスバリデーション結果
        plt.subplot(6, 4, plot_idx)
        cv_means = []
        cv_stds = []
        model_labels = []
        for model_name in model_names:
            if model_name in target_results:
                res = target_results[model_name]
                cv_means.append(res['cv_mean'])
                cv_stds.append(res['cv_std'])
                model_labels.append(model_name)
        
        plt.bar(model_labels, cv_means, yerr=cv_stds, capsize=5)
        plt.title(f'Cross-Validation ROC AUC: {target_name}')
        plt.ylabel('ROC AUC')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plot_idx += 1
    
    # 全体性能比較
    plt.subplot(6, 4, plot_idx)
    roc_auc_matrix = []
    for model_name in model_names:
        model_aucs = []
        for target_name in target_names:
            if model_name in results[target_name]:
                model_aucs.append(results[target_name][model_name]['roc_auc'])
            else:
                model_aucs.append(0.5)
        roc_auc_matrix.append(model_aucs)
    
    roc_auc_matrix = np.array(roc_auc_matrix)
    sns.heatmap(roc_auc_matrix, xticklabels=target_names, yticklabels=model_names,
                annot=True, cmap='YlOrRd', vmin=0.5, vmax=1.0)
    plt.title('ROC AUC Heatmap (All Models vs Targets)')
    plt.xticks(rotation=45)
    plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('nkat_comprehensive_confusion_roc_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 包括的可視化保存: nkat_comprehensive_confusion_roc_analysis.png")

def print_detailed_summary(results):
    """詳細サマリー表示"""
    print("\n" + "🏆" * 50)
    print("NKAT機械学習性能詳細サマリー")
    print("🏆" * 50)
    
    for target_name, target_results in results.items():
        print(f"\n📊 {target_name}:")
        
        best_model = None
        best_auc = 0
        
        for model_name, res in target_results.items():
            roc_auc = res['roc_auc']
            cv_mean = res['cv_mean']
            cv_std = res['cv_std']
            accuracy = res['accuracy']
            
            print(f"   {model_name:15}: ROC AUC={roc_auc:.3f} (CV: {cv_mean:.3f}±{cv_std:.3f}) Acc={accuracy:.3f}")
            
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_model = model_name
        
        if best_model:
            print(f"   🥇 ベストモデル: {best_model} (ROC AUC={best_auc:.3f})")

def main():
    """メイン実行関数"""
    print("📊" * 30)
    print("NKAT混同行列・ROC AUC詳細分析システム")
    print("参考: GitHub avysogorets/riemann-zeta & Medium記事")
    print("📊" * 30)
    
    # データ読み込み
    df = load_nkat_data()
    if df is None:
        return
    
    # 特徴量抽出
    X, feature_names, df_sorted = extract_advanced_features(df)
    
    # ターゲット作成
    targets = create_classification_targets(df_sorted)
    
    # 詳細分析
    results = detailed_confusion_matrix_analysis(X, targets, feature_names)
    
    # 可視化
    create_comprehensive_visualization(results, feature_names)
    
    # サマリー
    print_detailed_summary(results)
    
    print(f"\n🎊 NKAT機械学習詳細分析完了!")
    print(f"📊 分析データ: 38,832個のゼロ点")
    print(f"🔬 特徴量数: {len(feature_names)}")
    print(f"🎯 分類タスク数: {len(targets)}")
    print(f"🤖 評価モデル数: 3 (RandomForest, SVM, LogisticRegression)")
    print(f"📈 可視化ファイル: nkat_comprehensive_confusion_roc_analysis.png")
    
    return results

if __name__ == "__main__":
    results = main() 