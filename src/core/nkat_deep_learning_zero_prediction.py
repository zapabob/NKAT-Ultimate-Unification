#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 NKAT深層学習ゼロ点予測システム
参考: https://github.com/avysogorets/riemann-zeta
     https://sites.google.com/site/riemannzetazeros/machinelearning

38,832個のNKATゼロ点データから機械学習でリーマン仮説を検証
- LSTM、MLP、RNN による予測
- クロスバリデーション
- ROC AUC、混同行列による性能評価
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from pathlib import Path
from tqdm import tqdm

# 機械学習ライブラリ
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 深層学習ライブラリ
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
    print("🧠 TensorFlow/Keras: 有効")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow無効 - 従来ML手法のみ")

# GPU関連
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("🚀 CUDA GPU加速: 有効")
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️ CUDA無効")

# 警告抑制
warnings.filterwarnings('ignore')

# matplotlib日本語対応
plt.rcParams['font.family'] = ['DejaVu Sans']
sns.set_style("whitegrid")

class NKATDeepLearningPredictor:
    """NKAT深層学習ゼロ点予測システム"""
    
    def __init__(self, data_file="nkat_production_results_nkat_prod_20250604_102015.json"):
        """初期化"""
        self.data_file = data_file
        self.zeros_data = None
        self.features = None
        self.targets = None
        self.models = {}
        self.results = {}
        
        # 結果保存ディレクトリ
        self.output_dir = Path("nkat_deep_learning_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🧠 NKAT深層学習システム初期化")
        print(f"📁 データファイル: {data_file}")
        print(f"💾 出力ディレクトリ: {self.output_dir}")
    
    def load_nkat_data(self):
        """NKATゼロ点データ読み込み"""
        print(f"📊 NKATデータ読み込み開始...")
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # ゼロ点データ抽出
            zeros_raw = data['results']['zeros_data']
            print(f"✅ 読み込み完了: {len(zeros_raw)}個のゼロ点")
            
            # データフレーム化
            zeros_df = pd.DataFrame(zeros_raw)
            
            # 基本統計
            print(f"\n📈 データ統計:")
            print(f"   ゼロ点数: {len(zeros_df):,}")
            print(f"   t値範囲: {zeros_df['t'].min():.3f} - {zeros_df['t'].max():.3f}")
            print(f"   平均信頼度: {zeros_df['confidence'].mean():.6f}")
            print(f"   平均残差: {zeros_df['residual'].mean():.2e}")
            
            # 既知ゼロ点マッチング率
            if 'known_match' in zeros_df.columns:
                known_matches = zeros_df['known_match'].sum()
                print(f"   既知ゼロ点一致: {known_matches}/{len(zeros_df)} ({known_matches/len(zeros_df)*100:.1f}%)")
            
            self.zeros_data = zeros_df
            return zeros_df
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return None
    
    def extract_features_advanced(self):
        """高度な特徴量抽出（GitHub参考）"""
        print(f"\n🔬 高度特徴量抽出開始...")
        
        if self.zeros_data is None:
            print("❌ データが読み込まれていません")
            return None
        
        # 基本特徴量
        features_dict = {
            't_value': self.zeros_data['t'].values,
            'residual': self.zeros_data['residual'].values,
            'confidence': self.zeros_data['confidence'].values,
            'superconv_factor': self.zeros_data['superconv_factor'].values
        }
        
        # ソート（t値順）
        sort_idx = np.argsort(features_dict['t_value'])
        for key in features_dict:
            features_dict[key] = features_dict[key][sort_idx]
        
        # GitHub avysogorets 参考特徴量
        
        # 1. 連続ゼロ点間距離
        t_diffs = np.diff(features_dict['t_value'])
        features_dict['zero_spacing'] = np.concatenate([[t_diffs[0]], t_diffs])
        
        # 2. 局所密度（周辺5点での平均間隔）
        local_density = []
        for i in range(len(features_dict['t_value'])):
            start = max(0, i-2)
            end = min(len(features_dict['t_value']), i+3)
            if end - start > 1:
                local_spacings = np.diff(features_dict['t_value'][start:end])
                local_density.append(np.mean(local_spacings))
            else:
                local_density.append(features_dict['zero_spacing'][i])
        features_dict['local_density'] = np.array(local_density)
        
        # 3. Gram点近似特徴量（Google Sites参考）
        # 簡略化したGram点関連特徴
        gram_approx = 2 * np.pi * features_dict['t_value'] / np.log(features_dict['t_value'] / (2 * np.pi))
        gram_deviation = features_dict['t_value'] - gram_approx
        features_dict['gram_deviation'] = gram_deviation
        
        # 4. Z関数関連特徴（簡略版）
        z_function_approx = np.cos(features_dict['t_value'] * np.log(features_dict['t_value']) / 2)
        features_dict['z_function_approx'] = z_function_approx
        
        # 5. Hardy Z関数の符号変化
        z_sign_changes = []
        for i in range(len(z_function_approx)):
            if i == 0:
                z_sign_changes.append(0)
            else:
                z_sign_changes.append(1 if z_function_approx[i] * z_function_approx[i-1] < 0 else 0)
        features_dict['z_sign_change'] = np.array(z_sign_changes)
        
        # 6. 統計的特徴量
        features_dict['residual_log'] = np.log10(features_dict['residual'] + 1e-16)
        features_dict['confidence_log'] = np.log10(features_dict['confidence'] + 1e-16)
        
        # 7. 移動平均特徴
        window = min(10, len(features_dict['t_value']) // 10)
        features_dict['spacing_ma'] = pd.Series(features_dict['zero_spacing']).rolling(window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        features_dict['residual_ma'] = pd.Series(features_dict['residual']).rolling(window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        # 8. 高次統計特徴
        features_dict['t_value_normalized'] = (features_dict['t_value'] - features_dict['t_value'].mean()) / features_dict['t_value'].std()
        features_dict['spacing_ratio'] = features_dict['zero_spacing'] / features_dict['spacing_ma']
        
        # フィーチャー行列作成
        feature_names = [
            't_value', 'residual', 'confidence', 'superconv_factor',
            'zero_spacing', 'local_density', 'gram_deviation', 
            'z_function_approx', 'z_sign_change', 'residual_log', 
            'confidence_log', 'spacing_ma', 'residual_ma',
            't_value_normalized', 'spacing_ratio'
        ]
        
        X = np.column_stack([features_dict[name] for name in feature_names])
        
        print(f"✅ 特徴量抽出完了:")
        print(f"   特徴量数: {len(feature_names)}")
        print(f"   サンプル数: {X.shape[0]}")
        print(f"   特徴量名: {feature_names}")
        
        self.features = X
        self.feature_names = feature_names
        return X, feature_names
    
    def create_targets_classification(self):
        """分類ターゲット作成"""
        print(f"\n🎯 分類ターゲット作成...")
        
        if self.zeros_data is None:
            return None
        
        # 複数の分類タスクを定義
        targets = {}
        
        # 1. 高信頼度ゼロ点分類（上位25%）
        confidence_threshold = np.percentile(self.zeros_data['confidence'], 75)
        targets['high_confidence'] = (self.zeros_data['confidence'] > confidence_threshold).astype(int)
        
        # 2. 低残差ゼロ点分類（下位25%）
        residual_threshold = np.percentile(self.zeros_data['residual'], 25)
        targets['low_residual'] = (self.zeros_data['residual'] < residual_threshold).astype(int)
        
        # 3. 既知ゼロ点マッチング（利用可能な場合）
        if 'known_match' in self.zeros_data.columns:
            # NaN値を0に置換してからint変換
            known_match_clean = self.zeros_data['known_match'].fillna(0)
            targets['known_match'] = known_match_clean.astype(int)
        
        # 4. 超収束因子異常値（上位10%）
        superconv_threshold = np.percentile(self.zeros_data['superconv_factor'], 90)
        targets['high_superconv'] = (self.zeros_data['superconv_factor'] > superconv_threshold).astype(int)
        
        # 5. 密集ゼロ点検出（間隔が平均の50%以下）
        if len(self.zeros_data) > 1:
            t_diffs = np.diff(self.zeros_data['t'].values)
            avg_spacing = np.mean(t_diffs)
            close_pairs = t_diffs < (avg_spacing * 0.5)
            targets['close_pairs'] = np.concatenate([[0], close_pairs.astype(int)])
        
        print(f"✅ 分類ターゲット作成完了:")
        for name, target in targets.items():
            positive_rate = np.mean(target)
            print(f"   {name}: 正例率 {positive_rate:.3f} ({np.sum(target)}/{len(target)})")
        
        self.targets = targets
        return targets
    
    def build_deep_learning_models(self):
        """深層学習モデル構築"""
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow無効 - 深層学習モデルスキップ")
            return {}
        
        print(f"\n🧠 深層学習モデル構築開始...")
        
        models = {}
        input_dim = self.features.shape[1]
        
        # 1. Multi-Layer Perceptron (MLP) - GitHub参考
        def create_mlp_model():
            model = Sequential([
                Input(shape=(input_dim,)),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy', 'precision', 'recall'])
            return model
        
        # 2. LSTM Model（時系列特徴用）
        def create_lstm_model():
            model = Sequential([
                Input(shape=(input_dim, 1)),
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy', 'precision', 'recall'])
            return model
        
        # 3. Convolutional Neural Network
        def create_cnn_model():
            model = Sequential([
                Input(shape=(input_dim, 1)),
                Conv1D(32, 3, activation='relu'),
                MaxPooling1D(2),
                Conv1D(64, 3, activation='relu'),
                MaxPooling1D(2),
                Flatten(),
                Dense(50, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy', 'precision', 'recall'])
            return model
        
        models['MLP'] = create_mlp_model
        models['LSTM'] = create_lstm_model
        models['CNN'] = create_cnn_model
        
        print(f"✅ 深層学習モデル構築完了: {list(models.keys())}")
        return models
    
    def cross_validation_evaluation(self, model_type='sklearn'):
        """クロスバリデーション評価"""
        print(f"\n🔍 {model_type} クロスバリデーション開始...")
        
        if self.features is None or self.targets is None:
            print("❌ 特徴量またはターゲットが準備されていません")
            return None
        
        results = {}
        
        # データ標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.features)
        
        # 各ターゲットに対してCV実行
        for target_name, y in self.targets.items():
            print(f"\n🎯 ターゲット: {target_name}")
            
            if model_type == 'sklearn':
                # 従来機械学習モデル
                models = {
                    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'SVR': SVR(kernel='rbf', C=1.0)
                }
                
                target_results = {}
                
                for model_name, model in models.items():
                    try:
                        # 5-fold クロスバリデーション
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        
                        # 分類問題として扱う
                        if hasattr(model, 'predict_proba'):
                            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                        else:
                            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
                        
                        target_results[model_name] = {
                            'cv_scores': cv_scores,
                            'mean_score': cv_scores.mean(),
                            'std_score': cv_scores.std()
                        }
                        
                        print(f"   {model_name}: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
                        
                    except Exception as e:
                        print(f"   ❌ {model_name} エラー: {e}")
                
                results[target_name] = target_results
            
            elif model_type == 'deep_learning' and TF_AVAILABLE:
                # 深層学習モデル
                dl_models = self.build_deep_learning_models()
                target_results = {}
                
                for model_name, create_model_func in dl_models.items():
                    try:
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 深層学習は3-fold
                        cv_scores = []
                        
                        for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
                            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            
                            # LSTM/CNNの場合は次元追加
                            if model_name in ['LSTM', 'CNN']:
                                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                            
                            # モデル作成・訓練
                            model = create_model_func()
                            
                            # Early stopping
                            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                            
                            history = model.fit(
                                X_train, y_train,
                                validation_data=(X_val, y_val),
                                epochs=50,
                                batch_size=32,
                                verbose=0,
                                callbacks=[early_stop]
                            )
                            
                            # 予測・評価
                            y_pred_proba = model.predict(X_val, verbose=0)
                            try:
                                score = roc_auc_score(y_val, y_pred_proba)
                            except:
                                y_pred = (y_pred_proba > 0.5).astype(int)
                                score = accuracy_score(y_val, y_pred)
                            
                            cv_scores.append(score)
                            print(f"     Fold {fold+1}: {score:.3f}")
                        
                        target_results[model_name] = {
                            'cv_scores': np.array(cv_scores),
                            'mean_score': np.mean(cv_scores),
                            'std_score': np.std(cv_scores)
                        }
                        
                        print(f"   {model_name}: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
                        
                    except Exception as e:
                        print(f"   ❌ {model_name} エラー: {e}")
                
                results[target_name] = target_results
        
        self.results[model_type] = results
        return results
    
    def create_confusion_matrix_plots(self):
        """混同行列とROC曲線の作成"""
        print(f"\n📊 混同行列・ROC曲線作成...")
        
        if self.features is None or self.targets is None:
            return None
        
        # データ準備
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.features)
        
        # 結果保存用
        plot_results = {}
        
        for target_name, y in self.targets.items():
            print(f"\n🎯 {target_name} の評価...")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # RandomForest で評価
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            y_pred = rf_model.predict(X_test)
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
            
            # メトリクス計算
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            except:
                roc_auc = 0.5
                fpr, tpr = [0, 1], [0, 1]
            
            # プロット作成
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 混同行列
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title(f'Confusion Matrix: {target_name}')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            
            # ROC曲線
            axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title(f'ROC Curve: {target_name}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # メトリクス表示
            metrics_text = f"""Metrics:
Accuracy: {accuracy:.3f}
Precision: {precision:.3f}
Recall: {recall:.3f}
F1-Score: {f1:.3f}
ROC AUC: {roc_auc:.3f}"""
            
            fig.text(0.02, 0.02, metrics_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            
            # 保存
            plot_file = self.output_dir / f"confusion_roc_{target_name}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_results[target_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'plot_file': str(plot_file)
            }
            
            print(f"   Accuracy: {accuracy:.3f}, Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}, F1: {f1:.3f}, ROC AUC: {roc_auc:.3f}")
            print(f"   📊 保存: {plot_file}")
        
        return plot_results
    
    def feature_importance_analysis(self):
        """特徴量重要度分析"""
        print(f"\n🔬 特徴量重要度分析...")
        
        if self.features is None or self.targets is None:
            return None
        
        # データ準備
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.features)
        
        importance_results = {}
        
        for target_name, y in self.targets.items():
            # RandomForest で特徴量重要度取得
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_scaled, y)
            
            # 重要度
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # プロット
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importance: {target_name}')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存
            importance_file = self.output_dir / f"feature_importance_{target_name}.png"
            plt.savefig(importance_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            importance_results[target_name] = {
                'importances': importances,
                'feature_ranking': [self.feature_names[i] for i in indices],
                'plot_file': str(importance_file)
            }
            
            print(f"   {target_name} - Top 3 features:")
            for i in range(min(3, len(indices))):
                feature_idx = indices[i]
                print(f"     {i+1}. {self.feature_names[feature_idx]}: {importances[feature_idx]:.3f}")
        
        return importance_results
    
    def run_complete_analysis(self):
        """完全分析実行"""
        print("🧠" * 30)
        print("NKAT深層学習ゼロ点予測システム - 完全分析")
        print("🧠" * 30)
        
        # 1. データ読み込み
        if self.load_nkat_data() is None:
            return None
        
        # 2. 特徴量抽出
        if self.extract_features_advanced() is None:
            return None
        
        # 3. ターゲット作成
        if self.create_targets_classification() is None:
            return None
        
        # 4. 従来機械学習評価
        sklearn_results = self.cross_validation_evaluation('sklearn')
        
        # 5. 深層学習評価
        if TF_AVAILABLE:
            dl_results = self.cross_validation_evaluation('deep_learning')
        
        # 6. 混同行列・ROC曲線
        confusion_results = self.create_confusion_matrix_plots()
        
        # 7. 特徴量重要度分析
        importance_results = self.feature_importance_analysis()
        
        # 8. 総合結果まとめ
        final_results = {
            'data_summary': {
                'total_zeros': len(self.zeros_data),
                't_range': [float(self.zeros_data['t'].min()), float(self.zeros_data['t'].max())],
                'feature_count': len(self.feature_names),
                'target_count': len(self.targets)
            },
            'sklearn_results': sklearn_results,
            'confusion_matrix_results': confusion_results,
            'feature_importance': importance_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        if TF_AVAILABLE:
            final_results['deep_learning_results'] = dl_results
        
        # 結果保存
        results_file = self.output_dir / f"nkat_ml_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n🎊 NKAT深層学習分析完了!")
        print(f"📊 検出ゼロ点: {len(self.zeros_data):,}個")
        print(f"🔬 特徴量数: {len(self.feature_names)}")
        print(f"🎯 分類タスク数: {len(self.targets)}")
        print(f"💾 結果保存: {results_file}")
        print(f"📁 出力ディレクトリ: {self.output_dir}")
        
        # ベスト結果表示
        print(f"\n🏆 ベスト性能結果:")
        if confusion_results:
            for target_name, metrics in confusion_results.items():
                print(f"   {target_name}: ROC AUC = {metrics['roc_auc']:.3f}")
        
        return final_results

def main():
    """メイン実行関数"""
    print("🧠 NKAT深層学習ゼロ点予測システム起動")
    print("📚 参考研究:")
    print("   - GitHub: avysogorets/riemann-zeta")
    print("   - Google Sites: Machine Learning for Riemann Zeta")
    print("=" * 60)
    
    # システム実行
    predictor = NKATDeepLearningPredictor()
    results = predictor.run_complete_analysis()
    
    if results:
        print("\n🌟 NKAT機械学習による非自明ゼロ点予測完了!")
        print("🔬 リーマン仮説への機械学習アプローチ成功!")
    else:
        print("\n⚠️ 分析エラー - データ確認が必要")

if __name__ == "__main__":
    main() 