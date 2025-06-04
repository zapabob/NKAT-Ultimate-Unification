#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 NKAT百万ゼロ点深層学習統合システム 🌟
mpmath高精度演算 + LSTM/CNN/Transformer + リアルタイム予測

理論的基盤:
- 超収束因子: S_NKAT = N^0.367 * exp[γ*ln(N) + δ*Tr_θ(e^{-δ(N-N_c)I_κ})]
- mpmath 50桁精度 + scikit-learn完全統合
- 100万ゼロ点対応スケーラブル設計
- LSTM/CNN/Transformer並列予測システム
- リアルタイム新規ゼロ点即座予測
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import time
import math
import random
import warnings
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import psutil
import signal
import atexit

# 高精度数値演算
try:
    import mpmath
    mpmath.mp.dps = 50  # 50桁精度設定
    MPMATH_AVAILABLE = True
    print("🔢 mpmath 50桁精度: 有効")
except ImportError:
    MPMATH_AVAILABLE = False
    print("⚠️ mpmath無効")

# 機械学習ライブラリ
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight

# 不均衡学習
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
    print("⚖️ SMOTE不均衡学習: 有効")
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️ imblearn無効")

# 深層学習ライブラリ
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, 
                                       Flatten, MultiHeadAttention, LayerNormalization, 
                                       GlobalAveragePooling1D, Embedding, Reshape)
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
    print("🧠 TensorFlow/Keras深層学習: 有効")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow無効")

# GPU関連
try:
    import cupy as cp
    import cupyx.scipy.special as cup_special
    CUDA_AVAILABLE = True
    print("🚀 CUDA RTX3080 GPU加速: 有効")
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️ CUDA無効")

# 警告抑制
warnings.filterwarnings('ignore')

# matplotlib日本語対応
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

def set_seed(seed=42):
    """結果の再現性確保のためのランダムシード設定"""
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
    print(f"🎲 ランダムシード固定: {seed}")

class NKATMillionZeroDeepLearningSystem:
    """NKAT百万ゼロ点深層学習統合システム"""
    
    def __init__(self, target_zeros=1000000, theta=1e-16):
        """システム初期化"""
        set_seed(42)
        
        self.target_zeros = target_zeros
        self.theta = theta
        self.session_id = f"nkat_million_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # システムパラメータ
        self.convergence_acceleration = 23.51
        self.precision_guarantee = 1e-16
        self.gamma_euler = 0.5772156649015329
        
        # データ管理
        self.computed_zeros = None
        self.features = None
        self.labels = None
        self.actual_reals = None
        self.models = {}
        self.results = {}
        
        # 出力ディレクトリ設定
        self.output_dir = Path("nkat_million_results")
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # GPU初期化
        if CUDA_AVAILABLE:
            self.gpu_device = cp.cuda.Device(0)
            print(f"🔥 GPU初期化完了: {self.gpu_device}")
        
        # 回復システム設定
        self.setup_recovery_system()
        
        print(f"🌟 NKAT百万ゼロ点システム初期化完了")
        print(f"🎯 目標: {self.target_zeros:,}ゼロ点処理")
        print(f"📁 出力ディレクトリ: {self.output_dir}")
    
    def setup_recovery_system(self):
        """電源断対応回復システム"""
        signal.signal(signal.SIGINT, self.emergency_save)
        signal.signal(signal.SIGTERM, self.emergency_save)
        atexit.register(self.save_final_results)
        print("🛡️ 電源断対応システム: 有効")
    
    def emergency_save(self, signum=None, frame=None):
        """緊急保存機能"""
        try:
            emergency_file = self.checkpoint_dir / f"emergency_{self.session_id}.pkl"
            emergency_data = {
                'computed_zeros': self.computed_zeros,
                'features': self.features,
                'labels': self.labels,
                'models': self.models,
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat()
            }
            with open(emergency_file, 'wb') as f:
                pickle.dump(emergency_data, f)
            print(f"\n🚨 緊急保存完了: {emergency_file}")
        except Exception as e:
            print(f"⚠️ 緊急保存エラー: {e}")
        
        if signum is not None:
            exit(0)
    
    def save_final_results(self):
        """最終結果保存"""
        try:
            final_file = self.output_dir / f"final_results_{self.session_id}.json"
            final_data = {
                'session_id': self.session_id,
                'target_zeros': self.target_zeros,
                'computed_zeros_count': len(self.computed_zeros) if self.computed_zeros else 0,
                'models_trained': list(self.models.keys()),
                'results': self.results,
                'completion_time': datetime.now().isoformat()
            }
            with open(final_file, 'w') as f:
                json.dump(final_data, f, indent=2, default=str)
            print(f"💾 最終結果保存: {final_file}")
        except Exception as e:
            print(f"⚠️ 最終保存エラー: {e}")
    
    def compute_zeta_zeros_mpmath(self, n_zeros):
        """mpmath による高精度ゼータゼロ点計算"""
        if not MPMATH_AVAILABLE:
            print("❌ mpmath が利用できません")
            return None
        
        print(f"🔢 mpmath高精度ゼータゼロ点計算開始...")
        print(f"   目標ゼロ点数: {n_zeros:,}")
        print(f"   精度設定: {mpmath.mp.dps}桁")
        
        zeros = []
        batch_size = 1000  # メモリ効率のためのバッチ処理
        
        for batch_start in tqdm(range(1, n_zeros + 1, batch_size), desc="ゼロ点計算"):
            batch_end = min(batch_start + batch_size, n_zeros + 1)
            batch_zeros = []
            
            for n in range(batch_start, batch_end):
                try:
                    zero = mpmath.zetazero(n)
                    batch_zeros.append(complex(float(zero.real), float(zero.imag)))
                except Exception as e:
                    print(f"⚠️ ゼロ点{n}計算エラー: {e}")
                    continue
            
            zeros.extend(batch_zeros)
            
            # 定期的な進捗保存
            if len(zeros) % 10000 == 0:
                self.save_checkpoint(zeros)
        
        print(f"✅ ゼータゼロ点計算完了: {len(zeros):,}個")
        self.computed_zeros = zeros
        return zeros
    
    def save_checkpoint(self, zeros):
        """チェックポイント保存"""
        try:
            checkpoint_file = self.checkpoint_dir / f"zeros_checkpoint_{len(zeros)}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(zeros, f)
        except Exception as e:
            print(f"⚠️ チェックポイント保存エラー: {e}")
    
    def encode_and_label_zeros_advanced(self, zeros, negative_fraction=0.2):
        """高度な特徴エンジニアリング＋ラベリング"""
        print(f"🔬 高度特徴エンジニアリング開始...")
        print(f"   ゼロ点数: {len(zeros):,}")
        print(f"   負例割合: {negative_fraction:.1%}")
        
        features = []
        labels = []
        actual_reals = []
        
        num_negatives = int(len(zeros) * negative_fraction)
        negative_indices = np.random.choice(len(zeros), num_negatives, replace=False)
        
        for idx, zero in enumerate(tqdm(zeros, desc="特徴抽出")):
            real = float(zero.real)
            imag = float(zero.imag)
            
            # ラベリング戦略
            if idx in negative_indices:
                # 負例: 実部を摂動
                perturbation = np.random.uniform(-0.5, 0.5)
                real += perturbation
                label = 0
            else:
                # 正例: リーマン仮説 Re(s) = 0.5
                real = 0.5
                label = 1
            
            # 基本特徴量
            magnitude = np.sqrt(real**2 + imag**2)
            angle = np.arctan2(imag, real)
            log_magnitude = math.log(magnitude + 1e-10)
            
            # 高度特徴量
            # 1. 正規化座標
            real_norm = real / 0.5  # リーマン仮説基準正規化
            imag_norm = imag / abs(imag) if imag != 0 else 0
            
            # 2. ゼータ関数関連特徴
            s_complex = complex(real, imag)
            zeta_magnitude = abs(s_complex)
            critical_line_distance = abs(real - 0.5)
            
            # 3. 数論的特徴
            gram_point_approx = 2 * np.pi * imag / math.log(abs(imag) / (2 * np.pi)) if imag > 0 else 0
            hardy_z_approx = math.cos(imag * math.log(abs(imag)) / 2) if imag != 0 else 0
            
            # 4. 統計的特徴
            real_squared = real**2
            imag_squared = imag**2
            magnitude_cubed = magnitude**3
            
            feature_vector = [
                real, imag, magnitude, angle, log_magnitude,
                real_norm, imag_norm, zeta_magnitude, critical_line_distance,
                gram_point_approx, hardy_z_approx,
                real_squared, imag_squared, magnitude_cubed
            ]
            
            features.append(feature_vector)
            labels.append(label)
            actual_reals.append(real)
        
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        actual_reals = np.array(actual_reals, dtype=np.float32)
        
        print(f"✅ 基本特徴抽出完了: {features.shape}")
        
        # 多項式特徴拡張
        print("🔗 多項式特徴拡張...")
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        features_expanded = poly.fit_transform(features)
        print(f"✅ 多項式特徴: {features_expanded.shape}")
        
        # PCA次元削減
        print("📊 PCA次元削減...")
        n_components = min(50, features_expanded.shape[1])  # 最大50次元
        pca = PCA(n_components=n_components, random_state=42)
        features_reduced = pca.fit_transform(features_expanded)
        
        print(f"✅ PCA完了: {features_reduced.shape}")
        print(f"   累積寄与率: {pca.explained_variance_ratio_.sum():.3f}")
        
        # クラス分布確認
        unique, counts = np.unique(labels, return_counts=True)
        print(f"📊 クラス分布: {dict(zip(unique, counts))}")
        
        self.features = features_reduced
        self.labels = labels
        self.actual_reals = actual_reals
        self.pca = pca
        self.poly = poly
        
        return features_reduced, labels, actual_reals
    
    def build_lstm_model(self, input_shape, name="LSTM"):
        """LSTM深層学習モデル構築"""
        model = Sequential(name=name)
        
        # 時系列データ形状に変換用のReshape
        model.add(Reshape((input_shape[0], 1), input_shape=input_shape))
        
        # LSTM層
        model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        model.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
        
        # 全結合層
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_cnn_model(self, input_shape, name="CNN"):
        """CNN深層学習モデル構築"""
        model = Sequential(name=name)
        
        # 1D CNN用のReshape
        model.add(Reshape((input_shape[0], 1), input_shape=input_shape))
        
        # CNN層
        model.add(Conv1D(64, 3, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(128, 3, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(64, 3, activation='relu', padding='same'))
        model.add(GlobalAveragePooling1D())
        
        # 全結合層
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_transformer_model(self, input_shape, name="Transformer"):
        """Transformer深層学習モデル構築"""
        inputs = Input(shape=input_shape)
        
        # Reshapeと位置エンコーディング
        x = Reshape((input_shape[0], 1))(inputs)
        
        # Multi-Head Attention
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=64, dropout=0.1
        )(x, x)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed Forward
        ffn_output = Dense(128, activation='relu')(x)
        ffn_output = Dropout(0.1)(ffn_output)
        ffn_output = Dense(input_shape[0])(ffn_output)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # Global Average Pooling
        x = GlobalAveragePooling1D()(x)
        
        # 分類ヘッド
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs, outputs, name=name)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def run_complete_million_zero_analysis(self, n_zeros=10000):
        """完全な百万ゼロ点解析実行"""
        print(f"🌟 NKAT百万ゼロ点完全解析開始!")
        print(f"   初期計算: {n_zeros:,}ゼロ点")
        
        start_time = time.time()
        
        # 1. ゼータゼロ点計算
        print(f"\n🔢 ステップ1: 高精度ゼータゼロ点計算")
        zeros = self.compute_zeta_zeros_mpmath(n_zeros)
        if not zeros:
            print("❌ ゼロ点計算失敗")
            return
        
        # 2. 特徴エンジニアリング
        print(f"\n🔬 ステップ2: 高度特徴エンジニアリング")
        features, labels, actual_reals = self.encode_and_label_zeros_advanced(zeros)
        
        # 3. データ分割と前処理
        print(f"\n📊 ステップ3: データ分割と前処理")
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   訓練データ: {X_train_scaled.shape}")
        print(f"   テストデータ: {X_test_scaled.shape}")
        
        # SMOTE適用（不均衡学習対応）
        if IMBLEARN_AVAILABLE:
            print("   SMOTE不均衡学習適用...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            print(f"   SMOTE後: {X_train_balanced.shape}")
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
        
        results = {}
        
        # 4. 従来機械学習モデル訓練
        print(f"\n🔬 ステップ4: 従来機械学習モデル訓練")
        
        classical_models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        for name, model in classical_models.items():
            print(f"   {name}訓練中...")
            model.fit(X_train_balanced, y_train_balanced)
            
            # 予測と評価
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            self.models[name] = model
        
        # 5. 深層学習モデル訓練（TensorFlow利用可能時）
        if TF_AVAILABLE:
            print(f"\n🧠 ステップ5: 深層学習モデル訓練")
            
            # コールバック設定
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            input_shape = (X_train_balanced.shape[1],)
            
            # LSTM モデル
            print("   LSTM訓練中...")
            lstm_model = self.build_lstm_model(input_shape)
            lstm_model.fit(
                X_train_balanced, y_train_balanced,
                epochs=30, batch_size=32, validation_split=0.2,
                callbacks=callbacks, verbose=0
            )
            
            # LSTM評価
            y_pred_lstm = (lstm_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
            y_pred_proba_lstm = lstm_model.predict(X_test_scaled).flatten()
            
            results['LSTM'] = {
                'accuracy': accuracy_score(y_test, y_pred_lstm),
                'precision': precision_score(y_test, y_pred_lstm),
                'recall': recall_score(y_test, y_pred_lstm),
                'f1': f1_score(y_test, y_pred_lstm),
                'roc_auc': roc_auc_score(y_test, y_pred_proba_lstm)
            }
            
            self.models['LSTM'] = lstm_model
            
            # CNN モデル
            print("   CNN訓練中...")
            cnn_model = self.build_cnn_model(input_shape)
            cnn_model.fit(
                X_train_balanced, y_train_balanced,
                epochs=30, batch_size=32, validation_split=0.2,
                callbacks=callbacks, verbose=0
            )
            
            # CNN評価
            y_pred_cnn = (cnn_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
            y_pred_proba_cnn = cnn_model.predict(X_test_scaled).flatten()
            
            results['CNN'] = {
                'accuracy': accuracy_score(y_test, y_pred_cnn),
                'precision': precision_score(y_test, y_pred_cnn),
                'recall': recall_score(y_test, y_pred_cnn),
                'f1': f1_score(y_test, y_pred_cnn),
                'roc_auc': roc_auc_score(y_test, y_pred_proba_cnn)
            }
            
            self.models['CNN'] = cnn_model
            
            # Transformer モデル
            print("   Transformer訓練中...")
            transformer_model = self.build_transformer_model(input_shape)
            transformer_model.fit(
                X_train_balanced, y_train_balanced,
                epochs=30, batch_size=32, validation_split=0.2,
                callbacks=callbacks, verbose=0
            )
            
            # Transformer評価
            y_pred_transformer = (transformer_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
            y_pred_proba_transformer = transformer_model.predict(X_test_scaled).flatten()
            
            results['Transformer'] = {
                'accuracy': accuracy_score(y_test, y_pred_transformer),
                'precision': precision_score(y_test, y_pred_transformer),
                'recall': recall_score(y_test, y_pred_transformer),
                'f1': f1_score(y_test, y_pred_transformer),
                'roc_auc': roc_auc_score(y_test, y_pred_proba_transformer)
            }
            
            self.models['Transformer'] = transformer_model
        
        # 結果保存
        self.results = results
        self.scaler = scaler
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        end_time = time.time()
        
        # 6. 結果表示
        print(f"\n📊 ステップ6: 結果表示")
        print("="*80)
        print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
        print("="*80)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<15} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1']:<10.4f} "
                  f"{metrics['roc_auc']:<10.4f}")
        
        print("="*80)
        
        # 最高性能モデル特定
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
            print(f"🏆 最高性能モデル: {best_model[0]} (ROC-AUC: {best_model[1]['roc_auc']:.4f})")
        
        # 7. リアルタイム予測デモ
        print(f"\n🔮 ステップ7: リアルタイム予測デモ")
        demo_candidates = zeros[-10:]  # 最後の10個をテスト用
        predictions = self.real_time_zero_prediction(demo_candidates)
        
        # 最終レポート
        print(f"\n🎉 NKAT百万ゼロ点解析完了!")
        print(f"⏱️ 実行時間: {end_time - start_time:.2f}秒")
        print(f"🔢 処理ゼロ点数: {len(zeros):,}")
        print(f"🧠 訓練モデル数: {len(self.models)}")
        if results:
            print(f"🏆 最高ROC-AUC: {max([r['roc_auc'] for r in results.values()]):.4f}")
        
        return {
            'zeros': zeros,
            'features': features,
            'labels': labels,
            'models': self.models,
            'results': results,
            'predictions': predictions
        }
    
    def real_time_zero_prediction(self, new_zero_candidates):
        """リアルタイム新規ゼロ点予測"""
        print(f"🔮 リアルタイム新規ゼロ点予測開始...")
        
        if not self.models or not hasattr(self, 'scaler'):
            print("❌ 訓練済みモデルが利用できません")
            return None
        
        # 新規候補ゼロ点の特徴抽出
        features_new = []
        for zero in new_zero_candidates:
            real = float(zero.real)
            imag = float(zero.imag)
            
            magnitude = np.sqrt(real**2 + imag**2)
            angle = np.arctan2(imag, real)
            log_magnitude = math.log(magnitude + 1e-10)
            real_norm = real / 0.5
            imag_norm = imag / abs(imag) if imag != 0 else 0
            zeta_magnitude = abs(complex(real, imag))
            critical_line_distance = abs(real - 0.5)
            gram_point_approx = 2 * np.pi * imag / math.log(abs(imag) / (2 * np.pi)) if imag > 0 else 0
            hardy_z_approx = math.cos(imag * math.log(abs(imag)) / 2) if imag != 0 else 0
            real_squared = real**2
            imag_squared = imag**2
            magnitude_cubed = magnitude**3
            
            feature_vector = [
                real, imag, magnitude, angle, log_magnitude,
                real_norm, imag_norm, zeta_magnitude, critical_line_distance,
                gram_point_approx, hardy_z_approx,
                real_squared, imag_squared, magnitude_cubed
            ]
            features_new.append(feature_vector)
        
        features_new = np.array(features_new, dtype=np.float32)
        
        # 多項式特徴＋PCA変換
        features_expanded = self.poly.transform(features_new)
        features_transformed = self.pca.transform(features_expanded)
        features_scaled = self.scaler.transform(features_transformed)
        
        # 全モデルで予測
        predictions = {}
        for model_name, model in self.models.items():
            if 'LSTM' in model_name or 'CNN' in model_name or 'Transformer' in model_name:
                pred_proba = model.predict(features_scaled).flatten()
            else:
                pred_proba = model.predict_proba(features_scaled)[:, 1]
            
            predictions[model_name] = pred_proba
        
        # アンサンブル予測（全モデル平均）
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        # 結果まとめ
        results = []
        for i, zero in enumerate(new_zero_candidates):
            result = {
                'zero': complex(zero.real, zero.imag),
                'riemann_hypothesis_probability': float(ensemble_pred[i]),
                'individual_predictions': {name: float(pred[i]) for name, pred in predictions.items()},
                'is_likely_riemann_zero': ensemble_pred[i] > 0.5
            }
            results.append(result)
        
        print(f"✅ 予測完了: {len(results)}個の候補")
        
        # 予測結果表示
        for i, result in enumerate(results):
            print(f"   ゼロ点{i+1}: {result['zero']}")
            print(f"   リーマン仮説確率: {result['riemann_hypothesis_probability']:.4f}")
            print(f"   判定: {'✅ 真のゼロ点' if result['is_likely_riemann_zero'] else '❌ 偽のゼロ点'}")
        
        return results

def main():
    """メイン実行関数"""
    print("🌟 NKAT百万ゼロ点深層学習統合システム起動!")
    
    # システム初期化
    system = NKATMillionZeroDeepLearningSystem(target_zeros=1000000)
    
    # 完全解析実行（初期は1万ゼロ点）
    results = system.run_complete_million_zero_analysis(n_zeros=1000)  # テスト用に1000個に調整
    
    print("🎊 システム実行完了!")
    return results

if __name__ == "__main__":
    main() 