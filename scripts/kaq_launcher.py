
# KAQ統合理論 Google Colab起動コード
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm.notebook import tqdm

# 設定読み込み
with open("kaq_colab_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

print("🌌 KAQ統合理論 Google Colab版")
print("=" * 50)
print(f"📊 K-A次元: {config['kaq_settings']['ka_dimension']}")
print(f"⚛️ 量子ビット: {config['kaq_settings']['qft_qubits']}")
print(f"🔧 非可換パラメータ: {config['kaq_settings']['theta']:.2e}")

# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ デバイス: {device}")

# 日本語フォント設定
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("✅ 初期化完了 - 実験を開始できます！")
