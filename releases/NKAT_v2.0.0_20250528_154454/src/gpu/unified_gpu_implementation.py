import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
import matplotlib.pyplot as plt
from nkat_implementation import NKATNetwork, QuantumStateRepresentation
from quantum_gravity_implementation import QuantumGravityNetwork, EntropicGravity

class UnifiedUniverseModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, theta: float = 0.1):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # NKATネットワークの初期化
        self.nkat_network = NKATNetwork(input_dim, hidden_dims, output_dim, theta).to(self.device)
        
        # 量子重力ネットワークの初期化
        self.gravity_network = QuantumGravityNetwork(input_dim, hidden_dims, theta).to(self.device)
        
        # 量子状態表現の初期化
        self.qsr = QuantumStateRepresentation(n_qubits=2)
        
        # エントロピー重力の初期化
        self.entropic_gravity = EntropicGravity(input_dim, theta)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # NKATネットワークの出力
        nkat_output = self.nkat_network(x)
        
        # 量子重力ネットワークの出力
        gravity_output = self.gravity_network(x)
        
        return nkat_output, gravity_output

def train_unified_model(
    model: UnifiedUniverseModel,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    epochs: int = 100,
    learning_rate: float = 0.01,
    batch_size: int = 32
) -> Tuple[List[float], List[float]]:
    """統合モデルの学習"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    nkat_losses = []
    gravity_losses = []
    
    # データをGPUに移動
    train_data = train_data.to(model.device)
    train_labels = train_labels.to(model.device)
    
    n_batches = len(train_data) // batch_size
    
    for epoch in range(epochs):
        epoch_nkat_loss = 0
        epoch_gravity_loss = 0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_data = train_data[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            # フォワードパス
            nkat_output, gravity_output = model(batch_data)
            
            # 損失の計算
            nkat_loss = criterion(nkat_output, batch_labels)
            gravity_loss = criterion(gravity_output, batch_labels)
            
            # バックワードパス
            total_loss = nkat_loss + gravity_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_nkat_loss += nkat_loss.item()
            epoch_gravity_loss += gravity_loss.item()
        
        nkat_losses.append(epoch_nkat_loss / n_batches)
        gravity_losses.append(epoch_gravity_loss / n_batches)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], NKAT Loss: {nkat_losses[-1]:.4f}, Gravity Loss: {gravity_losses[-1]:.4f}')
    
    return nkat_losses, gravity_losses

def plot_results(nkat_losses: List[float], gravity_losses: List[float]):
    """学習結果のプロット"""
    plt.figure(figsize=(10, 6))
    plt.plot(nkat_losses, label='NKAT Loss')
    plt.plot(gravity_losses, label='Gravity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig('training_losses.png')
    plt.close()

def main():
    # GPUの確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # パラメータ設定
    input_dim = 4
    hidden_dims = [8, 16]
    output_dim = 2
    theta = 0.1
    batch_size = 32
    epochs = 100
    
    # モデルの作成
    model = UnifiedUniverseModel(input_dim, hidden_dims, output_dim, theta)
    
    # サンプルデータの生成
    n_samples = 1000
    train_data = torch.randn(n_samples, input_dim).to(device)
    train_labels = torch.randn(n_samples, output_dim).to(device)
    
    # 学習の実行
    nkat_losses, gravity_losses = train_unified_model(
        model, train_data, train_labels, epochs=epochs, batch_size=batch_size
    )
    
    # 結果のプロット
    plot_results(nkat_losses, gravity_losses)
    
    # 量子状態とエントロピー重力の計算
    with torch.no_grad():
        # 量子もつれ状態の生成
        state1 = torch.tensor([1.0, 0.0]).to(device)
        state2 = torch.tensor([0.0, 1.0]).to(device)
        entangled_state = model.qsr.create_entangled_state(state1, state2)
        
        # エントロピー重力の計算
        state = torch.randn(input_dim).to(device)
        entropy = model.entropic_gravity.compute_entropy(state)
        distance = torch.tensor(1.0).to(device)
        force = model.entropic_gravity.compute_force(entropy, distance)
        
        print("\n最終結果:")
        print(f"もつれ状態のノルム: {torch.norm(entangled_state).item():.4f}")
        print(f"エントロピー: {entropy.item():.4f}")
        print(f"エントロピー力: {force.item():.4f}")
        print(f"最終NKAT損失: {nkat_losses[-1]:.4f}")
        print(f"最終重力損失: {gravity_losses[-1]:.4f}")

if __name__ == "__main__":
    main() 