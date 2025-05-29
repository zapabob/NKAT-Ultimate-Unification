import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class NonCommutativeLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, theta: float = 0.1):
        super().__init__()
        self.theta = theta
        # 重みの初期化をさらに改善
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim))
        self.V = nn.Parameter(torch.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))
        self.alpha = nn.Parameter(torch.ones(1))  # スケーリングパラメータ
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 非可換積の実装をさらに改善
        x_hat = torch.matmul(x, self.W) + self.alpha * torch.matmul(x, self.V) * self.theta
        x_hat = x_hat + self.b
        # 非可換活性化関数をさらに改善
        return torch.tanh(x_hat) + self.theta * torch.sin(x_hat) + self.theta * torch.cos(x_hat) + self.theta * torch.tanh(x_hat)

class NKATNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 入力層から最初の隠れ層
        self.layers.append(NonCommutativeLayer(input_dim, hidden_dims[0]))
        self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.layers.append(nn.Dropout(0.3))  # ドロップアウト率を増加
        
        # 隠れ層
        for i in range(len(hidden_dims)-1):
            self.layers.append(NonCommutativeLayer(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.layers.append(nn.Dropout(0.3))
            
        # 出力層
        self.layers.append(NonCommutativeLayer(hidden_dims[-1], output_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class NKATLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 非可換性を考慮した損失関数をさらに改善
        mse_loss = torch.mean((pred - target)**2)
        l1_loss = torch.mean(torch.abs(pred - target))
        cosine_loss = 1 - torch.mean(torch.nn.functional.cosine_similarity(pred, target))
        return mse_loss + 0.03 * l1_loss + 0.02 * cosine_loss

class CustomDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_nkat_network(
    model: NKATNetwork,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    epochs: int = 100,
    learning_rate: float = 0.0005,  # 学習率をさらに調整
    batch_size: int = 64  # バッチサイズを増加
) -> List[float]:
    # データの正規化を改善
    data_mean = train_data.mean(dim=0, keepdim=True)
    data_std = train_data.std(dim=0, keepdim=True)
    train_data = (train_data - data_mean) / (data_std + 1e-8)
    
    # データローダーの設定
    dataset = CustomDataset(train_data, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # オプティマイザとスケジューラーの設定を改善
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.3
    )
    criterion = NKATLoss()
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_data, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 勾配クリッピング
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return losses

def plot_losses(losses: List[float]):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('nkat_training_loss.png')
    plt.close()

def main():
    # データの生成
    input_dim = 10
    hidden_dims = [64, 128, 64]  # より広いネットワーク
    output_dim = 5
    
    # モデルの作成
    model = NKATNetwork(input_dim, hidden_dims, output_dim)
    
    # サンプルデータの生成（より多くのデータ）
    n_samples = 2000  # データ量をさらに増加
    train_data = torch.randn(n_samples, input_dim)
    train_labels = torch.randn(n_samples, output_dim)
    
    # 学習
    losses = train_nkat_network(
        model, 
        train_data, 
        train_labels,
        epochs=200,
        learning_rate=0.0005,
        batch_size=64
    )
    
    print(f"Final loss: {losses[-1]:.4f}")
    plot_losses(losses)

if __name__ == "__main__":
    main() 