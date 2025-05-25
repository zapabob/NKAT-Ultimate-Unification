import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import math
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# GPUの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class NonCommutativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, theta: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.theta = theta
        
        self.W_q = nn.Parameter(torch.randn(d_model, d_model) * np.sqrt(1.0 / d_model))
        self.W_k = nn.Parameter(torch.randn(d_model, d_model) * np.sqrt(1.0 / d_model))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model) * np.sqrt(1.0 / d_model))
        self.W_o = nn.Parameter(torch.randn(d_model, d_model) * np.sqrt(1.0 / d_model))
        
        self.alpha = nn.Parameter(torch.ones(1))
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        return x.transpose(1, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 非可換クエリ、キー、バリューの生成
        Q = torch.matmul(x, self.W_q)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)
        
        # 非可換性を考慮した追加項
        Q_nc = Q + self.theta * torch.sin(Q)
        K_nc = K + self.theta * torch.cos(K)
        V_nc = V + self.alpha * self.theta * torch.tanh(V)
        
        # マルチヘッドに分割
        Q = self.split_heads(Q_nc)
        K = self.split_heads(K_nc)
        V = self.split_heads(V_nc)
        
        # スケーリングドット積アテンション
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        
        # 非可換アテンション出力
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return torch.matmul(out, self.W_o)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class NonCommutativeTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, theta: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.self_attn = NonCommutativeMultiHeadAttention(d_model, n_heads, theta)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 非可換フィードフォワードネットワーク
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class NKATTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 d_ff: int,
                 num_classes: int,
                 theta: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            NonCommutativeTransformerLayer(d_model, n_heads, d_ff, theta, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 入力の形状を変更 (batch_size, channels, height, width) -> (batch_size, features)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the image
        
        # 入力の射影
        x = self.input_projection(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Position encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
            
        # 出力の射影
        x = x.squeeze(1)  # Remove sequence dimension
        return self.output_projection(x)

def calculate_model_metrics(model: NKATTransformer, data_loader: DataLoader, device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    return accuracy, precision, recall, f1

def evaluate_model(model: NKATTransformer, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy

def train_nkat_transformer(
    model: NKATTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 200,
    learning_rate: float = 0.0003,
    patience: int = 10,
    min_delta: float = 0.001
) -> Tuple[List[float], List[float]]:
    # モデルをGPUに移動
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    criterion = nn.CrossEntropyLoss()
    
    # 早期停止のための変数
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    # エポックのプログレスバー
    pbar_epoch = tqdm(range(epochs), desc="Training", position=0)
    
    for epoch in pbar_epoch:
        # 学習フェーズ
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # バッチのプログレスバー
        pbar_batch = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                         leave=False, position=1)
        
        for batch_data, batch_labels in pbar_batch:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
            
            pbar_batch.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        
        # 検証フェーズ
        val_loss, val_accuracy = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        
        # モデルメトリクスの計算
        train_metrics = calculate_model_metrics(model, train_loader, device)
        val_metrics = calculate_model_metrics(model, val_loader, device)
        
        # エポックごとの進捗を更新
        pbar_epoch.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'train_acc': f'{train_accuracy:.2f}%',
            'val_acc': f'{val_accuracy:.2f}%',
            'train_f1': f'{train_metrics[3]:.4f}',
            'val_f1': f'{val_metrics[3]:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
        # 早期停止のチェック
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                # 最良モデルを復元
                model.load_state_dict(best_model_state)
                break
    
    return train_losses, val_losses

def plot_losses(train_losses: List[float], val_losses: List[float]):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('NKAT-Transformer MNIST Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('nkat_transformer_mnist_loss.png')
    plt.close()

def transfer_learning(
    model: NKATTransformer,
    source_loader: DataLoader,
    target_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.0001,
    fine_tune: bool = True
) -> Tuple[List[float], List[float]]:
    """
    転移学習を実行する関数
    
    Args:
        model: 事前学習済みのNKATモデル
        source_loader: ソースデータセットのDataLoader
        target_loader: ターゲットデータセットのDataLoader
        epochs: 学習エポック数
        learning_rate: 学習率
        fine_tune: 微調整を行うかどうか
    
    Returns:
        学習損失と検証損失の履歴
    """
    model = model.to(device)
    
    # 微調整の場合は全パラメータを更新、そうでない場合は分類層のみ更新
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.output_projection.parameters():
            param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(target_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    
    pbar_epoch = tqdm(range(epochs), desc="Transfer Learning", position=0)
    
    for epoch in pbar_epoch:
        # 学習フェーズ
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar_batch = tqdm(target_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                         leave=False, position=1)
        
        for batch_data, batch_labels in pbar_batch:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
            
            pbar_batch.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(target_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        
        # 検証フェーズ
        val_loss, val_accuracy = evaluate_model(model, source_loader, device)
        val_losses.append(val_loss)
        
        # モデルメトリクスの計算
        train_metrics = calculate_model_metrics(model, target_loader, device)
        val_metrics = calculate_model_metrics(model, source_loader, device)
        
        pbar_epoch.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'train_acc': f'{train_accuracy:.2f}%',
            'val_acc': f'{val_accuracy:.2f}%',
            'train_f1': f'{train_metrics[3]:.4f}',
            'val_f1': f'{val_metrics[3]:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    return train_losses, val_losses

def main():
    # データの前処理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # MNISTデータセットの読み込み（ソース）
    mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_train_size = int(0.8 * len(mnist_dataset))
    mnist_val_size = len(mnist_dataset) - mnist_train_size
    mnist_train_dataset, mnist_val_dataset = random_split(mnist_dataset, 
                                                         [mnist_train_size, mnist_val_size])
    
    # Fashion-MNISTデータセットの読み込み（ターゲット）
    fashion_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    fashion_dataset = datasets.FashionMNIST('./data', train=True, download=True, 
                                          transform=fashion_transform)
    fashion_train_size = int(0.8 * len(fashion_dataset))
    fashion_val_size = len(fashion_dataset) - fashion_train_size
    fashion_train_dataset, fashion_val_dataset = random_split(fashion_dataset, 
                                                            [fashion_train_size, fashion_val_size])
    
    # DataLoaderの作成
    mnist_train_loader = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    mnist_val_loader = DataLoader(mnist_val_dataset, batch_size=64, shuffle=False, pin_memory=True)
    fashion_train_loader = DataLoader(fashion_train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    fashion_val_loader = DataLoader(fashion_val_dataset, batch_size=64, shuffle=False, pin_memory=True)
    
    # モデルのハイパーパラメータ
    input_dim = 784  # 28x28
    d_model = 256
    n_heads = 8
    n_layers = 4
    d_ff = 1024
    num_classes = 10
    theta = 0.1
    dropout = 0.1
    
    # モデルの作成
    model = NKATTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        num_classes=num_classes,
        theta=theta,
        dropout=dropout
    )
    
    # MNISTでの事前学習
    print("Pre-training on MNIST...")
    train_losses, val_losses = train_nkat_transformer(
        model,
        mnist_train_loader,
        mnist_val_loader,
        epochs=200,
        learning_rate=0.0003
    )
    
    # Fashion-MNISTへの転移学習
    print("\nTransfer learning to Fashion-MNIST...")
    transfer_losses, transfer_val_losses = transfer_learning(
        model,
        mnist_val_loader,  # ソースデータセットの検証セット
        fashion_train_loader,  # ターゲットデータセットの学習セット
        epochs=100,
        learning_rate=0.0001,
        fine_tune=True
    )
    
    # 最終的な評価
    print("\nFinal Results:")
    print("MNIST (Source):")
    final_mnist_loss, final_mnist_acc = evaluate_model(model, mnist_val_loader, device)
    final_mnist_metrics = calculate_model_metrics(model, mnist_val_loader, device)
    print(f"Loss: {final_mnist_loss:.4f}")
    print(f"Accuracy: {final_mnist_acc:.2f}%")
    print(f"Precision: {final_mnist_metrics[1]:.4f}")
    print(f"Recall: {final_mnist_metrics[2]:.4f}")
    print(f"F1 Score: {final_mnist_metrics[3]:.4f}")
    
    print("\nFashion-MNIST (Target):")
    final_fashion_loss, final_fashion_acc = evaluate_model(model, fashion_val_loader, device)
    final_fashion_metrics = calculate_model_metrics(model, fashion_val_loader, device)
    print(f"Loss: {final_fashion_loss:.4f}")
    print(f"Accuracy: {final_fashion_acc:.2f}%")
    print(f"Precision: {final_fashion_metrics[1]:.4f}")
    print(f"Recall: {final_fashion_metrics[2]:.4f}")
    print(f"F1 Score: {final_fashion_metrics[3]:.4f}")
    
    # 損失曲線のプロット
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='MNIST Training')
    plt.plot(val_losses, label='MNIST Validation')
    plt.title('MNIST Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(transfer_losses, label='Fashion-MNIST Training')
    plt.plot(transfer_val_losses, label='Fashion-MNIST Validation')
    plt.title('Fashion-MNIST Transfer Learning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('nkat_transfer_learning_loss.png')
    plt.close()

if __name__ == "__main__":
    main() 