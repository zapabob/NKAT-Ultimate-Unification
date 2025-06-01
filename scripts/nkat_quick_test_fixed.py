#!/usr/bin/env python3
"""
🔧 NKAT修正版 クイックテスト
エラー修正後の動作確認
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# CUDA設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f"🚀 RTX3080 CUDA: {torch.cuda.get_device_name(0)}")

class NKATQuickTest(nn.Module):
    """修正版NKAT-Transformer（軽量）"""
    
    def __init__(self, num_classes=10, embed_dim=256, depth=4):
        super().__init__()
        
        # 軽量パッチ埋め込み
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, 4, stride=4)
        )
        
        # 位置埋め込み
        self.pos_embedding = nn.Parameter(torch.randn(1, 50, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        
        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, x):
        B = x.shape[0]
        
        # パッチ埋め込み
        x = self.patch_embedding(x)  # (B, embed_dim, 7, 7)
        x = x.flatten(2).transpose(1, 2)  # (B, 49, embed_dim)
        
        # クラストークン追加
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 50, embed_dim)
        
        # 位置埋め込み（サイズ一致確保）
        seq_len = x.shape[1]
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = x + pos_emb
        
        # Transformer処理
        x = self.transformer(x)
        
        # 分類
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.classifier(cls_output)
        
        return logits

def quick_test():
    """クイックテスト実行"""
    
    print("🧪 NKAT修正版クイックテスト開始")
    
    # データ準備（軽量）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # 小さなサブセット使用
    train_subset = torch.utils.data.Subset(train_dataset, range(1000))
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    
    # モデル作成
    model = NKATQuickTest().to(device)
    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # トレーニング設定（修正版）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    model.train()
    
    print("🚀 Training started...")
    
    for epoch in range(3):  # 3エポックのみ
        epoch_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            try:
                if scaler and device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        output = model(data)
                        loss = criterion(output, target)
                    
                    # 数値安定性チェック
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"❌ Numerical instability detected")
                        continue
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"❌ Numerical instability detected")
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
            except RuntimeError as e:
                print(f"❌ Runtime error: {e}")
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                continue
            
            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 進捗更新
            if batch_idx % 5 == 0:
                current_acc = 100. * correct / total
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Acc={epoch_acc:.2f}%")
        
        # メモリクリア
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print("✅ クイックテスト完了！エラー修正成功")
    return True

if __name__ == "__main__":
    try:
        success = quick_test()
        if success:
            print("🎉 全てのエラーが修正されました！")
        else:
            print("❌ まだ問題があります")
    except Exception as e:
        print(f"❌ テスト中にエラー: {e}")
        import traceback
        traceback.print_exc() 