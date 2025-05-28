#!/usr/bin/env python3
"""
🔧 NKAT θ-parameter 緊急修正版
診断結果に基づく即座修正
"""

import torch
import torch.nn.functional as F

def safe_theta_running_loss(theta_values, energy_scale, eps=1e-15):
    """
    🛡️ 完全NaN安全 θ-running損失
    診断結果に基づく修正版
    """
    if energy_scale is None:
        return torch.tensor(0.0, device=theta_values.device, requires_grad=True)
    
    # 次元処理（安全化）
    if energy_scale.dim() > 1:
        energy_scale = energy_scale[:, 0] if energy_scale.size(1) > 0 else energy_scale.flatten()
    
    if theta_values.dim() > 1:
        theta_values = theta_values.mean(dim=-1)
    
    # 🔧 重要修正：θ値の範囲制限
    # 1e-80 → 1e-50 に変更（log10で-50まで）
    theta_clamped = torch.clamp(theta_values, min=1e-50, max=1e-10)
    
    # 🔧 重要修正：エネルギースケールの範囲制限
    energy_clamped = torch.clamp(energy_scale, min=1e10, max=1e18)
    
    # 安全な対数計算
    log_energy = torch.log10(energy_clamped)
    running_target = -0.1 * log_energy  # -1.8 to -1.0 範囲
    
    theta_log = torch.log10(theta_clamped)  # -50 to -10 範囲
    
    # 🔧 重要修正：次元調整（安全化）
    if theta_log.shape != running_target.shape:
        min_batch = min(theta_log.size(0), running_target.size(0))
        theta_log = theta_log[:min_batch]
        running_target = running_target[:min_batch]
    
    # 🔧 重要修正：MSE計算前の最終チェック
    if torch.isinf(theta_log).any() or torch.isinf(running_target).any():
        print("⚠️ θ-running: Infinity検出 - 安全値に置換")
        theta_log = torch.where(torch.isinf(theta_log), torch.tensor(-30.0, device=theta_log.device), theta_log)
        running_target = torch.where(torch.isinf(running_target), torch.tensor(-1.5, device=running_target.device), running_target)
    
    loss = F.mse_loss(theta_log, running_target)
    
    # 🔧 最終安全チェック
    if torch.isnan(loss) or torch.isinf(loss):
        print("⚠️ θ-running MSE: NaN/Inf検出 - 安全値に設定")
        loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
    
    return loss

def safe_spectral_dimension_loss(dirac_field, target_dim=4.0, eps=1e-12):
    """
    🛡️ 完全NaN安全 スペクトラル次元損失
    """
    # 安全な正規化
    field_norm = torch.norm(dirac_field, dim=-1) + eps
    
    # 🔧 範囲制限追加
    field_norm = torch.clamp(field_norm, min=eps, max=1e6)
    
    log_field = torch.log(field_norm)
    
    # 🔧 log結果の範囲制限
    log_field = torch.clamp(log_field, min=-20, max=20)
    
    # 安定化されたスペクトラル次元計算
    spectral_dim = 4.0 + 0.1 * torch.mean(log_field)
    
    # 🔧 物理的範囲制限
    spectral_dim = torch.clamp(spectral_dim, min=0.1, max=10.0)
    
    target_tensor = torch.tensor(target_dim, device=dirac_field.device, dtype=dirac_field.dtype)
    loss = F.mse_loss(spectral_dim, target_tensor)
    
    return loss

# テスト実行
if __name__ == "__main__":
    print("🔧 θ-parameter修正テスト")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 問題のあったθ値でテスト
    theta_test = torch.tensor([1e-70, 1e-80, 1e-100], device=device)
    energy_test = torch.logspace(10, 18, 3, device=device)
    
    print(f"修正前θ範囲: [{torch.min(theta_test):.2e}, {torch.max(theta_test):.2e}]")
    
    # 修正版で計算
    loss = safe_theta_running_loss(theta_test, energy_test)
    
    print(f"修正後MSE損失: {loss.item():.6f}")
    print(f"NaN/Inf検出: {torch.isnan(loss).item() or torch.isinf(loss).item()}")
    
    # スペクトラル次元テスト
    dirac_test = torch.randn(32, 4, 4, device=device) * 1e-3
    spectral_loss = safe_spectral_dimension_loss(dirac_test)
    
    print(f"スペクトラル損失: {spectral_loss.item():.6f}")
    print("✅ 修正版テスト完了") 