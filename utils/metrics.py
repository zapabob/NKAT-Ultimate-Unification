import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

def tpe_metric(val_acc: float, lambda_theory: float, complexity_penalty: float = 0.) -> float:
    """
    🎯 TPE (Theory-Practical Equilibrium) 指標計算
    
    TPE = ValAcc / log10(1 + λ_theory + penalty)
    
    Args:
        val_acc: 検証精度 (0.0 ~ 1.0)
        lambda_theory: NKAT理論専用パラメータ数
        complexity_penalty: 追加の複雑度ペナルティ
    
    Returns:
        TPE スコア（高いほど良い理論と実践のバランス）
    """
    if val_acc <= 0:
        return 0.0
    
    denominator = math.log10(1.0 + lambda_theory + complexity_penalty)
    if denominator <= 0:
        return val_acc  # ペナルティなしの場合
    
    return val_acc / denominator

def advanced_tpe_metric(val_acc: float, lambda_theory: float, 
                       total_params: float, inference_time: float = 0.,
                       memory_usage: float = 0., 
                       generalization_gap: float = 0.) -> Dict[str, float]:
    """
    🚀 高度なTPE指標計算（複数要素考慮）
    
    Args:
        val_acc: 検証精度
        lambda_theory: NKAT理論パラメータ数
        total_params: 総パラメータ数
        inference_time: 推論時間（ミリ秒）
        memory_usage: メモリ使用量（MB）
        generalization_gap: 訓練精度 - 検証精度
    
    Returns:
        詳細なTPE分析結果
    """
    # 基本TPE
    basic_tpe = tpe_metric(val_acc, lambda_theory)
    
    # 効率性考慮TPE
    efficiency_penalty = 0.
    if inference_time > 0:
        efficiency_penalty += math.log10(1 + inference_time / 100)  # 100ms基準
    if memory_usage > 0:
        efficiency_penalty += math.log10(1 + memory_usage / 1000)   # 1GB基準
    
    efficiency_tpe = val_acc / math.log10(1.0 + lambda_theory + efficiency_penalty)
    
    # 汎化性考慮TPE
    generalization_penalty = max(0, generalization_gap * 10)  # 過学習ペナルティ
    generalization_tpe = val_acc / math.log10(1.0 + lambda_theory + generalization_penalty)
    
    # 理論密度（Theory Density）
    theory_density = lambda_theory / max(total_params, 1)
    
    # 総合TPEスコア
    comprehensive_tpe = (basic_tpe + efficiency_tpe + generalization_tpe) / 3.0
    
    return {
        'basic_tpe': basic_tpe,
        'efficiency_tpe': efficiency_tpe,
        'generalization_tpe': generalization_tpe,
        'comprehensive_tpe': comprehensive_tpe,
        'theory_density': theory_density,
        'lambda_theory': lambda_theory,
        'total_params': total_params
    }

def count_nkat_parameters(model: nn.Module) -> Dict[str, int]:
    """
    🔍 NKAT理論関連パラメータのカウント
    
    Args:
        model: PyTorchモデル
    
    Returns:
        パラメータ分析結果
    """
    nkat_params = 0
    attention_params = 0
    total_params = 0
    param_breakdown = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # NKAT関連パラメータの識別
        if any(keyword in name.lower() for keyword in ['nkat', 'alpha', 'beta', 'theta']):
            nkat_params += param_count
            param_breakdown[f'nkat_{name}'] = param_count
        
        # アテンション関連パラメータ
        if any(keyword in name.lower() for keyword in ['attn', 'attention', 'qkv', 'pos_bias']):
            attention_params += param_count
            param_breakdown[f'attention_{name}'] = param_count
    
    return {
        'nkat_params': nkat_params,
        'attention_params': attention_params,
        'total_params': total_params,
        'nkat_ratio': nkat_params / max(total_params, 1),
        'attention_ratio': attention_params / max(total_params, 1),
        'breakdown': param_breakdown
    }

def calculate_attention_efficiency(model: nn.Module, 
                                  input_tensor: torch.Tensor) -> Dict[str, float]:
    """
    📊 アテンション効率性の計算
    
    Args:
        model: NKATモデル
        input_tensor: 入力テンソル
    
    Returns:
        アテンション効率性指標
    """
    model.eval()
    attention_entropies = []
    
    def entropy_hook(module, input, output):
        if hasattr(module, 'get_attention_entropy'):
            entropy = module.get_attention_entropy()
            attention_entropies.append(entropy.item())
    
    # フック登録
    hooks = []
    for module in model.modules():
        if hasattr(module, 'get_attention_entropy'):
            hooks.append(module.register_forward_hook(entropy_hook))
    
    with torch.no_grad():
        _ = model(input_tensor)
    
    # フック削除
    for hook in hooks:
        hook.remove()
    
    if not attention_entropies:
        return {'mean_entropy': 0., 'entropy_std': 0., 'entropy_trend': 0.}
    
    mean_entropy = np.mean(attention_entropies)
    entropy_std = np.std(attention_entropies)
    
    # エントロピーの層別トレンド（単調性）
    entropy_trend = 0.
    if len(attention_entropies) > 1:
        trend_scores = []
        for i in range(1, len(attention_entropies)):
            trend_scores.append(attention_entropies[i] - attention_entropies[i-1])
        entropy_trend = np.mean(trend_scores)
    
    return {
        'mean_entropy': mean_entropy,
        'entropy_std': entropy_std,
        'entropy_trend': entropy_trend,
        'layer_entropies': attention_entropies
    }

def evaluate_hyperparameter_sensitivity(param_dict: Dict[str, float], 
                                       tpe_score: float) -> Dict[str, float]:
    """
    🎛️ ハイパーパラメータ感度分析
    
    Args:
        param_dict: ハイパーパラメータ辞書
        tpe_score: 現在のTPEスコア
    
    Returns:
        感度分析結果
    """
    sensitivity_scores = {}
    
    # Temperature感度
    if 'temperature' in param_dict:
        temp = param_dict['temperature']
        # 理想的な温度域からの逸脱度
        ideal_temp_range = (0.8, 1.2)
        if temp < ideal_temp_range[0]:
            temp_penalty = (ideal_temp_range[0] - temp) ** 2
        elif temp > ideal_temp_range[1]:
            temp_penalty = (temp - ideal_temp_range[1]) ** 2
        else:
            temp_penalty = 0.
        sensitivity_scores['temperature_sensitivity'] = temp_penalty
    
    # Top-K/Top-P バランス
    if 'top_k' in param_dict and 'top_p' in param_dict:
        top_k = param_dict.get('top_k', 0)
        top_p = param_dict.get('top_p', 1.0)
        
        # Top-KとTop-Pの相互作用評価
        if top_k > 0 and top_p < 1.0:
            # 両方使用している場合の冗長性ペナルティ
            redundancy_penalty = min(top_k / 20.0, 1.0) * (1.0 - top_p)
        else:
            redundancy_penalty = 0.
        sensitivity_scores['topk_topp_redundancy'] = redundancy_penalty
    
    # NKAT強度の適切性
    if 'nkat_strength' in param_dict:
        nkat_str = param_dict['nkat_strength']
        # 適切なNKAT強度範囲の評価
        optimal_nkat_range = (0.005, 0.025)
        if nkat_str < optimal_nkat_range[0]:
            nkat_penalty = (optimal_nkat_range[0] - nkat_str) ** 2 * 100
        elif nkat_str > optimal_nkat_range[1]:
            nkat_penalty = (nkat_str - optimal_nkat_range[1]) ** 2 * 100
        else:
            nkat_penalty = 0.
        sensitivity_scores['nkat_strength_deviation'] = nkat_penalty
    
    return sensitivity_scores

def comprehensive_model_analysis(model: nn.Module, 
                                val_acc: float,
                                train_acc: float = None,
                                input_tensor: torch.Tensor = None,
                                hyperparams: Dict[str, float] = None) -> Dict[str, any]:
    """
    🔬 モデルの総合分析
    
    Args:
        model: 分析対象モデル
        val_acc: 検証精度
        train_acc: 訓練精度（optional）
        input_tensor: 入力例（optional）
        hyperparams: ハイパーパラメータ（optional）
    
    Returns:
        総合分析結果
    """
    # パラメータ分析
    param_analysis = count_nkat_parameters(model)
    
    # TPE計算
    lambda_theory = param_analysis['nkat_params']
    generalization_gap = (train_acc - val_acc) if train_acc else 0.
    
    tpe_results = advanced_tpe_metric(
        val_acc, lambda_theory, param_analysis['total_params'],
        generalization_gap=generalization_gap
    )
    
    # アテンション効率性（入力が与えられた場合）
    attention_analysis = {}
    if input_tensor is not None:
        attention_analysis = calculate_attention_efficiency(model, input_tensor)
    
    # ハイパーパラメータ感度（ハイパーパラメータが与えられた場合）
    sensitivity_analysis = {}
    if hyperparams:
        sensitivity_analysis = evaluate_hyperparameter_sensitivity(
            hyperparams, tpe_results['comprehensive_tpe']
        )
    
    return {
        'tpe_metrics': tpe_results,
        'parameter_analysis': param_analysis,
        'attention_efficiency': attention_analysis,
        'sensitivity_analysis': sensitivity_analysis,
        'model_summary': {
            'val_accuracy': val_acc,
            'train_accuracy': train_acc,
            'generalization_gap': generalization_gap,
            'theory_parameter_ratio': param_analysis['nkat_ratio'],
            'comprehensive_tpe': tpe_results['comprehensive_tpe']
        }
    } 