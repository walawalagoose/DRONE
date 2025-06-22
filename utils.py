import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Any, Tuple

def is_power_of_two(n: int) -> bool:
    """
    检查一个数是否是2的幂次
    """
    return n > 0 and (n & (n - 1)) == 0


def generate_hadamard_matrix(n: int) -> torch.Tensor:
    """
    生成Hadamard矩阵
    Args:
        n: 矩阵大小，必须是2的幂次
    Returns:
        torch.Tensor: Hadamard矩阵，形状为 [n, n]
    """
    if n == 1:
        return torch.ones(1, 1)

    if not is_power_of_two(n):
        raise ValueError(f"Dimension {n} is not a power of 2")

    # 递归构造Hadamard矩阵
    H = generate_hadamard_matrix(n // 2)

    # 构造更大的Hadamard矩阵
    H_n = torch.cat([
        torch.cat([H, H], dim=1),
        torch.cat([H, -H], dim=1)
    ], dim=0)

    return H_n


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None
    
    
class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)
    

def generate_pseudo_labels_from_features(image_features, text_features, num_classes, device):
    """
    使用模型输出的image_features和text_features生成伪标签
    
    Args:
        image_features: 图像特征 [batch_size, feature_dim]
        text_features: 文本特征 [batch_size, num_classes, feature_dim]
        num_classes: 类别数量
        device: 设备
    
    Returns:
        pseudo_labels: 伪标签 [batch_size, num_classes]
    """
    with torch.no_grad():
        batch_size = image_features.size(0)
        
        # 归一化特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [batch_size, feature_dim]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # [batch_size, num_classes, feature_dim]
        
        # 计算每个图像与每个类别文本的相似度
        # image_features: [batch_size, feature_dim] -> [batch_size, 1, feature_dim]
        # text_features: [batch_size, num_classes, feature_dim]
        image_features_expanded = image_features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # 计算相似度: [batch_size, num_classes]
        logits = torch.sum(image_features_expanded * text_features, dim=-1)
        
        # 生成one-hot伪标签
        pseudo_labels = torch.zeros(batch_size, num_classes, device=device)
        pred_classes = logits.argmax(dim=-1)  # [batch_size]
        pred_classes = torch.clamp(pred_classes, 0, num_classes - 1)
        pseudo_labels.scatter_(1, pred_classes.unsqueeze(1), 1.0)
        
        return pseudo_labels