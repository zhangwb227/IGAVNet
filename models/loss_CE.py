import torch
import torch.nn as nn
import torch.nn.functional as F

"""Focal Tversky Loss"""
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        # 保证 target 有 channel 维度
        if len(target.shape) == 3:
            target = target.unsqueeze(1)

        pred = torch.sigmoid(pred)
        target = target.float()

        tp = torch.sum(pred * target)
        fp = torch.sum(pred * (1 - target))
        fn = torch.sum((1 - pred) * target)

        denominator = tp + self.alpha * fn + self.beta * fp + self.smooth
        denominator = torch.clamp(denominator, min=self.smooth)  # 防止除0

        tversky = (tp + self.smooth) / denominator
        tversky = torch.clamp(tversky, min=0.0, max=1.0)  # 限制到 [0,1]

        loss = (1 - tversky) ** self.gamma
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)

        return loss
