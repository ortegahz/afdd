import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossV0(nn.Module):
    def __init__(self, alpha=1, gamma=8, reduction='mean'):
        super(FocalLossV0, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt is the probability of the true class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class FocalLossV1(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class HardExampleMiningFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, hard_weight=2.0, hard_ratio=0.3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.hard_weight = hard_weight
        self.hard_ratio = hard_ratio

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_term = (1 - pt) ** self.gamma
        weight = torch.ones_like(targets)
        if self.hard_ratio < 1.0:
            valid_length = len(BCE_loss.flatten())
            k = max(1, min(
                int(self.hard_ratio * valid_length),
                valid_length - 1
            ))
            flattened_loss = BCE_loss.flatten()
            if len(flattened_loss) > 0:
                hard_loss, hard_indices = torch.topk(flattened_loss, k)
                weight.view(-1)[hard_indices] = self.hard_weight

        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        F_loss = weight * alpha_t * focal_term * BCE_loss
        return F_loss.mean()
