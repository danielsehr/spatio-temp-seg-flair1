import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, ignore_index: int | None = None, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, H, W]
        targets: [B, H, W], where ignored pixels = ignore_index
        """
        # Compute Cross-Entropy per pixel
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index
        )

        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce

        # Mask out ignored pixels
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            focal = focal[valid_mask]

        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int | None = None, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, H, W]
        targets: [B, H, W], where ignored pixels = ignore_index
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Remap targets: subtract 1 for labels > ignore_index
        if self.ignore_index is not None:
            targets_fixed = targets.clone()
            targets_fixed[targets_fixed != self.ignore_index] -= 1
        else:
            targets_fixed = targets

        # One-hot encoding for Dice computation
        targets_one_hot = F.one_hot(targets_fixed, num_classes).permute(0, 3, 1, 2).float()

        # Mask out ignored pixels
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1)  # [B,1,H,W]
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        dims = (0, 2, 3)  # sum over batch + height + width
        intersection = torch.sum(probs * targets_one_hot, dims)
        union = torch.sum(probs + targets_one_hot, dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Optionally skip ignored class
        if self.ignore_index == 0:
            dice = dice[1:]

        return 1.0 - dice.mean()


class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smooth: float = 1.0,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int | None = None
    ):
        super().__init__()
        self.focal = FocalLoss(ignore_index=ignore_index, alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(ignore_index=ignore_index, smooth=smooth)
        self.fw = focal_weight
        self.dw = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.fw * self.focal(logits, targets) + self.dw * self.dice(logits, targets)