"""Image metrics used in the paper."""

from __future__ import annotations

import torch
from torchmetrics.functional import structural_similarity_index_measure


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """SSIM (torchmetrics defaults: Gaussian window 11, sigma 1.5) of ``(H, W, 3)`` images in ``[0, 1]``."""
    value = structural_similarity_index_measure(
        pred.permute(2, 0, 1).unsqueeze(0), gt.permute(2, 0, 1).unsqueeze(0), data_range=1.0
    )
    return float(value.item())
