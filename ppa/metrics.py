"""Image metrics used in the paper.

Two SSIM implementations appear in the paper. ``ssim`` (torchmetrics, Gaussian window 11,
sigma 1.5) was used for the proposed method, its ablations, AM and STE, and is the default here.
``ssim_skimage`` (scikit-image defaults, 7x7 uniform window, computed on the saved 8-bit PNGs) was
used for the Naive / WM / Progressive WM rows of supplementary Tables 2-3.
"""

from __future__ import annotations

import numpy as np
import torch
from torchmetrics.functional import structural_similarity_index_measure


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """SSIM (torchmetrics defaults: Gaussian window 11, sigma 1.5) of ``(H, W, 3)`` images in ``[0, 1]``."""
    value = structural_similarity_index_measure(
        pred.permute(2, 0, 1).unsqueeze(0), gt.permute(2, 0, 1).unsqueeze(0), data_range=1.0
    )
    return float(value.item())


def ssim_skimage(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM with scikit-image defaults (7x7 uniform window) of ``(H, W, 3)`` float arrays in ``[0, 1]``."""
    from skimage.metrics import structural_similarity

    return float(structural_similarity(pred, gt, data_range=1.0, channel_axis=2))
