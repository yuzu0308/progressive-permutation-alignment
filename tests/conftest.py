"""Shared pytest fixtures."""

from __future__ import annotations

import warnings
from typing import Dict

import pytest
import torch

warnings.filterwarnings("ignore", category=UserWarning)


def random_perms(sizes: Dict[str, int], seed: int) -> Dict[str, torch.Tensor]:
    """Random permutation index vector for every group."""
    g = torch.Generator().manual_seed(seed)
    return {pid: torch.randperm(n, generator=g) for pid, n in sizes.items()}


@pytest.fixture
def skip_mlp() -> Dict[str, object]:
    """Small NeRF-like MLP: 3 hidden layers of width 8, layer 2 takes ``[x, h_1]`` (skip)."""
    g = torch.Generator().manual_seed(0)
    d_in, w = 5, 8

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    params = {
        "l0.weight": rnd(w, d_in),
        "l0.bias": rnd(w),
        "l1.weight": rnd(w, w),
        "l1.bias": rnd(w),
        "l2.weight": rnd(w, d_in + w),
        "l2.bias": rnd(w),
        "out.weight": rnd(2, w),
        "out.bias": rnd(2),
    }
    axes = {
        "l0.weight": ["P_0", None],
        "l0.bias": ["P_0"],
        "l1.weight": ["P_1", "P_0"],
        "l1.bias": ["P_1"],
        "l2.weight": ["P_2", "P_1"],
        "l2.bias": ["P_2"],
        "out.weight": [None, "P_2"],
        "out.bias": [None],
    }
    return {"params": params, "axes": axes, "d_in": d_in, "width": w}


def skip_mlp_forward(params: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the :func:`skip_mlp` fixture."""
    h = torch.relu(x @ params["l0.weight"].T + params["l0.bias"])
    h = torch.relu(h @ params["l1.weight"].T + params["l1.bias"])
    h = torch.relu(torch.cat([x, h], dim=-1) @ params["l2.weight"].T + params["l2.bias"])
    return h @ params["out.weight"].T + params["out.bias"]
