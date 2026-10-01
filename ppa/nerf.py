"""Vanilla NeRF (nerfstudio ``vanilla-nerf``) specific helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from ppa.spec import PermutationSpec, permutation_spec_from_axes_to_perm

NERF_NETWORKS = ("field_coarse", "field_fine")


def vanilla_nerf_permutation_spec() -> PermutationSpec:
    """Permutation spec of nerfstudio's vanilla NeRF (coarse and fine fields).

    Each field has an 8-layer, 256-wide base MLP whose layer 4 takes ``[x_enc, h_3]`` as input
    (skip connection, the 63 encoded input channels come first and are never permuted), a
    density head, a 2-layer 128-wide colour MLP and an RGB head.

    Returns:
        The permutation spec. Permutation ids are ``{field}_P_{i}`` and ``{field}_HEAD_P_{i}``.
    """
    axes_to_perm: Dict[str, List[Optional[str]]] = {}
    for net in NERF_NETWORKS:
        base = f"{net}.mlp_base.layers"
        axes_to_perm[f"{base}.0.weight"] = [f"{net}_P_0", None]
        axes_to_perm[f"{base}.0.bias"] = [f"{net}_P_0"]
        for i in range(1, 8):
            axes_to_perm[f"{base}.{i}.weight"] = [f"{net}_P_{i}", f"{net}_P_{i - 1}"]
            axes_to_perm[f"{base}.{i}.bias"] = [f"{net}_P_{i}"]
        axes_to_perm[f"{net}.field_output_density.net.weight"] = [None, f"{net}_P_7"]
        axes_to_perm[f"{net}.field_output_density.net.bias"] = [None]
        axes_to_perm[f"{net}.mlp_head.layers.0.weight"] = [f"{net}_HEAD_P_0", f"{net}_P_7"]
        axes_to_perm[f"{net}.mlp_head.layers.0.bias"] = [f"{net}_HEAD_P_0"]
        axes_to_perm[f"{net}.mlp_head.layers.1.weight"] = [f"{net}_HEAD_P_1", f"{net}_HEAD_P_0"]
        axes_to_perm[f"{net}.mlp_head.layers.1.bias"] = [f"{net}_HEAD_P_1"]
        axes_to_perm[f"{net}.field_heads.0.net.weight"] = [None, f"{net}_HEAD_P_1"]
        axes_to_perm[f"{net}.field_heads.0.net.bias"] = [None]
    return permutation_spec_from_axes_to_perm(axes_to_perm)


def filter_nerf_params(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep only the coarse/fine field parameters of a nerfstudio model state dict.

    Args:
        params: Full model state dict.

    Returns:
        The subset whose keys start with ``field_coarse`` or ``field_fine``, in original order.
    """
    return {k: v for k, v in params.items() if k.startswith(NERF_NETWORKS)}


def load_state_dict(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Load a model state dict saved either bare or wrapped in a checkpoint dict.

    Args:
        path: ``.pth`` / ``.ckpt`` file.

    Returns:
        The state dict on CPU.

    Raises:
        ValueError: If the file does not contain a tensor dictionary.
    """
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "pipeline"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict) or not all(isinstance(v, torch.Tensor) for v in obj.values()):
        raise ValueError(f"{path} does not contain a state dict of tensors")
    if any(k.startswith("_model.") for k in obj):
        # nerfstudio pipeline checkpoint (step-*.ckpt): strip the pipeline prefix.
        obj = {k[len("_model.") :]: v for k, v in obj.items() if k.startswith("_model.")}
    return obj
