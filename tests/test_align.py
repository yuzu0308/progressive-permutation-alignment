"""End-to-end tests of the proposed alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
import torch
from conftest import random_perms

from ppa import align, apply_permutation
from ppa.align import squared_distance
from ppa.nerf import filter_nerf_params, load_state_dict, vanilla_nerf_permutation_spec
from ppa.spec import permutation_spec_from_axes_to_perm

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
SCENES = ["lego", "chair", "ficus", "hotdog", "mic", "ship", "materials", "drums"]


@pytest.mark.parametrize("refine", [False, True])
@pytest.mark.parametrize("use_svd", [True, False])
def test_recovers_shuffled_copy(skip_mlp: Dict[str, object], refine: bool, use_svd: bool) -> None:
    spec = permutation_spec_from_axes_to_perm(skip_mlp["axes"])  # type: ignore[arg-type]
    params = skip_mlp["params"]
    w = skip_mlp["width"]
    shuffled = apply_permutation(spec, params, random_perms({"P_0": w, "P_1": w, "P_2": w}, seed=3))  # type: ignore[arg-type]
    result = align(spec, params, shuffled, refine=refine, use_svd=use_svd)  # type: ignore[arg-type]
    aligned = apply_permutation(spec, shuffled, result.perms)
    assert squared_distance(params, aligned) == 0.0  # type: ignore[arg-type]


def _random_nerf_params(seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    shapes = {}
    for net in ("field_coarse", "field_fine"):
        base = f"{net}.mlp_base.layers"
        shapes[f"{base}.0.weight"] = (256, 63)
        for i in range(1, 8):
            shapes[f"{base}.{i}.weight"] = (256, 319 if i == 4 else 256)
        for i in range(8):
            shapes[f"{base}.{i}.bias"] = (256,)
        shapes[f"{net}.field_output_density.net.weight"] = (1, 256)
        shapes[f"{net}.field_output_density.net.bias"] = (1,)
        shapes[f"{net}.mlp_head.layers.0.weight"] = (128, 256 + 27)
        shapes[f"{net}.mlp_head.layers.0.bias"] = (128,)
        shapes[f"{net}.mlp_head.layers.1.weight"] = (128, 128)
        shapes[f"{net}.mlp_head.layers.1.bias"] = (128,)
        shapes[f"{net}.field_heads.0.net.weight"] = (3, 128)
        shapes[f"{net}.field_heads.0.net.bias"] = (3,)
    return {k: torch.randn(*s, generator=g) * 0.1 for k, s in shapes.items()}


@pytest.mark.parametrize("refine", [False, True])
def test_recovers_shuffled_nerf(refine: bool) -> None:
    """Table 2 setting on random weights: a shuffled copy is recovered exactly."""
    spec = vanilla_nerf_permutation_spec()
    params = _random_nerf_params(0)
    sizes = {pid: (128 if "HEAD" in pid else 256) for pid in spec.perm_to_axes}
    shuffled = apply_permutation(spec, params, random_perms(sizes, seed=4))
    result = align(spec, params, shuffled, refine=refine)
    assert squared_distance(params, apply_permutation(spec, shuffled, result.perms)) == 0.0


@pytest.mark.parametrize("scene", SCENES)
def test_matches_original_research_code(scene: str) -> None:
    """Permutations are bit-identical to the code that produced the paper results.

    ``weights/<scene>/ref_perms_cpu.pt`` is produced by the original implementation
    (``codes/algorithm/nerf3d_bcd.py`` and ``nerf3d_bcd_wm_merged.py``); the test is skipped when
    the released weights are not downloaded.
    """
    ref_path = WEIGHTS / scene / "ref_perms_cpu.pt"
    if not ref_path.exists():
        pytest.skip(f"{ref_path} not found (run scripts/download_weights.py)")
    ref = torch.load(ref_path)
    spec = vanilla_nerf_permutation_spec()
    a = filter_nerf_params(load_state_dict(WEIGHTS / scene / "model_a.pth"))
    b = filter_nerf_params(load_state_dict(WEIGHTS / scene / "model_b.pth"))
    for key, refine in (("pa", False), ("pa_wm", True)):
        perms = align(spec, a, b, refine=refine).perms
        for pid, expected in ref[key].items():
            assert torch.equal(perms[pid], expected), f"{scene} {key} {pid}"


def test_float64_separates_near_duplicate_units(skip_mlp: Dict[str, object]) -> None:
    """Units that differ only by ~1e-9 are indistinguishable in float32 but recovered in float64."""
    spec = permutation_spec_from_axes_to_perm(skip_mlp["axes"])  # type: ignore[arg-type]
    params = {k: v.clone().double() for k, v in skip_mlp["params"].items()}  # type: ignore[union-attr]
    for name in ("l0.weight", "l0.bias", "l1.weight"):
        params[name][[2, 5]] = 1e-9 * torch.arange(1, 3, dtype=torch.float64).reshape(
            2, *([1] * (params[name].dim() - 1))
        )
    params["l1.weight"][:, [2, 5]] = 1e-9 * torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    w = skip_mlp["width"]
    shuffled = apply_permutation(spec, params, random_perms({"P_0": w, "P_1": w, "P_2": w}, seed=6))  # type: ignore[arg-type]
    result = align(spec, params, shuffled, dtype=torch.float64)
    assert squared_distance(params, apply_permutation(spec, shuffled, result.perms)) == 0.0
