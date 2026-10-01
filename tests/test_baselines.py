"""Tests of the baselines and the shuffle-recovery benchmark helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pytest
import torch
from conftest import random_perms

from ppa import apply_permutation
from ppa.align import squared_distance
from ppa.baselines import ot4p, progressive_ot4p, progressive_weight_matching, weight_matching
from ppa.ops import apply_matrices
from ppa.spec import permutation_spec_from_axes_to_perm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.mark.parametrize("fn", [weight_matching, progressive_weight_matching])
def test_weight_matching_recovers_shuffle(skip_mlp: Dict[str, object], fn) -> None:
    spec = permutation_spec_from_axes_to_perm(skip_mlp["axes"])  # type: ignore[arg-type]
    params = skip_mlp["params"]
    w = skip_mlp["width"]
    shuffled = apply_permutation(spec, params, random_perms({"P_0": w, "P_1": w, "P_2": w}, seed=5))  # type: ignore[arg-type]
    perms = fn(spec, params, shuffled)  # type: ignore[arg-type]
    assert squared_distance(params, apply_permutation(spec, shuffled, perms)) == 0.0  # type: ignore[arg-type]


@pytest.mark.parametrize("variant", ["original", "modified"])
@pytest.mark.parametrize("fn", [ot4p, progressive_ot4p])
def test_ot4p_is_repeatable_with_seed(skip_mlp: Dict[str, object], fn, variant: str) -> None:
    spec = permutation_spec_from_axes_to_perm(skip_mlp["axes"])  # type: ignore[arg-type]
    params = skip_mlp["params"]
    r1 = fn(spec, params, params, tau=0.5, seed=3, max_iterations=20, variant=variant)  # type: ignore[arg-type]
    r2 = fn(spec, params, params, tau=0.5, seed=3, max_iterations=20, variant=variant)  # type: ignore[arg-type]
    for k in r1.matrices:
        assert torch.equal(r1.matrices[k], r2.matrices[k])
    for q in r1.matrices.values():
        torch.testing.assert_close(q @ q.T, torch.eye(q.shape[0]), atol=1e-4, rtol=1e-4)


def test_ot4p_original_map_endpoints() -> None:
    """OT4P's orth2perm returns the input at tau = 1 and the Hungarian anchor at tau = 0 (det > 0 case)."""
    from ppa.baselines.ot4p import orth2perm
    from ppa.spectral import hungarian

    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    p0 = orth2perm(q, 0.0)
    assert p0.abs().argmax(1).tolist() == hungarian(q).tolist()
    sign = torch.sign(torch.linalg.det(torch.eye(8)[hungarian(q)]))
    expected = q.clone()
    expected[:, -1] *= sign
    torch.testing.assert_close(orth2perm(q, 1.0), expected, atol=1e-4, rtol=1e-4)


def test_apply_matrices_conv_matches_index_permutation() -> None:
    """The N-D path of ``apply_matrices`` agrees with index permutation on conv kernels."""
    g = torch.Generator().manual_seed(0)
    params = {"c1.weight": torch.randn(6, 3, 3, 3, generator=g), "c2.weight": torch.randn(4, 6, 3, 3, generator=g)}
    spec = permutation_spec_from_axes_to_perm(
        {"c1.weight": ["P", None, None, None], "c2.weight": [None, "P", None, None]}
    )
    perm = torch.randperm(6, generator=g)
    by_index = apply_permutation(spec, params, {"P": perm})
    by_matrix = apply_matrices(spec, params, {"P": torch.eye(6)[perm]})
    for k in params:
        assert torch.equal(by_index[k], by_matrix[k])


def test_shuffle_precision_definition() -> None:
    import shuffle_recovery as sr

    from ppa.zoo import mlp5_spec

    torch.manual_seed(0)
    from ppa.zoo import MLP5, state_params

    spec = mlp5_spec()
    a = state_params(MLP5())
    b, gt = sr.shuffle(spec, a, seed=1)
    inverse = {pid: torch.argsort(p) for pid, p in gt.items()}
    assert squared_distance(a, apply_permutation(spec, b, inverse)) == 0.0
    assert sr.precision(inverse, gt) == 1.0
    identity = {pid: torch.arange(len(p)) for pid, p in gt.items()}
    assert sr.precision(identity, gt) < 0.05
