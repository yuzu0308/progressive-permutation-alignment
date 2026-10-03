"""Tests of the baselines and the shuffle-recovery benchmark helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
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
    from ppa.baselines.ot4p import OT4P
    from ppa.spectral import hungarian

    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    module = OT4P(8)
    p0 = module.orth2perm(q.unsqueeze(0), 0.0)[0]
    assert p0.abs().argmax(1).tolist() == hungarian(q).tolist()
    sign = torch.sign(torch.linalg.det(torch.eye(8)[hungarian(q)]))
    expected = q.clone()
    expected[:, -1] *= sign
    torch.testing.assert_close(module.orth2perm(q.unsqueeze(0), 1.0)[0], expected, atol=1e-4, rtol=1e-4)


# Output of the official implementation (github.com/YamingGuo98/OT4P, commit 99915004) for
# np.random.seed(0); torch.manual_seed(0); OT4P(4)(W, tau) on CPU.
OFFICIAL_W = [[0.3, -1.2, 0.5, 0.8], [1.1, 0.2, -0.7, 0.4], [-0.6, 0.9, 0.1, -1.3], [0.2, -0.4, 1.5, 0.6]]
OFFICIAL_BASE = [
    [0.6794847, -0.7336896, 0.0, 0.0],
    [0.7336896, 0.6794847, 0.0, 0.0],
    [0.0, 0.0, 0.5780923, -0.8159714],
    [0.0, 0.0, 0.8159714, 0.5780923],
]
OFFICIAL_OUT = {
    0.3: [
        [-0.0240355, -0.2172269, 0.9358802, -0.2763402],
        [0.2098972, 0.008196, 0.283568, 0.9356628],
        [0.9763373, -0.0531844, -0.047548, -0.2041465],
        [0.046155, 0.9746373, 0.2036094, -0.0806004],
    ],
    0.5: [
        [-0.0775384, -0.3537061, 0.8264081, -0.4311966],
        [0.3377643, -0.0276537, 0.4507442, 0.82582],
        [0.9358137, -0.0834666, -0.1160402, -0.3222116],
        [0.0644574, 0.9312152, 0.3168812, -0.1681409],
    ],
}


@pytest.mark.parametrize("tau", [0.3, 0.5])
def test_ot4p_matches_official_outputs(tau: float) -> None:
    from ppa.baselines.ot4p import OT4P

    np.random.seed(0)
    torch.manual_seed(0)
    module = OT4P(4)
    torch.testing.assert_close(module.base[0], torch.tensor(OFFICIAL_BASE), atol=1e-6, rtol=0)
    out = module(torch.tensor(OFFICIAL_W), tau=tau)
    # float32 eigendecomposition differs by ~1e-6 across LAPACK builds (CI runners vs. MKL)
    torch.testing.assert_close(out, torch.tensor(OFFICIAL_OUT[tau]), atol=1e-5, rtol=0)


def test_ot4p_does_not_swallow_errors() -> None:
    """A failing matrix power propagates (the research code used to replace it by the identity)."""
    from ppa.baselines.ot4p import matrix_power

    with pytest.raises(RuntimeError):
        matrix_power(torch.full((1, 3, 3), float("nan")), 0.5)


def test_update_base_rejects_batched_parameters() -> None:
    from ppa.baselines.ot4p import OT4P

    with pytest.raises(ValueError):
        OT4P(3).update_base(torch.zeros(2, 3, 3))


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
