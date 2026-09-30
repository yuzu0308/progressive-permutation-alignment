"""Tests for permutation application and functional equivalence."""

from __future__ import annotations

from typing import Dict

import torch
from conftest import random_perms, skip_mlp_forward

from ppa.ops import apply_matrices, apply_permutation, correlation_matrix, interpolate
from ppa.spec import mlp_permutation_spec, permutation_spec_from_axes_to_perm


def test_permutation_preserves_function(skip_mlp: Dict[str, object]) -> None:
    spec = permutation_spec_from_axes_to_perm(skip_mlp["axes"])  # type: ignore[arg-type]
    params = skip_mlp["params"]
    w = skip_mlp["width"]
    perms = random_perms({"P_0": w, "P_1": w, "P_2": w}, seed=0)
    permuted = apply_permutation(spec, params, perms)  # type: ignore[arg-type]
    assert any(not torch.equal(permuted[k], params[k]) for k in params)  # type: ignore[index]
    x = torch.randn(4, skip_mlp["d_in"])  # type: ignore[arg-type]
    torch.testing.assert_close(skip_mlp_forward(permuted, x), skip_mlp_forward(params, x))  # type: ignore[arg-type]


def test_matrix_and_index_application_agree(skip_mlp: Dict[str, object]) -> None:
    spec = permutation_spec_from_axes_to_perm(skip_mlp["axes"])  # type: ignore[arg-type]
    params = skip_mlp["params"]
    w = skip_mlp["width"]
    perms = random_perms({"P_0": w, "P_1": w, "P_2": w}, seed=1)
    mats = {pid: torch.eye(w)[p] for pid, p in perms.items()}
    by_index = apply_permutation(spec, params, perms)  # type: ignore[arg-type]
    by_matrix = apply_matrices(spec, params, mats)  # type: ignore[arg-type]
    for k in by_index:
        assert torch.equal(by_index[k], by_matrix[k]), k


def test_skip_layer_keeps_input_channels_fixed(skip_mlp: Dict[str, object]) -> None:
    spec = permutation_spec_from_axes_to_perm(skip_mlp["axes"])  # type: ignore[arg-type]
    params = skip_mlp["params"]
    w, d_in = skip_mlp["width"], skip_mlp["d_in"]
    perms = random_perms({"P_0": w, "P_1": w, "P_2": w}, seed=2)
    out = apply_permutation(spec, params, perms)  # type: ignore[arg-type]
    fixed = out["l2.weight"][:, :d_in]  # type: ignore[misc]
    assert torch.equal(fixed, params["l2.weight"][perms["P_2"], :d_in])  # type: ignore[index]


def test_correlation_is_normalised_per_layer() -> None:
    spec = mlp_permutation_spec(1)
    g = torch.Generator().manual_seed(0)
    a = {
        "layers.0.weight": torch.randn(4, 3, generator=g),
        "layers.0.bias": torch.randn(4, generator=g),
        "layers.1.weight": torch.randn(2, 4, generator=g),
        "layers.1.bias": torch.randn(2, generator=g),
    }
    c = correlation_matrix(spec, a, a, "P_0")
    terms = [
        a["layers.0.weight"] @ a["layers.0.weight"].T,
        torch.outer(a["layers.0.bias"], a["layers.0.bias"]),
        a["layers.1.weight"].T @ a["layers.1.weight"],
    ]
    expected = sum(t / (t.norm() + 1e-8) for t in terms)
    torch.testing.assert_close(c, expected)


def test_interpolate_endpoints() -> None:
    a = {"w": torch.zeros(3)}
    b = {"w": torch.ones(3)}
    assert torch.equal(interpolate(a, b, 0.0)["w"], a["w"])
    assert torch.equal(interpolate(a, b, 1.0)["w"], b["w"])
    torch.testing.assert_close(interpolate(a, b, 0.25)["w"], torch.full((3,), 0.25))
