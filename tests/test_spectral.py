"""Tests for the spectral sharpening step."""

from __future__ import annotations

import itertools

import pytest
import torch

from ppa.spectral import hungarian, project_to_permutation, sharpen, svd_power


def _is_permutation_matrix(m: torch.Tensor) -> bool:
    return bool(((m == 0) | (m == 1)).all() and (m.sum(0) == 1).all() and (m.sum(1) == 1).all())


def test_hungarian_matches_brute_force() -> None:
    torch.manual_seed(0)
    c = torch.randn(5, 5)
    best = max(itertools.permutations(range(5)), key=lambda p: sum(c[i, p[i]].item() for i in range(5)))
    assert hungarian(c).tolist() == list(best)
    assert project_to_permutation(c).tolist() == list(best)


def test_svd_power_endpoints() -> None:
    torch.manual_seed(0)
    m = torch.randn(6, 6)
    torch.testing.assert_close(svd_power(m, 1.0), m, atol=1e-5, rtol=1e-5)
    q = svd_power(m, 0.0)
    torch.testing.assert_close(q @ q.T, torch.eye(6), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_svd", [True, False])
def test_sharpen_tau_zero_is_hungarian_anchor(use_svd: bool) -> None:
    torch.manual_seed(1)
    c = torch.randn(7, 7)
    out = sharpen(c, 0.0, use_svd=use_svd)
    assert _is_permutation_matrix(out)
    assert out.argmax(1).tolist() == hungarian(c).tolist()


def _last_column_sign(c: torch.Tensor) -> torch.Tensor:
    """Sign that OT4P's determinant adjustment applies to the last column."""
    p_hat = torch.eye(c.shape[0])[hungarian(c)]
    sign = torch.ones(c.shape[0])
    sign[-1] = torch.sign(torch.linalg.det(p_hat))
    return sign


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("use_svd", [True, False])
def test_sharpen_tau_one_returns_correlation(use_svd: bool, seed: int) -> None:
    """tau = 1 returns C, except that the last column is negated when det(P_hat) < 0."""
    torch.manual_seed(seed)
    c = torch.randn(7, 7)
    expected = c * _last_column_sign(c)
    torch.testing.assert_close(sharpen(c, 1.0, use_svd=use_svd), expected, atol=1e-4, rtol=1e-4)


def test_det_adjustment_cases_are_covered() -> None:
    """The seeds above include both determinant signs, so both branches are exercised."""
    signs = set()
    for seed in range(6):
        torch.manual_seed(seed)
        signs.add(_last_column_sign(torch.randn(7, 7))[-1].item())
    assert signs == {1.0, -1.0}


def test_sharpen_orthogonal_part_at_intermediate_tau() -> None:
    """(update - (1 - tau) P_hat) / tau = P_hat M^<tau> has singular values sigma(P_hat^T C)^tau.

    The determinant sign flip of the last column only multiplies by an orthogonal diagonal
    matrix, so singular values are unaffected.
    """
    torch.manual_seed(4)
    c = torch.randn(6, 6)
    p_hat = sharpen(c, 0.0)
    tau = 0.5
    s_in = torch.linalg.svdvals(p_hat.T @ c)
    s_out = torch.linalg.svdvals((sharpen(c, tau) - (1 - tau) * p_hat) / tau)
    torch.testing.assert_close(s_out, s_in**tau, atol=1e-4, rtol=1e-4)
