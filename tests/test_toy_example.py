"""3-layer linear toy example of the supplementary material, Sec. 5.

Only the statements that were verified numerically are tested: the correlation matrices of
Eqs. (12)-(13) and the first Weight Matching step (P2 -> (2, 1, 0), trace 37).
"""

from __future__ import annotations

import itertools

import torch

from ppa.spectral import hungarian

W_A = [
    torch.tensor([[5.0, 5, -3], [3, -1, -2], [-4, 1, -3]]),
    torch.tensor([[2.0, 4, 5], [-5, 1, -3], [0, -2, -5]]),
    torch.tensor([[-5.0, -4, 5], [-2, 2, -1], [-3, 2, -5]]),
]
W_B = [
    torch.tensor([[3.0, 4, 5], [2, 5, 3], [1, 2, -4]]),
    torch.tensor([[-2.0, -1, 4], [-3, 2, 2], [2, 2, 4]]),
    torch.tensor([[0.0, -5, -3], [5, -4, 2], [1, 4, 3]]),
]
I3 = torch.eye(3)


def c1(p2: torch.Tensor) -> torch.Tensor:
    """Eq. (10)."""
    return W_A[0] @ W_B[0].T + W_A[1].T @ p2 @ W_B[1]


def c2(p1: torch.Tensor) -> torch.Tensor:
    """Eq. (11)."""
    return W_A[1] @ (p1 @ W_B[1]).T + W_A[2].T @ W_B[2]


def test_correlation_matrices() -> None:
    assert torch.equal(c1(I3), torch.tensor([[31.0, 14, 25], [-20, -11, 19], [-34, -33, 4]]))
    assert torch.equal(c2(I3), torch.tensor([[-1.0, 33, 34], [9, 31, 2], [-28, -55, -56]]))
    assert torch.trace(c1(I3)).item() == 24
    assert torch.trace(c2(I3)).item() == -26


def test_first_weight_matching_step() -> None:
    best = hungarian(c2(I3))
    assert best.tolist() == [2, 1, 0]
    assert sum(c2(I3)[i, best[i]].item() for i in range(3)) == 37
    traces = [sum(c2(I3)[i, p[i]].item() for i in range(3)) for p in itertools.permutations(range(3))]
    assert max(traces) == 37
