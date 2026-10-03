"""Temperature-controlled spectral sharpening (Eq. (5)-(7), Alg. 1 lines 6-10).

Adapted from ``OT4P.orth2perm`` and ``utils.matching`` of OT4P
(https://github.com/YamingGuo98/OT4P, MIT License, Copyright (c) 2024 Yaming Guo).
Changes: the matrix power uses the SVD ``U diag(S^tau) V^T`` instead of an eigendecomposition,
the powered matrix is linearly anchored to the identity, ``tau * M^<tau> + (1 - tau) I``,
and the linear assignment uses SciPy. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def hungarian(matrix: torch.Tensor) -> torch.Tensor:
    """Maximum-weight assignment of a square matrix.

    Args:
        matrix: ``(n, n)`` score matrix.

    Returns:
        Index vector ``col`` of length ``n`` (``col[i]`` is matched to row ``i``) on the input device.
    """
    _, col = linear_sum_assignment(matrix.detach().cpu().numpy(), maximize=True)
    return torch.as_tensor(col, dtype=torch.long, device=matrix.device)


def project_to_permutation(matrix: torch.Tensor) -> torch.Tensor:
    """Hungarian projection of a soft permutation onto the nearest permutation.

    Solves ``min_P Tr(P^T (-Q))`` (written this way, rather than with ``maximize=True``, to break
    ties exactly as in the published runs).

    Args:
        matrix: ``(n, n)`` soft permutation ``Q``.

    Returns:
        Index vector of length ``n`` on the input device.
    """
    _, col = linear_sum_assignment(-matrix.detach().cpu().numpy())
    return torch.as_tensor(col, dtype=torch.long, device=matrix.device)


def _hungarian_anchor(corr: torch.Tensor) -> torch.Tensor:
    """Permutation matrix ``P_hat = argmax_P Tr(P^T C)`` for a batch ``(1, n, n)``.

    Reproduces OT4P's ``matching`` exactly (shift to non-negative, flip, minimise) so that ties are
    broken the same way as in the published runs.
    """
    m = corr.clone()
    m -= m.min()
    m = m.max() - m
    batch, n, _ = m.shape
    cost = m.detach().cpu().numpy()
    perm = torch.zeros(batch, n, n, device=corr.device, dtype=corr.dtype)
    rows = torch.arange(n, device=corr.device)
    for i in range(batch):
        _, col = linear_sum_assignment(cost[i])
        perm[i, rows, torch.as_tensor(col, dtype=torch.long, device=corr.device)] = 1
    return perm


def svd_power(matrix: torch.Tensor, tau: float) -> torch.Tensor:
    """Spectral power ``M^<tau> = U diag(S^tau) V^T`` (Eq. (6)).

    Args:
        matrix: ``(..., n, n)`` matrix.
        tau: Exponent.

    Returns:
        ``(..., n, n)`` matrix.
    """
    u, s, vh = torch.linalg.svd(matrix)
    return torch.real(u @ torch.diag_embed(s**tau) @ vh)


def sharpen(corr: torch.Tensor, tau: float, use_svd: bool = True) -> torch.Tensor:
    """One temperature-controlled update of a layer's soft permutation (Eq. (7)).

    ``P_hat = Hungarian(C)``, ``M = P_hat^T C`` and the update is
    ``P_hat [tau * M^<tau> + (1 - tau) I]``. With ``use_svd=False`` the spectral power is skipped,
    ``P_hat [tau * M + (1 - tau) I]`` (the "w/o SVD" ablation). As in OT4P, the last column of
    ``P_hat`` is sign-flipped when ``det(P_hat) < 0`` and flipped back after the update.

    ``tau = 1`` returns (numerically) ``C`` itself and ``tau = 0`` returns ``P_hat``.

    Args:
        corr: ``(n, n)`` correlation matrix of the layer.
        tau: Temperature in ``[0, 1]``.
        use_svd: Apply the spectral power.

    Returns:
        ``(n, n)`` updated soft permutation.
    """
    c = corr.unsqueeze(0)
    p_hat = _hungarian_anchor(c)
    dets = torch.linalg.det(p_hat)
    adjust = (-2 * (dets < 0).float() + 1).unsqueeze(1)
    p_hat[:, :, -1].mul_(adjust)
    if use_svd:
        powered = svd_power(p_hat.transpose(-2, -1) @ c, tau)
        powered = tau * powered + (1 - tau) * torch.eye(c.size(-1), device=c.device, dtype=c.dtype)
        out = p_hat @ powered
    else:
        # P_hat [tau P_hat^T C + (1 - tau) I] = tau C + (1 - tau) P_hat, written as in the ablation run.
        out = tau * c + (1 - tau) * p_hat
    out[:, :, -1].mul_(adjust)
    return out.squeeze(0)
