"""OT4P baselines (Guo et al., NeurIPS 2024): gradient descent over an orthogonal relaxation.

The :class:`OT4P` module and its helpers (``matrix_power``, ``matching``, ``create_perm_matrix``,
``generate_random_orthogonal_matrix``) follow the official implementation
https://github.com/YamingGuo98/OT4P (commit 99915004; MIT License, Copyright (c) 2024 Yaming Guo) line by
line, including its error handling: nothing is caught, and the matrix power prints
"Imaginary part is too large" like the original. Outputs, gradients and optimiser steps are bit-identical
to the official code (``tests/test_baselines.py`` checks the map against fixed values). Differences:

* the batched linear assignment uses SciPy instead of ``torch_linear_assignment`` (same solutions);
* ``power_device`` optionally computes the eigendecomposition power on another device. The default
  (``None``) uses the input's device as the official code does. On an RTX 5090 with CUDA 11.8 wheels,
  ``x ** 0.5`` of a complex CUDA tensor is dispatched to a JIT-compiled kernel that fails to compile, so
  the official code raises at ``tau = 0.5`` there; pass ``power_device="cpu"`` in that case;
* ``update_base`` raises a ``ValueError`` for parameters that are not 2-D (the official code fails with
  an ``UnboundLocalError``).

``variant="modified"`` replaces ``orth2perm`` by the map of our method (:func:`ppa.spectral.sharpen`,
SVD power + identity anchoring); it is only kept to compare with the research code.

* :func:`ot4p` ("OT4P"): all groups are optimised jointly with AdamW.
* :func:`progressive_ot4p` ("Progressive OT4P"): groups are optimised one at a time (sorted
  permutation ids), each with its own AdamW and early stopping, the others kept fixed.

Both start from random parameters. The original scripts did not fix a seed; pass ``seed`` to make
runs repeatable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from ppa.align import squared_distance
from ppa.ops import Params, apply_matrices, perm_size
from ppa.spec import PermutationSpec
from ppa.spectral import project_to_permutation, sharpen

VARIANTS = ("original", "modified")


# ---------------------------------------------------------------------------------------------------------
# Official OT4P helpers (src/utils.py)
# ---------------------------------------------------------------------------------------------------------


def batch_linear_assignment(cost: torch.Tensor) -> torch.Tensor:
    """Row-wise minimum-cost assignment of a ``(batch, n, n)`` cost tensor (``torch_linear_assignment`` API)."""
    cols = [torch.as_tensor(linear_sum_assignment(c)[1], dtype=torch.long) for c in cost.detach().cpu().numpy()]
    return torch.stack(cols).to(cost.device)


def matrix_power(matrix: torch.Tensor, s: float, power_device: Optional[str] = None) -> torch.Tensor:
    """Computes the power of a square matrix using eigen decomposition (official ``matrix_power``).

    Args:
        matrix: Batch x N x N.
        s: Scalar power.
        power_device: Device for the computation (``None``: the input's device, as in the official code).

    Returns:
        Batch x N x N matrix raised to the power ``s`` (real part), on the input's device.
    """
    m = matrix if power_device is None else matrix.to(power_device)
    L, V = torch.linalg.eig(m)  # noqa: N806 (official names)
    Lambda_s = torch.diag_embed(L**s)  # noqa: N806
    result = V @ Lambda_s @ torch.linalg.inv(V)
    if torch.imag(result).abs().max() > 1e-5:
        print("Imaginary part is too large")
    out = torch.real(result)
    if out.device != matrix.device:
        out = out.contiguous()
        if matrix.is_cuda:
            # Page-locked copy: the pageable host-to-device path intermittently failed with
            # "CUDA error: unknown error" for 4096 x 4096 matrices on an RTX 5090 (Windows, CUDA 11.8 wheels).
            out = out.pin_memory()
        out = out.to(matrix.device)
    return out


def create_perm_matrix(indx: torch.Tensor) -> torch.Tensor:
    """Creates a Batch x N x N permutation matrix from Batch x N indices (official ``create_perm_matrix``)."""
    device = indx.device
    batch_size, size = indx.shape
    perm_matrix = torch.zeros(batch_size, size, size, device=device, dtype=torch.float)
    rows = torch.arange(size, device=device).repeat(batch_size, 1)
    perm_matrix[torch.arange(batch_size, device=device).unsqueeze(1), rows, indx] = 1
    return perm_matrix


def matching(
    orth_matrix: torch.Tensor, constraint: Optional[torch.Tensor] = None, maximize: bool = True
) -> torch.Tensor:
    """Solves the linear assignment problem using the provided orthogonal matrix (official ``matching``).

    Args:
        orth_matrix: Batch x N x N orthogonal matrix.
        constraint: Optional Batch x N x N constraint matrix.
        maximize: Whether to maximize or minimize the assignment.

    Returns:
        Batch x N x N permutation matrix.
    """
    matrix = orth_matrix.clone()
    matrix -= matrix.min()
    if torch.is_tensor(constraint):
        matrix *= constraint
    if maximize:
        matrix = matrix.max() - matrix
        col_idxs = batch_linear_assignment(matrix)
    else:
        col_idxs = batch_linear_assignment(matrix)
    return create_perm_matrix(col_idxs)


def generate_random_orthogonal_matrix(batch_size: int, n: int) -> torch.Tensor:
    """Generates a batch of random orthogonal matrices (official, uses NumPy's global RNG).

    Args:
        batch_size: Number of matrices to generate.
        n: Size of each orthogonal matrix.

    Returns:
        Batch x N x N block-rotation matrices with angles in ``[pi/8, 3 pi/8]``.
    """
    n_minus_one = n - 1 if n % 2 != 0 else n
    theta_min = np.pi / 8
    theta_max = 3 * np.pi / 8
    angles = np.random.uniform(theta_min, theta_max, (batch_size, n_minus_one // 2))
    Q_batch = torch.zeros((batch_size, n, n), dtype=torch.float32)  # noqa: N806 (official names)
    for b in range(batch_size):
        Q = torch.eye(n, dtype=torch.float32)  # noqa: N806
        for i in range(0, n_minus_one, 2):
            theta = angles[b, i // 2]
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            Q[i, i] = cos_theta
            Q[i, i + 1] = -sin_theta
            Q[i + 1, i] = sin_theta
            Q[i + 1, i + 1] = cos_theta
        if n % 2 != 0:
            Q[-1, -1] = 1
        Q_batch[b] = Q
    return Q_batch


# ---------------------------------------------------------------------------------------------------------
# Official OT4P module (src/ot4p.py)
# ---------------------------------------------------------------------------------------------------------


class OT4P(nn.Module):
    """OT4P: maps unconstrained matrices to the orthogonal group, concentrated near permutations as tau -> 0."""

    def __init__(self, size: int, batch: int = 1, variant: str = "original", power_device: Optional[str] = None):
        """Initialize the OT4P model.

        Args:
            size: The size of the orthogonal matrix.
            batch: The batch size.
            variant: ``original`` (official ``orth2perm``) or ``modified`` (map of our method, research code only).
            power_device: Device for the eigendecomposition power (``None``: same as the input).

        Raises:
            ValueError: For an unknown variant.
        """
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant}")
        self.register_buffer("base", generate_random_orthogonal_matrix(batch, size))
        self.register_buffer("constraint", None)
        self.variant = variant
        self.power_device = power_device

    def ver2orth(self, matrix: torch.Tensor) -> torch.Tensor:
        """Map the vector (Batch x N x N) to the orthogonal matrix."""
        skew = matrix.triu(1)
        return torch.matrix_exp(skew - skew.transpose(-2, -1))

    def orth2perm(self, orth_matrix: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Moves the Batch x N x N orthogonal matrix toward the closest permutation matrix."""
        if self.variant == "modified":
            return torch.stack([sharpen(o, tau) for o in orth_matrix])
        perm_matrix = matching(orth_matrix, self.constraint)

        dets = torch.linalg.det(perm_matrix)
        adjust = (-2 * (dets < 0).float() + 1).unsqueeze(1)
        perm_matrix[:, :, -1].mul_(adjust)

        matrix_powered = matrix_power(perm_matrix.transpose(-2, -1) @ orth_matrix, tau, self.power_device)
        new_orth_matrix = perm_matrix @ matrix_powered

        new_orth_matrix[:, :, -1].mul_(adjust)
        return new_orth_matrix

    def get_trans_matrix(self, orth_matrix: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Compute the equivalent transformation of the mapping ``orth2perm``."""
        with torch.no_grad():
            perm_matrix = self.orth2perm(orth_matrix, tau)
            trans_matrix = perm_matrix @ orth_matrix.transpose(-2, -1)
        return trans_matrix

    def forward(self, matrix: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Map an unconstrained (Batch x) N x N tensor to an orthogonal matrix close to a permutation matrix."""
        is_2d = matrix.dim() == 2
        if is_2d:
            matrix = matrix.unsqueeze(0)

        orth_matrix = self.base @ self.ver2orth(matrix)
        trans_matrix = self.get_trans_matrix(orth_matrix, tau)
        perm_matrix = trans_matrix @ orth_matrix

        if is_2d:
            perm_matrix = perm_matrix.squeeze(0)
        return perm_matrix

    def update_base(self, param: torch.Tensor) -> None:
        """Update the base orthogonal matrix with an N x N parameter and reset the parameter to zero.

        Raises:
            ValueError: If ``param`` is not 2-D (the official code fails with an ``UnboundLocalError``).
        """
        with torch.no_grad():
            if param.dim() != 2:
                raise ValueError(f"update_base expects an N x N parameter, got shape {tuple(param.shape)}")
            matrix = param.unsqueeze(0)
            orth_matrix = self.base @ self.ver2orth(matrix)
            self.base.copy_(orth_matrix)
            param.set_(torch.zeros_like(param))


@dataclass
class OT4PResult:
    """Output of the OT4P baselines.

    Attributes:
        matrices: Final (near-)permutation matrices at ``tau = 0``; the paper applied these directly.
        perms: Their Hungarian projection as index vectors.
    """

    matrices: Dict[str, torch.Tensor]
    perms: Dict[str, torch.Tensor]


def _setup(
    spec: PermutationSpec,
    a: Params,
    device: str,
    seed: Optional[int],
    init_scale: float = 0.1,
    variant: str = "original",
    power_device: Optional[str] = None,
):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    modules: Dict[str, OT4P] = {}
    weights: Dict[str, nn.Parameter] = {}
    sizes: Dict[str, int] = {}
    for pid, axes in spec.perm_to_axes.items():
        if not axes or axes[0][0] not in a:
            continue
        n = perm_size(spec, a, pid)
        sizes[pid] = n
        modules[pid] = OT4P(n, variant=variant, power_device=power_device).to(device)
        init = torch.randn(n, n, device=device)
        weights[pid] = nn.Parameter(init if init_scale == 1.0 else init * init_scale)
    return modules, weights, sizes, sorted(sizes)


def _loss(spec: PermutationSpec, a: Params, b: Params, mats: Dict[str, torch.Tensor], device: str) -> torch.Tensor:
    tb = apply_matrices(spec, b, mats, device)
    loss = torch.tensor(0.0, device=device)
    for k, v in a.items():
        if k in tb:
            loss = loss + torch.sum((v - tb[k]) ** 2)
    return loss


def _finish(modules: Dict[str, OT4P], weights: Dict[str, nn.Parameter], order) -> OT4PResult:
    with torch.no_grad():
        mats = {pid: modules[pid](weights[pid].unsqueeze(0), tau=0.0).squeeze(0) for pid in order}
    return OT4PResult(
        matrices={k: v.cpu() for k, v in mats.items()},
        perms={k: project_to_permutation(v).cpu() for k, v in mats.items()},
    )


def ot4p(
    spec: PermutationSpec,
    params_a: Params,
    params_b: Params,
    tau: float = 0.5,
    lr: float = 0.05,
    max_iterations: int = 500,
    patience: int = 10,
    tol: float = 1e-6,
    seed: Optional[int] = None,
    device: str = "cpu",
    log: Optional[Callable[[str], None]] = None,
    timeout: Optional[float] = None,
    init_scale: float = 0.1,
    variant: str = "original",
    power_device: Optional[str] = None,
) -> OT4PResult:
    """OT4P: joint AdamW over all groups at a fixed temperature, with early stopping.

    Args:
        spec: Permutation spec.
        params_a: Reference model.
        params_b: Model to align.
        tau: Temperature.
        lr: AdamW learning rate.
        max_iterations: Maximum gradient steps.
        patience: Stop after this many steps without a decrease of more than ``tol``.
        tol: Improvement tolerance.
        seed: Seed for NumPy and torch (``None`` keeps the global RNG state).
        device: Torch device.
        log: Optional logging callback.
        timeout: Stop optimising after this many seconds (paper: 3600 for Table 1).
        init_scale: Std of the random initial parameters (paper runs: 0.1 for NeRF, 1.0 for MLP5/CNNs).
        variant: ``original`` OT4P map or the ``modified`` one of our method.
        power_device: Device of the eigendecomposition power (``None``: same as ``device``).

    Returns:
        The result.
    """
    a = {k: v.to(device) for k, v in params_a.items() if k in spec.axes_to_perm}
    b = {k: v.to(device) for k, v in params_b.items() if k in spec.axes_to_perm}
    modules, weights, _, order = _setup(spec, a, device, seed, init_scale, variant, power_device)
    optimizer = torch.optim.AdamW(weights.values(), lr=lr)
    best = float("inf")
    stall = 0
    start = time.perf_counter()
    for it in range(max_iterations):
        if timeout is not None and time.perf_counter() - start >= timeout:
            break
        optimizer.zero_grad()
        mats = {pid: modules[pid](weights[pid].unsqueeze(0), tau=tau).squeeze(0) for pid in order}
        loss = _loss(spec, a, b, mats, device)
        loss.backward()
        optimizer.step()
        for pid in order:
            modules[pid].update_base(weights[pid])
        cur = loss.item()
        if log is not None:
            log(f"[ot4p] it={it + 1} loss={cur:.6f}")
        if cur < best - tol:
            best, stall = cur, 0
        else:
            stall += 1
            if stall >= patience:
                break
    return _finish(modules, weights, order)


def progressive_ot4p(
    spec: PermutationSpec,
    params_a: Params,
    params_b: Params,
    tau: float = 0.5,
    lr: float = 0.05,
    max_iterations: int = 100,
    patience: int = 10,
    tol: float = 1e-6,
    seed: Optional[int] = None,
    device: str = "cpu",
    log: Optional[Callable[[str], None]] = None,
    timeout: Optional[float] = None,
    variant: str = "original",
    power_device: Optional[str] = None,
) -> OT4PResult:
    """Progressive OT4P: layer-by-layer AdamW at a fixed temperature.

    Args:
        spec: Permutation spec.
        params_a: Reference model.
        params_b: Model to align.
        tau: Temperature.
        lr: AdamW learning rate.
        max_iterations: Maximum gradient steps per group.
        patience: Stop a group after this many steps without a decrease of more than ``tol``.
        tol: Improvement tolerance.
        seed: Seed for NumPy and torch (``None`` keeps the global RNG state).
        device: Torch device.
        log: Optional logging callback.
        timeout: Stop optimising after this many seconds (paper: 3600 for Table 1).
        variant: ``original`` OT4P map or the ``modified`` one of our method.
        power_device: Device of the eigendecomposition power (``None``: same as ``device``).

    Returns:
        The result.
    """
    a = {k: v.to(device) for k, v in params_a.items() if k in spec.axes_to_perm}
    b = {k: v.to(device) for k, v in params_b.items() if k in spec.axes_to_perm}
    modules, weights, sizes, order = _setup(spec, a, device, seed, variant=variant, power_device=power_device)
    current: Dict[str, torch.Tensor] = {pid: torch.eye(sizes[pid], device=device) for pid in order}
    start = time.perf_counter()
    for pid in order:
        if timeout is not None and time.perf_counter() - start >= timeout:
            break
        optimizer = torch.optim.AdamW([weights[pid]], lr=lr)
        best = float("inf")
        stall = 0
        for _ in range(max_iterations):
            optimizer.zero_grad()
            current[pid] = modules[pid](weights[pid].unsqueeze(0), tau=tau).squeeze(0)
            loss = _loss(spec, a, b, current, device)
            loss.backward()
            optimizer.step()
            modules[pid].update_base(weights[pid])
            cur = loss.item()
            if cur < best - tol:
                best, stall = cur, 0
            else:
                stall += 1
                if stall >= patience:
                    break
        with torch.no_grad():
            current[pid] = modules[pid](weights[pid].unsqueeze(0), tau=tau).squeeze(0).detach()
        if log is not None:
            log(f"[prog-ot4p] {pid} done, cost={squared_distance(a, apply_matrices(spec, b, current, device)):.6f}")
    return _finish(modules, weights, order)
