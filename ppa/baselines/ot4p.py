"""OT4P baselines (Guo et al., NeurIPS 2024): gradient descent over an orthogonal relaxation.

Adapted from https://github.com/YamingGuo98/OT4P (MIT License, Copyright (c) 2024 Yaming Guo).
The temperature map ``orth2perm`` is OT4P's original one (``variant="original"``, default): the
matrix power uses an eigendecomposition and there is no anchoring to the identity. The variant used
by our method (:func:`ppa.spectral.sharpen`, SVD power + identity anchoring) is available as
``variant="modified"``; some runs of the research code used it for the baselines as well.

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

from ppa.align import squared_distance
from ppa.ops import Params, apply_matrices, perm_size
from ppa.spec import PermutationSpec
from ppa.spectral import _hungarian_anchor, project_to_permutation, sharpen

VARIANTS = ("original", "modified")


def eig_power(matrix: torch.Tensor, s: float) -> torch.Tensor:
    """OT4P's matrix power ``V diag(L^s) V^{-1}`` via an eigendecomposition (real part returned).

    Computed on the CPU: complex powers are JIT-compiled by PyTorch on CUDA, which fails on GPUs newer
    than the installed CUDA toolkit (e.g. an RTX 5090 with CUDA 11.8 wheels).
    """
    eigvals, eigvecs = torch.linalg.eig(matrix.detach().cpu())
    out = torch.real(eigvecs @ torch.diag_embed(eigvals**s) @ torch.linalg.inv(eigvecs))
    return out.to(matrix.device)


def orth2perm(orth: torch.Tensor, tau: float) -> torch.Tensor:
    """OT4P's original ``orth2perm`` for one ``(n, n)`` orthogonal matrix.

    ``P_hat = Hungarian(O)`` and the result is ``P_hat (P_hat^T O)^tau`` with an eigendecomposition
    power, with OT4P's determinant sign adjustment of the last column.
    """
    o = orth.unsqueeze(0)
    p_hat = _hungarian_anchor(o)
    adjust = (-2 * (torch.linalg.det(p_hat) < 0).float() + 1).unsqueeze(1)
    p_hat[:, :, -1].mul_(adjust)
    out = p_hat @ eig_power(p_hat.transpose(-2, -1) @ o, tau)
    out[:, :, -1].mul_(adjust)
    return out.squeeze(0)


def random_block_rotation(batch: int, n: int) -> torch.Tensor:
    """OT4P's random initial base: 2x2 rotations with angles in ``[pi/8, 3pi/8]`` (uses NumPy's RNG)."""
    n_even = n - 1 if n % 2 != 0 else n
    angles = np.random.uniform(np.pi / 8, 3 * np.pi / 8, (batch, n_even // 2))
    out = torch.zeros((batch, n, n), dtype=torch.float32)
    for b in range(batch):
        q = torch.eye(n, dtype=torch.float32)
        for i in range(0, n_even, 2):
            c, s = np.cos(angles[b, i // 2]), np.sin(angles[b, i // 2])
            q[i, i], q[i, i + 1], q[i + 1, i], q[i + 1, i + 1] = c, -s, s, c
        if n % 2 != 0:
            q[-1, -1] = 1
        out[b] = q
    return out


class OT4P(nn.Module):
    """Orthogonal relaxation ``R^{n x n} -> O(n)`` concentrated near permutations as ``tau -> 0``."""

    def __init__(self, size: int, batch: int = 1, variant: str = "original") -> None:
        """Create the module with a random base rotation.

        Args:
            size: Matrix size ``n``.
            batch: Batch size.
            variant: ``original`` (OT4P's eigendecomposition map) or ``modified`` (:func:`sharpen`).

        Raises:
            ValueError: For an unknown variant.
        """
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant}")
        self.register_buffer("base", random_block_rotation(batch, size))
        self.target_map = orth2perm if variant == "original" else sharpen

    @staticmethod
    def ver2orth(matrix: torch.Tensor) -> torch.Tensor:
        """Map an unconstrained matrix to ``SO(n)`` via the exponential of its skew-symmetric part."""
        skew = matrix.triu(1)
        return torch.matrix_exp(skew - skew.transpose(-2, -1))

    def forward(self, matrix: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Relaxed permutation for parameters ``matrix`` (``(n, n)`` or ``(batch, n, n)``)."""
        squeeze = matrix.dim() == 2
        if squeeze:
            matrix = matrix.unsqueeze(0)
        orth = self.base @ self.ver2orth(matrix)
        with torch.no_grad():
            target = torch.stack([self.target_map(o, tau) for o in orth])
            trans = target @ orth.transpose(-2, -1)
        out = trans @ orth
        return out.squeeze(0) if squeeze else out

    @torch.no_grad()
    def update_base(self, param: torch.Tensor) -> None:
        """Absorb the current parameters into the base and reset them to zero."""
        matrix = param.unsqueeze(0) if param.dim() == 2 else param
        self.base.copy_(self.base @ self.ver2orth(matrix))
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
        modules[pid] = OT4P(n, variant=variant).to(device)
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

    Returns:
        The result.
    """
    a = {k: v.to(device) for k, v in params_a.items() if k in spec.axes_to_perm}
    b = {k: v.to(device) for k, v in params_b.items() if k in spec.axes_to_perm}
    modules, weights, _, order = _setup(spec, a, device, seed, init_scale, variant)
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

    Returns:
        The result.
    """
    a = {k: v.to(device) for k, v in params_a.items() if k in spec.axes_to_perm}
    b = {k: v.to(device) for k, v in params_b.items() if k in spec.axes_to_perm}
    modules, weights, sizes, order = _setup(spec, a, device, seed, variant=variant)
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
