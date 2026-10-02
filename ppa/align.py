"""Progressive permutation alignment with temperature-controlled spectral sharpening.

Stage 1 (Alg. 1): for each temperature ``tau`` the layers are visited one by one; each layer's
correlation with the partially aligned model B is sharpened by :func:`ppa.spectral.sharpen`.
Stage 2 ("Ref."): the Hungarian projection of the stage-1 result initialises a layer-wise
weight-matching coordinate ascent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

from ppa.ops import Params, apply_matrices, apply_permutation, correlation_matrix, perm_size, permute_param
from ppa.spec import PermutationSpec
from ppa.spectral import hungarian, project_to_permutation, sharpen

DEFAULT_TAU_SCHEDULE = (1.0, 0.5, 0.0)


@dataclass
class AlignmentResult:
    """Output of :func:`align`.

    Attributes:
        perms: Permutation id -> index vector (``aligned_b = apply_permutation(spec, b, perms)``).
        soft_matrices: Stage-1 soft permutations before the Hungarian projection.
        stage1_time: Wall-clock seconds of stage 1.
        stage2_time: Wall-clock seconds of stage 2 (0 if ``refine=False``).
        history: Stage-1 alignment cost ``||A - T(B)||^2`` after every sweep.
    """

    perms: Dict[str, torch.Tensor]
    soft_matrices: Dict[str, torch.Tensor]
    stage1_time: float = 0.0
    stage2_time: float = 0.0
    history: List[float] = field(default_factory=list)

    @property
    def time(self) -> float:
        """Total alignment time in seconds."""
        return self.stage1_time + self.stage2_time


def squared_distance(params_a: Params, params_b: Params) -> float:
    """``sum_k ||A_k - B_k||_F^2`` over the keys of ``params_a`` present in ``params_b``."""
    total = 0.0
    for k, a in params_a.items():
        if k in params_b:
            total += torch.sum((a - params_b[k]) ** 2).item()
    return total


def l1_distance(params_a: Params, params_b: Params) -> float:
    """``sum_k |A_k - B_k|_1`` over the keys of ``params_a`` present in ``params_b``."""
    total = 0.0
    for k, a in params_a.items():
        if k in params_b:
            total += torch.sum(torch.abs(a.float() - params_b[k].float().to(a.device))).item()
    return total


def spectral_alignment(
    spec: PermutationSpec,
    params_a: Params,
    params_b: Params,
    tau_schedule: Sequence[float] = DEFAULT_TAU_SCHEDULE,
    max_sweeps: int = 100,
    tol: float = 1e-6,
    use_svd: bool = True,
    order: Optional[Sequence[str]] = None,
    device: str = "cpu",
    log: Optional[Callable[[str], None]] = None,
) -> tuple[Dict[str, torch.Tensor], List[float]]:
    """Stage 1: progressive layer-wise alignment with spectral annealing.

    Within one temperature, sweeps are repeated while the cost ``||A - T(B)||^2`` decreases by
    more than ``tol`` (at most ``max_sweeps``); the best cost is carried across temperatures.
    Each layer update uses the current soft matrices of all other layers.

    Args:
        spec: Permutation spec.
        params_a: Reference model.
        params_b: Model to align.
        tau_schedule: Temperatures, from 1 down to 0.
        max_sweeps: Maximum sweeps per temperature.
        tol: Minimum cost decrease to keep sweeping.
        use_svd: Use the spectral power (``False`` = "w/o SVD" ablation).
        order: Layer visiting order. Defaults to ``sorted(perm ids)``, the order used for the paper.
        device: Torch device.
        log: Optional logging callback.

    Returns:
        Soft matrices per permutation id, and the cost after every sweep.
    """
    a = {k: v.to(device) for k, v in params_a.items()}
    b = {k: v.to(device) for k, v in params_b.items()}

    sizes = {pid: perm_size(spec, a, pid) for pid, axes in spec.perm_to_axes.items() if any(n in a for n, _ in axes)}
    dtype = next(iter(a.values())).dtype
    matrices = {pid: torch.eye(n, device=device, dtype=dtype) for pid, n in sizes.items()}
    visit = list(order) if order is not None else sorted(sizes)

    best = squared_distance(a, apply_matrices(spec, b, matrices, device))
    history = [best]
    for tau in tau_schedule:
        tau = float(tau)
        for sweep in range(max_sweeps):
            for pid in visit:
                others = dict(matrices)
                others[pid] = torch.eye(sizes[pid], device=device, dtype=dtype)
                b_partial = apply_matrices(spec, b, others, device)
                corr = correlation_matrix(spec, a, b_partial, pid)
                matrices[pid] = sharpen(corr, tau, use_svd=use_svd)
            cost = squared_distance(a, apply_matrices(spec, b, matrices, device))
            history.append(cost)
            if log is not None:
                log(f"[stage1] tau={tau:.3f} sweep={sweep + 1} cost={cost:.6f}")
            if cost < best - tol:
                best = cost
            else:
                break
    return matrices, history


def weight_matching_refinement(
    spec: PermutationSpec,
    params_a: Params,
    params_b: Params,
    init_perms: Dict[str, torch.Tensor],
    max_sweeps: int = 10000,
    tol: float = 1e-15,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, torch.Tensor]:
    """Stage 2: layer-wise weight-matching coordinate ascent starting from ``init_perms``.

    Layers are visited in spec order. A layer's new Hungarian solution is kept only if the total
    matching score increases by more than ``tol``; iteration stops after a sweep with no update.
    Runs on CPU.

    Args:
        spec: Permutation spec.
        params_a: Reference model.
        params_b: Model to align.
        init_perms: Initial permutation id -> index vector.
        max_sweeps: Maximum number of sweeps.
        tol: Minimum score increase to accept an update.
        log: Optional logging callback.

    Returns:
        Refined permutations (CPU index vectors).
    """
    a = {k: v.cpu() for k, v in params_a.items()}
    b = {k: v.cpu() for k, v in params_b.items()}
    perms = {k: v.cpu() for k, v in init_perms.items()}
    pids = [pid for pid in spec.perm_to_axes if pid in perms]

    def overlap(pid: str) -> torch.Tensor:
        n = len(perms[pid])
        total: Optional[torch.Tensor] = None
        for name, axis in spec.perm_to_axes[pid]:
            pa = a[name]
            pb = permute_param(spec, perms, b, name, except_perm_id=pid)
            if pa.shape[axis] != n:
                pa = pa.narrow(axis, pa.shape[axis] - n, n)
                pb = pb.narrow(axis, pb.shape[axis] - n, n)
            pa = torch.moveaxis(pa, axis, 0)
            pb = torch.moveaxis(pb, axis, 0)
            o = torch.outer(pa, pb) if pa.dim() == 1 else pa.flatten(1) @ pb.flatten(1).T
            total = o if total is None else total + o
        assert total is not None
        return total

    def score() -> float:
        return sum(overlap(pid)[torch.arange(len(perms[pid])), perms[pid]].sum().item() for pid in pids)

    prev = score()
    for sweep in range(max_sweeps):
        updated = False
        for pid in pids:
            old = perms[pid].clone()
            perms[pid] = hungarian(overlap(pid))
            new = score()
            if new > prev + tol:
                prev, updated = new, True
            else:
                perms[pid] = old
        if log is not None:
            log(f"[stage2] sweep={sweep + 1} score={prev:.6f}")
        if not updated:
            break
    return perms


def align(
    spec: PermutationSpec,
    params_a: Params,
    params_b: Params,
    tau_schedule: Sequence[float] = DEFAULT_TAU_SCHEDULE,
    refine: bool = True,
    use_svd: bool = True,
    max_sweeps: int = 100,
    tol: float = 1e-6,
    refine_max_sweeps: int = 10000,
    order: Optional[Sequence[str]] = None,
    device: str = "cpu",
    log: Optional[Callable[[str], None]] = None,
    dtype: Optional[torch.dtype] = None,
) -> AlignmentResult:
    """Align ``params_b`` to ``params_a`` (the proposed method, "Ours").

    Args:
        spec: Permutation spec.
        params_a: Reference model (only parameters in ``spec`` are used).
        params_b: Model to align.
        tau_schedule: Temperature schedule of stage 1.
        refine: Run the weight-matching refinement (``False`` = "Ours w/o Ref.").
        use_svd: Use the spectral power (``False`` = "w/o SVD" ablation).
        max_sweeps: Maximum stage-1 sweeps per temperature.
        tol: Stage-1 convergence tolerance.
        refine_max_sweeps: Maximum stage-2 sweeps.
        order: Stage-1 layer order (default ``sorted`` permutation ids).
        device: Device for stage 1 (stage 2 always runs on CPU).
        log: Optional logging callback.
        dtype: Compute in this precision (default: that of the parameters). ``torch.float64`` separates
            units that differ only by ~1e-12, e.g. the near-zero ("dead") filters of a pretrained ResNet18,
            which float32 cannot tell apart.

    Returns:
        The alignment result.
    """
    a = {k: v if dtype is None else v.to(dtype) for k, v in params_a.items() if k in spec.axes_to_perm}
    b = {k: v if dtype is None else v.to(dtype) for k, v in params_b.items() if k in spec.axes_to_perm}

    _sync()
    t0 = time.perf_counter()
    soft, history = spectral_alignment(spec, a, b, tau_schedule, max_sweeps, tol, use_svd, order, device, log)
    perms = {pid: project_to_permutation(q) for pid, q in soft.items()}
    _sync()
    t1 = time.perf_counter()
    if refine:
        perms = weight_matching_refinement(spec, a, b, perms, refine_max_sweeps, log=log)
    _sync()
    t2 = time.perf_counter()
    return AlignmentResult(
        perms={k: v.cpu() for k, v in perms.items()},
        soft_matrices={k: v.cpu() for k, v in soft.items()},
        stage1_time=t1 - t0,
        stage2_time=t2 - t1 if refine else 0.0,
        history=history,
    )


def align_and_apply(
    spec: PermutationSpec, params_a: Params, params_b: Params, **kwargs: object
) -> tuple[Params, AlignmentResult]:
    """Run :func:`align` and return the aligned copy of the full ``params_b``.

    Args:
        spec: Permutation spec.
        params_a: Reference model.
        params_b: Model to align (full state dict allowed).
        **kwargs: Forwarded to :func:`align`.

    Returns:
        ``(aligned_b, result)``.
    """
    result = align(spec, params_a, params_b, **kwargs)  # type: ignore[arg-type]
    return apply_permutation(spec, params_b, result.perms), result


def _sync() -> None:
    """Synchronise CUDA so that timings include queued kernels."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
