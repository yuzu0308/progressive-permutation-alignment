"""Weight Matching baselines (Git Re-Basin, Ainsworth et al., ICLR 2023), PyTorch re-implementation.

* :func:`weight_matching` ("WM" in the paper): coordinate ascent over permutation groups in a
  random order, repeated for ``num_trials`` orders (seeds ``base_seed + t``); the best run is kept.
* :func:`progressive_weight_matching` ("Progressive WM"): the same coordinate ascent, visiting the
  groups in spec order (input to output).

The acceptance criterion reproduces the code used for the paper: after each group update, the
score is ``sum_g max_P Tr(P^T O_g)`` over all groups ``g`` (each group's overlap ``O_g`` evaluated
with the other groups' current permutations and maximised by a fresh Hungarian solve); an update
is kept only if this score increases by more than ``tol``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ppa.ops import Params, permute_param
from ppa.spec import PermutationSpec
from ppa.spectral import hungarian


def overlap_matrix(
    spec: PermutationSpec, params_a: Params, params_b: Params, perms: Dict[str, torch.Tensor], perm_id: str
) -> torch.Tensor:
    """Unnormalised weight overlap of group ``perm_id`` given the other groups' permutations.

    Args:
        spec: Permutation spec.
        params_a: Reference model.
        params_b: Model to align.
        perms: Current permutations.
        perm_id: Group to compute the overlap for.

    Returns:
        ``(n, n)`` matrix ``sum_layers A_axis (P B)_axis^T``.

    Raises:
        KeyError: If the group has no parameters.
    """
    n = len(perms[perm_id])
    total: Optional[torch.Tensor] = None
    for name, axis in spec.perm_to_axes[perm_id]:
        pa = params_a[name]
        pb = permute_param(spec, perms, params_b, name, except_perm_id=perm_id)
        if pa.shape[axis] != n:
            pa = pa.narrow(axis, pa.shape[axis] - n, n)
            pb = pb.narrow(axis, pb.shape[axis] - n, n)
        pa = torch.moveaxis(pa, axis, 0)
        pb = torch.moveaxis(pb, axis, 0)
        o = torch.outer(pa, pb) if pa.dim() == 1 else pa.flatten(1) @ pb.flatten(1).T
        total = o if total is None else total + o
    if total is None:
        raise KeyError(f"no parameter of permutation {perm_id} found")
    return total


def _total_score(spec: PermutationSpec, a: Params, b: Params, perms: Dict[str, torch.Tensor]) -> float:
    score = 0.0
    for pid in spec.perm_to_axes:
        o = overlap_matrix(spec, a, b, perms, pid)
        best = hungarian(o)
        score += o[torch.arange(len(best)), best].sum().item()
    return score


def _identity_perms(spec: PermutationSpec, params: Params) -> Dict[str, torch.Tensor]:
    perms = {}
    for pid, axes in spec.perm_to_axes.items():
        name, axis = axes[0]
        perms[pid] = torch.arange(params[name].shape[axis], dtype=torch.long, device=params[name].device)
    return perms


def coordinate_ascent(
    spec: PermutationSpec,
    params_a: Params,
    params_b: Params,
    order: Sequence[str],
    max_sweeps: int = 10000,
    tol: float = 1e-15,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Weight-matching coordinate ascent from the identity, visiting groups in ``order``.

    Args:
        spec: Permutation spec.
        params_a: Reference model.
        params_b: Model to align.
        order: Group visiting order.
        max_sweeps: Maximum number of sweeps.
        tol: Minimum score increase to accept an update.
        log: Optional logging callback.

    Returns:
        ``(permutations, final score)``.
    """
    perms = _identity_perms(spec, params_a)
    prev = _total_score(spec, params_a, params_b, perms)
    for sweep in range(max_sweeps):
        updated = False
        for pid in order:
            new_perm = hungarian(overlap_matrix(spec, params_a, params_b, perms, pid))
            old = perms[pid].clone()
            perms[pid] = new_perm
            new = _total_score(spec, params_a, params_b, perms)
            if new > prev + tol:
                prev, updated = new, True
            else:
                perms[pid] = old
        if log is not None:
            log(f"[wm] sweep={sweep + 1} score={prev:.6f}")
        if not updated:
            break
    return perms, prev


def progressive_weight_matching(
    spec: PermutationSpec,
    params_a: Params,
    params_b: Params,
    max_sweeps: int = 10000,
    tol: float = 1e-15,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, torch.Tensor]:
    """Progressive WM: coordinate ascent in spec order (input to output).

    Args:
        spec: Permutation spec.
        params_a: Reference model.
        params_b: Model to align.
        max_sweeps: Maximum number of sweeps.
        tol: Minimum score increase to accept an update.
        log: Optional logging callback.

    Returns:
        Permutations.
    """
    np.random.seed(0)  # the original implementation reseeded NumPy here; kept for identical side effects
    return coordinate_ascent(spec, params_a, params_b, list(spec.perm_to_axes), max_sweeps, tol, log)[0]


def weight_matching(
    spec: PermutationSpec,
    params_a: Params,
    params_b: Params,
    num_trials: int = 10,
    base_seed: int = 42,
    max_sweeps: int = 10000,
    tol: float = 1e-15,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, torch.Tensor]:
    """WM (Git Re-Basin): best of ``num_trials`` coordinate ascents with random group orders.

    Trial ``t`` shuffles the groups with ``np.random.seed(base_seed + t); np.random.shuffle``.
    Note that this reseeds NumPy's global RNG, as the original implementation did.

    Args:
        spec: Permutation spec.
        params_a: Reference model.
        params_b: Model to align.
        num_trials: Number of random orders.
        base_seed: Seed of the first trial.
        max_sweeps: Maximum sweeps per trial.
        tol: Minimum score increase to accept an update.
        log: Optional logging callback.

    Returns:
        Permutations of the trial with the highest final score (earliest trial on ties).
    """
    best_perms: Optional[Dict[str, torch.Tensor]] = None
    best_score = float("-inf")
    for t in range(num_trials):
        np.random.seed(base_seed + t)
        order: List[str] = list(spec.perm_to_axes)
        np.random.shuffle(order)
        perms, score = coordinate_ascent(spec, params_a, params_b, order, max_sweeps, tol)
        if log is not None:
            log(f"[wm] trial={t + 1}/{num_trials} seed={base_seed + t} score={score:.6f}")
        if score > best_score:
            best_score = score
            best_perms = {k: v.clone() for k, v in perms.items()}
    assert best_perms is not None
    return best_perms
