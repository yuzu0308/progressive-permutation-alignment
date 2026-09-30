"""Tensor operations shared by all alignment methods.

Several parameters are wider than their permutation along an axis (e.g. the NeRF skip layer
takes ``[x_enc, h]``). In that case the leading ``shape[axis] - n`` entries are treated as fixed
and only the trailing ``n`` entries are permuted.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ppa.spec import PermutationSpec

Params = Dict[str, torch.Tensor]


def _split(param: torch.Tensor, axis: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``param`` along ``axis`` into its fixed head and permutable tail of length ``n``."""
    num_fixed = param.shape[axis] - n
    return param.narrow(axis, 0, num_fixed), param.narrow(axis, num_fixed, n)


def permutable_part(param: torch.Tensor, axis: int, n: int) -> torch.Tensor:
    """Return the trailing ``n`` entries of ``param`` along ``axis`` (the whole tensor if equal)."""
    if param.shape[axis] == n:
        return param
    return _split(param, axis, n)[1]


def perm_size(spec: PermutationSpec, params: Params, perm_id: str) -> int:
    """Size of permutation ``perm_id``, read from the first output axis (axis 0) it acts on.

    Args:
        spec: Permutation spec.
        params: Parameters containing the layers of ``perm_id``.
        perm_id: Permutation id.

    Returns:
        Number of neurons permuted by ``perm_id``.

    Raises:
        KeyError: If none of the layers of ``perm_id`` is present in ``params``.
    """
    axes = [(n, a) for n, a in spec.perm_to_axes[perm_id] if n in params]
    if not axes:
        raise KeyError(f"no parameter of permutation {perm_id} found")
    for name, axis in axes:
        if axis == 0:
            return params[name].shape[0]
    name, axis = axes[0]
    return params[name].shape[axis]


def correlation_matrix(spec: PermutationSpec, params_a: Params, params_b: Params, perm_id: str) -> torch.Tensor:
    """Layer-normalised weight correlation used by the spectral alignment stage.

    For every ``(layer, axis)`` of ``perm_id`` the matrix ``A_axis B_axis^T`` (rows = neurons of
    that axis, remaining axes flattened) is divided by its Frobenius norm and the results are
    summed. This per-layer normalisation is not written in Eq. (3) of the paper but is what the
    published results were produced with.

    Args:
        spec: Permutation spec.
        params_a: Reference model parameters.
        params_b: Model parameters, already transformed by all other permutations.
        perm_id: Permutation to compute the correlation for.

    Returns:
        ``(n, n)`` correlation matrix, ``C[i, j]`` = similarity of neuron ``i`` of A and ``j`` of B.

    Raises:
        KeyError: If no parameter of ``perm_id`` is found.
    """
    axes = [(n, a) for n, a in spec.perm_to_axes[perm_id] if n in params_a]
    if not axes:
        raise KeyError(f"no parameter of permutation {perm_id} found")
    first_name, first_axis = axes[0]
    n = params_a[first_name].shape[first_axis]

    total: Optional[torch.Tensor] = None
    for name, axis in axes:
        if name not in params_b:
            continue
        a = torch.moveaxis(permutable_part(params_a[name], axis, n), axis, 0)
        b = torch.moveaxis(permutable_part(params_b[name], axis, n), axis, 0)
        corr = torch.outer(a, b) if a.dim() == 1 else a.flatten(1) @ b.flatten(1).T
        corr = corr / (torch.norm(corr, p="fro") + 1e-8)
        if total is None:
            total = corr
        elif corr.shape == total.shape:
            total += corr
    if total is None:
        raise KeyError(f"no parameter of permutation {perm_id} found in params_b")
    return total


def apply_matrices(
    spec: PermutationSpec, params: Params, matrices: Dict[str, torch.Tensor], device: str = "cpu"
) -> Params:
    """Apply (soft) per-group transforms ``Q``: output axes become ``Q W``, input axes ``W Q^T``.

    1-D and 2-D parameters use plain matrix products (the path used for the NeRF results);
    higher-dimensional parameters (convolution kernels) are transformed along ``axis`` via
    ``movedim``. Groups are applied in the iteration order of ``matrices``; keep it fixed for reproducibility.

    Args:
        spec: Permutation spec.
        params: Parameters to transform (not modified).
        matrices: Permutation id -> ``(n, n)`` matrix.
        device: Device of the returned tensors.

    Returns:
        New parameter dictionary on ``device``.
    """
    out = {k: v.clone().to(device) for k, v in params.items()}
    for perm_id, q in matrices.items():
        q = q.to(device)
        n = q.shape[0]
        for name, axis in spec.perm_to_axes[perm_id]:
            if name not in out:
                continue
            p = out[name]
            if p.dim() > 2:
                p = _transform_axis(p, q, axis)
            elif p.shape[axis] != n:
                fixed, part = _split(p, axis, n)
                part = q @ part if axis == 0 else part @ q.T
                p = torch.cat([fixed, part], dim=axis)
            else:
                p = q @ p if axis == 0 else p @ q.T
            out[name] = p
    return out


def _transform_axis(p: torch.Tensor, q: torch.Tensor, axis: int) -> torch.Tensor:
    """Multiply ``q`` into axis ``axis`` of an N-D tensor: ``out[.., i, ..] = sum_j q[i, j] p[.., j, ..]``."""
    moved = torch.movedim(p, axis, 0)
    out = (q @ moved.reshape(moved.shape[0], -1)).reshape(moved.shape)
    return torch.movedim(out, 0, axis)


def permute_param(
    spec: PermutationSpec,
    perms: Dict[str, torch.Tensor],
    params: Params,
    name: str,
    except_perm_id: Optional[str] = None,
) -> torch.Tensor:
    """Permute every axis of one parameter, optionally leaving the axes of one group untouched.

    ``perms[perm_id][i] = j`` means neuron ``i`` of the aligned model is neuron ``j`` of the
    original model (``index_select`` convention).

    Args:
        spec: Permutation spec.
        perms: Permutation id -> index vector.
        params: Parameters.
        name: Parameter to permute.
        except_perm_id: Group whose axes are skipped.

    Returns:
        The permuted parameter.
    """
    p = params[name].clone()
    for axis, perm_id in enumerate(spec.axes_to_perm.get(name, [])):
        if perm_id is None or perm_id == except_perm_id:
            continue
        idx = perms[perm_id].to(p.device)
        if p.shape[axis] != len(idx):
            fixed, part = _split(p, axis, len(idx))
            p = torch.cat([fixed, torch.index_select(part, axis, idx)], dim=axis)
        else:
            p = torch.index_select(p, axis, idx)
    return p


def apply_permutation(spec: PermutationSpec, params: Params, perms: Dict[str, torch.Tensor]) -> Params:
    """Apply hard permutations to a parameter dictionary.

    Parameters that are not in the spec (e.g. positional encodings, LPIPS buffers) are copied
    unchanged, so a full model state dict can be passed.

    Args:
        spec: Permutation spec.
        params: Parameters (not modified).
        perms: Permutation id -> index vector (on the same device as ``params``).

    Returns:
        New parameter dictionary.
    """
    out = dict(params)
    for name in params:
        if name in spec.axes_to_perm:
            out[name] = permute_param(spec, perms, params, name)
    return out


def interpolate(params_a: Params, params_b: Params, lam: float) -> Params:
    """Linear interpolation ``(1 - lam) * A + lam * B`` over the keys shared by both models.

    Args:
        params_a: Parameters at ``lam = 0``.
        params_b: Parameters at ``lam = 1``.
        lam: Interpolation coefficient.

    Returns:
        Interpolated parameters.
    """
    return {k: (1 - lam) * v + lam * params_b[k] for k, v in params_a.items() if k in params_b}
