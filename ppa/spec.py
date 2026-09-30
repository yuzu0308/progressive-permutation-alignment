"""Permutation specifications describing which weight axes share a neuron permutation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple


class PermutationSpec(NamedTuple):
    """Bidirectional mapping between permutation groups and weight axes.

    Attributes:
        perm_to_axes: Permutation id -> list of ``(param_name, axis)`` it acts on.
        axes_to_perm: Param name -> permutation id per axis (``None`` if the axis is fixed).
    """

    perm_to_axes: Dict[str, List[Tuple[str, int]]]
    axes_to_perm: Dict[str, List[Optional[str]]]


def permutation_spec_from_axes_to_perm(
    axes_to_perm: Dict[str, List[Optional[str]]],
) -> PermutationSpec:
    """Build a :class:`PermutationSpec` from the per-parameter axis mapping.

    The insertion order of ``perm_to_axes`` follows the order in which permutation ids first
    appear in ``axes_to_perm``. Several routines iterate over this order, so it is part of the
    numerical behaviour and must stay stable.

    Args:
        axes_to_perm: Param name -> permutation id per axis.

    Returns:
        The corresponding permutation spec.
    """
    perm_to_axes: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for name, axis_perms in axes_to_perm.items():
        for axis, perm_id in enumerate(axis_perms):
            if perm_id is not None:
                perm_to_axes[perm_id].append((name, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def mlp_permutation_spec(num_hidden_layers: int, prefix: str = "layers") -> PermutationSpec:
    """Permutation spec for a plain MLP ``Linear -> act -> ... -> Linear``.

    Hidden layer ``i`` is permuted by ``P_i``; the input and output axes are fixed.

    Args:
        num_hidden_layers: Number of hidden layers (i.e. number of linear layers minus one).
        prefix: Module prefix of the linear layers (``{prefix}.{i}.weight``).

    Returns:
        The permutation spec.
    """
    axes_to_perm: Dict[str, List[Optional[str]]] = {}
    for i in range(num_hidden_layers + 1):
        out_perm = f"P_{i}" if i < num_hidden_layers else None
        in_perm = f"P_{i - 1}" if i > 0 else None
        axes_to_perm[f"{prefix}.{i}.weight"] = [out_perm, in_perm]
        axes_to_perm[f"{prefix}.{i}.bias"] = [out_perm]
    return permutation_spec_from_axes_to_perm(axes_to_perm)
