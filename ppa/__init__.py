"""Progressive Permutation Alignment with Temperature-Controlled Spectral Sharpening (ACCV 2026)."""

from ppa.align import AlignmentResult, align, align_and_apply, spectral_alignment, weight_matching_refinement
from ppa.ops import apply_permutation, interpolate
from ppa.spec import PermutationSpec, mlp_permutation_spec, permutation_spec_from_axes_to_perm

__all__ = [
    "AlignmentResult",
    "PermutationSpec",
    "align",
    "align_and_apply",
    "apply_permutation",
    "interpolate",
    "mlp_permutation_spec",
    "permutation_spec_from_axes_to_perm",
    "spectral_alignment",
    "weight_matching_refinement",
]

__version__ = "1.0.0"
