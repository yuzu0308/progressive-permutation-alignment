"""Baselines compared in the paper."""

from ppa.baselines.ot4p import OT4PResult, ot4p, progressive_ot4p
from ppa.baselines.weight_matching import progressive_weight_matching, weight_matching

__all__ = ["OT4PResult", "ot4p", "progressive_ot4p", "progressive_weight_matching", "weight_matching"]
