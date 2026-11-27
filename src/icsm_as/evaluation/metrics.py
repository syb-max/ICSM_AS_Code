"""Capacity metrics with explicit, non-interchangeable reference denominators."""
from __future__ import annotations

def normalized_capacity_ratio(capacity: float, greedy_capacity: float) -> float:
    """NCR = C_selected / C_Capacity-Greedy (main ICSM-AS results)."""
    if greedy_capacity <= 0: raise ValueError("Capacity-Greedy denominator must be positive")
    return float(capacity/greedy_capacity)

def hq_referenced_ratio(capacity: float, hq_capacity: float) -> float:
    """R_C = C_selected / C_HQ (static-score development results only)."""
    if hq_capacity <= 0: raise ValueError("HQ denominator must be positive")
    return float(capacity/hq_capacity)
