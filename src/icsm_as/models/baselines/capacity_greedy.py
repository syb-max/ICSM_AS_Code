"""Capacity-Greedy reference."""
from __future__ import annotations
from ...core.capacity_teacher import greedy_teacher, capacity_validated

def capacity_greedy(gram, snr_linear: float, k: int = 16):
    return greedy_teacher(gram, snr_linear, k)

def score_capacity(gram, snr_linear, subset, k: int = 16):
    return capacity_validated(gram, snr_linear, subset, final_k=k)
