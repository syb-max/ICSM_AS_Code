"""Correlation-Greedy baseline."""
from __future__ import annotations
import numpy as np

def correlation_greedy(gram: np.ndarray, k: int = 16) -> np.ndarray:
    g = np.asarray(gram)
    power = np.maximum(np.diagonal(g, axis1=-2, axis2=-1).real.mean(0), 0)
    corr = np.abs(g).mean(0) / np.sqrt(np.maximum(power[:, None] * power[None, :], 1e-24))
    selected = []
    penalty = np.zeros(len(power))
    for _ in range(k):
        score = power / (1.0 + penalty)
        score[selected] = -np.inf
        idx = int(np.argmax(score))
        selected.append(idx)
        penalty += corr[:, idx] ** 2
    return np.asarray(selected, dtype=np.int64)
