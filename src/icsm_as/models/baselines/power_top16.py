"""Power-Top16 baseline."""
from __future__ import annotations
import numpy as np

def power_topk(gram: np.ndarray, k: int = 16) -> np.ndarray:
    g = np.asarray(gram)
    power = np.diagonal(g, axis1=-2, axis2=-1).real.mean(0)
    return np.argsort(-power, kind="stable")[:k].astype(np.int64)
