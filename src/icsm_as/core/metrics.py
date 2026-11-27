"""Metrics for fixed-receiver 128-to-16 transmit antenna selection."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def hit_ratio_at_k(pred_idx: torch.Tensor, gt_idx: torch.Tensor) -> float:
    pred = pred_idx.detach().cpu().numpy()
    gt = gt_idx.detach().cpu().numpy()
    k = gt.shape[1]
    scores = []
    for p, g in zip(pred, gt):
        scores.append(len(set(p.tolist()).intersection(set(g.tolist()))) / k)
    return float(np.mean(scores))


def stable_capacity_from_gram_subset(Gf: np.ndarray, snr_linear: float, subset: np.ndarray) -> float:
    subset = np.asarray(subset, dtype=np.int64)
    n_stream = max(int(len(subset)), 1)
    total = 0.0
    for f in range(Gf.shape[0]):
        Gs = Gf[f][np.ix_(subset, subset)]
        eye = np.eye(len(subset), dtype=Gs.dtype)
        mat = eye + (snr_linear / n_stream) * Gs
        sign, logdet = np.linalg.slogdet(mat)
        if sign <= 0:
            eigvals = np.linalg.eigvalsh((mat + mat.conj().T) / 2.0)
            eigvals = np.clip(eigvals.real, 1e-12, None)
            total += float(np.sum(np.log2(eigvals)))
        else:
            total += float(logdet / np.log(2.0))
    return total / Gf.shape[0]


def batch_capacity_from_gram(batch: Dict[str, torch.Tensor], subset_idx: torch.Tensor) -> Optional[np.ndarray]:
    if "G_tx_real" not in batch:
        return None
    G_real = batch["G_tx_real"].detach().cpu().numpy()
    G_imag = batch["G_tx_imag"].detach().cpu().numpy()
    snr_db = batch["effective_snr_db"].detach().cpu().numpy()
    subsets = subset_idx.detach().cpu().numpy()
    caps = []
    for i in range(G_real.shape[0]):
        Gf = G_real[i] + 1j * G_imag[i]
        snr_linear = 10.0 ** (float(snr_db[i]) / 10.0)
        caps.append(stable_capacity_from_gram_subset(Gf, snr_linear, subsets[i]))
    return np.asarray(caps, dtype=np.float64)
