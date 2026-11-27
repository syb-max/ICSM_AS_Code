"""
Dataset utilities for paper-aligned Q2 transmit antenna selection.

Preferred data format: dataset.h5 with keys:
    X_rel / x_rel / relation_matrix / tx_relation_matrix / tx_relation / X: [N,128,128] or [N,16384]
    hq_labels: [N,16] indices or [N,128] multi-hot
    lq_labels: [N,16] optional
    split_id: [N] optional, 0=train, 1=val, 2=test
    effective_snr_db: [N] optional
    hq_capacity: [N] optional
    G_tx_real, G_tx_imag: [N,Nf,128,128] optional for capacity evaluation

CSV fallback is supported only when relation matrices and labels are available.
The old scalar-regression CSV pair (128 input -> 1 gain) is intentionally not
accepted for subset-selection training, because it is not aligned with the paper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

RELATION_KEYS = ["X_rel", "x_rel", "relation_matrix", "tx_relation_matrix", "tx_relation", "X"]


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_key(h5: h5py.File, candidates: Iterable[str]) -> Optional[str]:
    for k in candidates:
        if k in h5:
            return k
    return None


def labels_to_indices(labels: np.ndarray, n_tx: int = 128, n_sel: int = 16) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D, got {labels.shape}")
    if labels.shape[1] == n_sel:
        return labels.astype(np.int64)
    if labels.shape[1] == n_tx:
        # Multi-hot/probability vector -> top n_sel indices.
        return np.argsort(-labels, axis=1)[:, :n_sel].astype(np.int64)
    raise ValueError(
        f"Unsupported label shape {labels.shape}. Expected [N,{n_sel}] indices or [N,{n_tx}] multi-hot."
    )


def indices_to_multihot(indices: np.ndarray, n_tx: int = 128) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    out = np.zeros((indices.shape[0], n_tx), dtype=np.float32)
    rows = np.arange(indices.shape[0])[:, None]
    out[rows, indices] = 1.0
    return out


def relation_target_from_x(x_rel: np.ndarray) -> np.ndarray:
    """Use the paper's transmit-side relation matrix as redundancy target."""
    rel = np.clip(np.abs(x_rel).astype(np.float32), 0.0, 1.0)
    n_tx = rel.shape[1]
    diag = np.arange(n_tx)
    rel[:, diag, diag] = 0.0
    return rel.astype(np.float32)


def offdiag_relation_mask(num_samples: int, n_tx: int = 128) -> np.ndarray:
    mask = np.ones((num_samples, n_tx, n_tx), dtype=np.float32)
    diag = np.arange(n_tx)
    mask[:, diag, diag] = 0.0
    return mask


class Standardizer:
    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer is not fitted.")
        return ((x - self.mean) / self.std).astype(np.float32)


class H5TASDataset(Dataset):
    def __init__(self, h5_path: str, indices: np.ndarray, n_tx: int = 128, n_sel: int = 16):
        self.h5_path = h5_path
        # h5py fancy indices must be increasing. Preserve shuffled sample order
        # by reading sorted unique rows then restoring it, for every dataset.
        self.indices = np.asarray(indices, dtype=np.int64)
        sorted_indices, inverse = np.unique(self.indices, return_inverse=True)

        def read_rows(dataset):
            return dataset[sorted_indices][inverse]
        self.n_tx = int(n_tx)
        self.n_sel = int(n_sel)

        with h5py.File(h5_path, "r") as f:
            rel_key = _find_key(f, RELATION_KEYS)
            if rel_key is None:
                raise KeyError(f"No relation matrix key found. Expected one of {RELATION_KEYS}")
            if "hq_labels" not in f:
                raise KeyError("dataset.h5 must contain hq_labels for paper-aligned training.")

            x = read_rows(f[rel_key])
            if x.ndim == 2 and x.shape[1] == n_tx * n_tx:
                x = x.reshape(-1, n_tx, n_tx)
            if x.ndim != 3 or x.shape[1:] != (n_tx, n_tx):
                raise ValueError(
                    f"Relation matrix must be [N,{n_tx},{n_tx}] or [N,{n_tx*n_tx}], got {x.shape}"
                )
            self.x_rel = x.astype(np.float32)
            self.relation_target = relation_target_from_x(self.x_rel)

            self.hq_labels = labels_to_indices(read_rows(f["hq_labels"]), n_tx, n_sel)
            if "lq_labels" in f:
                self.lq_labels = labels_to_indices(read_rows(f["lq_labels"]), n_tx, n_sel)
            else:
                self.lq_labels = self.hq_labels.copy()

            self.hq_utility_target = indices_to_multihot(self.hq_labels, n_tx)
            self.lq_utility_target = indices_to_multihot(self.lq_labels, n_tx)
            self.utility_target = self.hq_utility_target
            self.relation_mask = offdiag_relation_mask(len(self.indices), n_tx)

            if "G_tx_real" in f and "effective_snr_db" not in f:
                raise KeyError("Capacity evaluation requires explicit effective_snr_db; cannot assume 0 dB")
            self.effective_snr_db = read_rows(f["effective_snr_db"]).astype(np.float32) if "effective_snr_db" in f else np.full(len(self.indices), np.nan, dtype=np.float32)
            self.hq_capacity = read_rows(f["hq_capacity"]).astype(np.float32) if "hq_capacity" in f else np.full(len(self.indices), np.nan, dtype=np.float32)

            self.has_gram = "G_tx_real" in f
            if self.has_gram:
                self.G_real = read_rows(f["G_tx_real"]).astype(np.float32)
                self.G_imag = read_rows(f["G_tx_imag"]).astype(np.float32) if "G_tx_imag" in f else np.zeros_like(self.G_real)
            else:
                self.G_real = None
                self.G_imag = None

    def apply_standardizer(self, standardizer: Standardizer) -> None:
        self.x_rel = standardizer.transform(self.x_rel)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        item = {
            "x_rel": torch.from_numpy(self.x_rel[i]),
            "utility_target": torch.from_numpy(self.utility_target[i]),
            "hq_utility_target": torch.from_numpy(self.hq_utility_target[i]),
            "lq_utility_target": torch.from_numpy(self.lq_utility_target[i]),
            "relation_target": torch.from_numpy(self.relation_target[i]),
            "relation_mask": torch.from_numpy(self.relation_mask[i]),
            "hq_labels": torch.from_numpy(self.hq_labels[i]),
            "lq_labels": torch.from_numpy(self.lq_labels[i]),
            "effective_snr_db": torch.tensor(self.effective_snr_db[i], dtype=torch.float32),
            "hq_capacity": torch.tensor(self.hq_capacity[i], dtype=torch.float32),
        }
        if self.G_real is not None:
            item["G_tx_real"] = torch.from_numpy(self.G_real[i])
            item["G_tx_imag"] = torch.from_numpy(self.G_imag[i])
        return item


def split_indices_from_h5(h5_path: str, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        rel_key = _find_key(f, RELATION_KEYS)
        if rel_key is None:
            raise KeyError(f"No relation matrix key found. Expected one of {RELATION_KEYS}")
        n = int(f[rel_key].shape[0])
        if "split_id" in f:
            split_id = f["split_id"][:]
            return (
                np.where(split_id == 0)[0].astype(np.int64),
                np.where(split_id == 1)[0].astype(np.int64),
                np.where(split_id == 2)[0].astype(np.int64),
            )

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def build_dataloaders(
    h5_path: str,
    batch_size: int,
    seed: int = 42,
    n_tx: int = 128,
    n_sel: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Standardizer]:
    train_idx, val_idx, test_idx = split_indices_from_h5(h5_path, seed=seed)
    train_ds = H5TASDataset(h5_path, train_idx, n_tx=n_tx, n_sel=n_sel)
    val_ds = H5TASDataset(h5_path, val_idx, n_tx=n_tx, n_sel=n_sel)
    test_ds = H5TASDataset(h5_path, test_idx, n_tx=n_tx, n_sel=n_sel)

    standardizer = Standardizer()
    standardizer.fit(train_ds.x_rel)
    train_ds.apply_standardizer(standardizer)
    val_ds.apply_standardizer(standardizer)
    test_ds.apply_standardizer(standardizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, standardizer
