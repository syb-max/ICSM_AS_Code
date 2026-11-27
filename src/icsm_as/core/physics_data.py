"""Worker-local lazy HDF5 reader for prepared physics studies."""
from __future__ import annotations

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .capacity_teacher import teacher_targets


class PhysicsDataset(Dataset):
    _ROW_KEYS = (
        "G_tx_real", "G_tx_imag", "effective_snr_db", "snr_reference_db",
        "snr_reference_linear", "hq_capacity", "greedy_capacity", "lq_capacity",
        "hq_labels", "lq_labels", "teacher_order", "greedy_order",
        "teacher_marginals", "sample_id", "base_sample_id", "site_id", "split_id",
        "los", "blocked", "shadow_gain_db", "beta_db",
    )
    _STATIC_KEYS = (
        "coupling_edge", "positions_wavelength", "T_tx_real", "T_tx_imag",
        "current_kernel_real", "current_kernel_imag", "accepted_power_per_source",
    )

    def __init__(self, path, split=0, require_teacher=True):
        self.path = str(path)
        self.split = int(split)
        self._file = None
        self._bank_file = None
        self._pid = None
        self._static_cache = None
        self._test_base_ids = None
        self._target_cache = {}
        self._last_gram_key = None
        self._last_gram_data = None
        self._last_teacher_key = None
        self._last_teacher_data = None
        with h5py.File(self.path, "r") as f:
            if not bool(f.attrs.get("complete", False)):
                raise ValueError("Use a completed prepare_physics_data.py output")
            self.metadata = json.loads(f.attrs["metadata_json"])
            self.n_tx = int(self.metadata["n_tx"])
            self.n_sel = int(self.metadata["n_sel"])
            self._lazy_bank = self.metadata.get("storage_mode") == "lazy_raw_bank"
            self._snrs = np.asarray(self.metadata.get("snr_grid_db", []), dtype=np.float32)
            if self._lazy_bank and self.split == 2:
                with h5py.File(self.metadata["source"], "r") as bank:
                    self._test_base_ids = np.flatnonzero(bank["split_id"][:] == 2)
                self.indices = np.arange(len(self._test_base_ids) * len(self._snrs), dtype=np.int64)
            else:
                self.indices = np.flatnonzero(f["split_id"][:] == self.split)
            if not len(self.indices):
                raise ValueError(f"Empty split {self.split}")
            if require_teacher and not {"teacher_order", "teacher_marginals", "hq_labels", "lq_labels"}.issubset(f):
                raise ValueError("Required teacher targets are missing")
            if "X_rel" not in f:
                if not self._lazy_bank:
                    raise ValueError("Prepared relation features are missing")
            self._has_frequency = "frequencies_hz" in f

    def __len__(self):
        return len(self.indices)

    def _handle(self):
        pid = os.getpid()
        if self._file is not None and self._pid != pid:
            self.close()
        if self._file is None:
            self._file = h5py.File(self.path, "r")
            self._pid = pid
            self._static_cache = {}
            for key in self._STATIC_KEYS:
                if key in self._file:
                    self._static_cache[key] = torch.from_numpy(
                        np.asarray(self._file[key][:], dtype=np.float32).copy()
                    )
            if not self._lazy_bank and self._has_frequency:
                freq = np.asarray(self._file["frequencies_hz"][:], dtype=np.float64)
                self._static_cache["frequency_coord"] = torch.from_numpy(
                    ((freq - freq.mean()) / 1e9).astype(np.float32)
                )
        if self._lazy_bank and self._bank_file is None:
            self._bank_file = h5py.File(self.metadata["source"], "r")
            self._static_cache["positions_wavelength"] = torch.from_numpy(
                np.asarray(self._bank_file["positions_wavelength"][:], dtype=np.float32).copy()
            )
            freq = np.asarray(self._bank_file["frequencies_hz"][:], dtype=np.float64)
            self._static_cache["frequency_coord"] = torch.from_numpy(
                ((freq - freq.mean()) / 1e9).astype(np.float32)
            )
        return self._file

    def close(self):
        if self._file is not None:
            self._file.close()
        if self._bank_file is not None:
            self._bank_file.close()
        self._file = None
        self._bank_file = None
        self._pid = None
        self._static_cache = None
        self._target_cache = {}
        self._last_gram_key = None
        self._last_gram_data = None
        self._last_teacher_key = None
        self._last_teacher_data = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_bank_file"] = None
        state["_pid"] = None
        state["_static_cache"] = None
        state["_target_cache"] = {}
        state["_last_gram_key"] = None
        state["_last_gram_data"] = None
        state["_last_teacher_key"] = None
        state["_last_teacher_data"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def evaluation_metadata(self, i):
        """Read decision IDs, SNR and static features only. No Gram or teacher targets."""
        idx = int(self.indices[i])
        f = self._handle()
        keys = ("sample_id", "base_sample_id", "site_id", "snr_reference_db",
                "snr_reference_linear", "effective_snr_db")
        if self._lazy_bank:
            if self.split == 2:
                base_offset, snr_idx = divmod(idx, len(self._snrs))
                source_idx = int(self._test_base_ids[base_offset])
            else:
                source_idx = int(f["source_row_idx"][idx])
                candidates = np.flatnonzero(np.isclose(self._snrs, f["snr_reference_db"][idx]))
                if len(candidates) != 1:
                    raise ValueError("Decision SNR does not match the frozen SNR grid")
                snr_idx = int(candidates[0])
            snr = float(self._snrs[snr_idx])
            base_id = int(self._bank_file["sample_id"][source_idx]) if "sample_id" in self._bank_file else source_idx
            row = dict(sample_id=source_idx*len(self._snrs)+snr_idx,
                       base_sample_id=base_id, site_id=int(self._bank_file["site_id"][source_idx]),
                       snr_reference_db=snr, snr_reference_linear=10.0**(snr/10.0), effective_snr_db=snr)
            batch = {key: torch.as_tensor(row[key]) for key in keys}
        else:
            batch = {key: torch.as_tensor(np.asarray(f[key][idx]).copy()) for key in keys}
        batch.update(self._static_cache)
        return batch

    def __getitem__(self, i):
        idx = int(self.indices[i])
        f = self._handle()
        if self._lazy_bank:
            if self.split == 2:
                base_offset, snr_idx = divmod(idx, len(self._snrs))
                source_idx = int(self._test_base_ids[base_offset])
                snr_db = float(self._snrs[snr_idx])
                sample_id = source_idx * len(self._snrs) + snr_idx
                row = dict(
                    source_row_idx=source_idx, sample_id=sample_id, base_sample_id=source_idx,
                    site_id=int(self._bank_file["site_id"][source_idx]), split_id=2,
                    snr_reference_db=snr_db, snr_reference_linear=10.0 ** (snr_db / 10.0),
                    effective_snr_db=snr_db, los=int(self._bank_file["los"][source_idx]),
                    blocked=int(self._bank_file["blocked"][source_idx]),
                    shadow_gain_db=float(self._bank_file["shadow_gain_db"][source_idx]),
                    beta_db=float(self._bank_file["beta_db"][source_idx]),
                )
            else:
                row = {key: f[key][idx].item() for key in (
                    "source_row_idx", "sample_id", "base_sample_id", "site_id", "split_id",
                    "snr_reference_db", "snr_reference_linear", "effective_snr_db", "los",
                    "blocked", "shadow_gain_db", "beta_db",
                )}
                source_idx = int(row["source_row_idx"])
            if self._last_gram_key != source_idx:
                h = (
                    np.asarray(self._bank_file["H_real"][source_idx], dtype=np.float64)
                    + 1j * np.asarray(self._bank_file["H_imag"][source_idx], dtype=np.float64)
                )
                gram = h.conj().transpose(0, 2, 1) @ h
                power = np.maximum(np.diagonal(gram, axis1=1, axis2=2).real, 0).mean(axis=0)
                denominator = np.sqrt((power[:, None] * power[None, :]).clip(1e-24))
                relation = np.clip(np.abs(gram) / denominator[None, :, :], 0, 1).mean(0).astype(np.float32)
                self._last_gram_key = source_idx
                self._last_gram_data = (
                    gram.real.astype(np.float32), gram.imag.astype(np.float32), relation
                )
            gram_real, gram_imag, relation = self._last_gram_data
            gram = gram_real.astype(np.float64) + 1j * gram_imag.astype(np.float64)
            batch = {
                key: torch.as_tensor(row[key]) for key in (
                    "sample_id", "base_sample_id", "site_id", "split_id", "snr_reference_db",
                    "snr_reference_linear", "effective_snr_db", "los", "blocked", "shadow_gain_db", "beta_db",
                )
            }
            batch["G_tx_real"] = torch.from_numpy(gram_real.copy())
            batch["G_tx_imag"] = torch.from_numpy(gram_imag.copy())
            batch["X_rel"] = torch.from_numpy(relation)
            relation_target = relation.copy()
            np.fill_diagonal(relation_target, 0)
            batch["relation_target"] = torch.from_numpy(relation_target)
            if self.split == 2:
                if self._last_teacher_key != source_idx:
                    rho_values = 10.0 ** (self._snrs.astype(np.float64) / 10.0)
                    self._last_teacher_data = teacher_targets(
                        gram, rho_values, self.n_sel,
                        refine_passes=int(self.metadata["refine_passes"]),
                        swap_budget=int(self.metadata["refine_swap_budget"]),
                    )
                    self._last_teacher_key = source_idx
                target = self._last_teacher_data[snr_idx]
                for key, value in target.items():
                    if key != "X_rel":
                        batch[key] = torch.as_tensor(np.asarray(value).copy())
            else:
                for key in (
                    "hq_capacity", "greedy_capacity", "lq_capacity", "hq_labels", "lq_labels",
                    "teacher_order", "greedy_order", "teacher_marginals",
                ):
                    batch[key] = torch.as_tensor(np.asarray(f[key][idx]).copy())
        else:
            batch = {
                key: torch.as_tensor(np.asarray(f[key][idx]).copy())
                for key in self._ROW_KEYS if key in f
            }
            relation = np.asarray(f["X_rel"][idx], dtype=np.float32).copy()
            np.fill_diagonal(relation, 0)
            batch["X_rel"] = torch.from_numpy(relation.copy())
            batch["relation_target"] = torch.from_numpy(relation)
        batch.update(self._static_cache)
        for prefix in ("hq", "lq"):
            target = torch.zeros(self.n_tx)
            target[batch[prefix + "_labels"].long()] = 1
            batch[prefix + "_utility_target"] = target
        batch["utility_target"] = batch["hq_utility_target"]
        return batch
