"""HDF5 schema checks. This module validates supplied data; it does not fabricate labels."""
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import json

REQUIRED_KEYS = (
    "G_tx_real", "G_tx_imag", "effective_snr_db", "snr_reference_db", "snr_reference_linear",
    "hq_labels", "lq_labels", "teacher_order", "teacher_marginals",
    "hq_capacity", "greedy_capacity", "split_id", "sample_id", "site_id",
)
OPTIONAL_PHYSICAL_KEYS = (
    "T_tx_real", "T_tx_imag", "Z_tx_real", "Z_tx_imag",
    "positions_wavelength", "current_kernel_real", "current_kernel_imag",
)

def validate_prepared_h5(path: str | Path, *, require_physical: bool = False) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        missing = [k for k in REQUIRED_KEYS if k not in f]
        if missing:
            raise ValueError(f"Missing required datasets: {missing}")
        shape = f["G_tx_real"].shape
        if len(shape) != 4 or shape[-1] != shape[-2]:
            raise ValueError("G_tx_real must have shape [N,F,Nt,Nt]")
        if f["G_tx_imag"].shape != shape:
            raise ValueError("G_tx_imag must match G_tx_real")
        n, nf, nt, _ = shape
        metadata = json.loads(f.attrs.get("metadata_json", "{}"))
        lazy_bank = metadata.get("storage_mode") == "lazy_raw_bank"
        if "X_rel" not in f and not lazy_bank:
            raise ValueError("Prepared HDF5 data must include X_rel [N,Nt,Nt]")
        if "X_rel" in f:
            if f["X_rel"].shape != (n, nt, nt):
                raise ValueError("X_rel must have shape [N,Nt,Nt]")
            if not np.isfinite(f["X_rel"][0]).all():
                raise ValueError("X_rel contains non-finite values")
        for key in ("hq_labels", "lq_labels", "teacher_order"):
            if f[key].shape[0] != n:
                raise ValueError(f"{key} sample count mismatch")
        if f["teacher_marginals"].shape[:2] != f["teacher_order"].shape:
            raise ValueError("teacher_marginals must start with [N,K]")
        split = np.asarray(f["split_id"][:])
        if split.shape != (n,) or not set(np.unique(split)).issubset({0,1,2}):
            raise ValueError("split_id must contain only 0=train, 1=val, 2=test")
        if "site_id" in f:
            site = np.asarray(f["site_id"][:])
            for sid in np.unique(site):
                if len(np.unique(split[site == sid])) != 1:
                    raise ValueError(f"site_id {sid} crosses data splits")
        physical_present = any(k in f for k in ("T_tx_real", "Z_tx_real"))
        if require_physical and not ("T_tx_real" in f or "Z_tx_real" in f):
            raise ValueError("Full-wave/physical PCR requires complex T or calibrated complex Z data")
        return {"samples": int(n), "frequencies": int(nf), "n_tx": int(nt),
                "n_sel": int(f["teacher_order"].shape[1]),
                "physical_descriptors_present": bool(physical_present),
                "split_counts": {str(k): int((split == k).sum()) for k in (0,1,2)}}
