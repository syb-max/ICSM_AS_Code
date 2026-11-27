"""Offline conversion, fixed-load physics, and exact greedy-prefix supervision.

Never overwrites the input. All methods must be evaluated on the SAME output
Gram and reference capacities. The references are feasible, not global optima.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid
import h5py
import numpy as np
import torch
from .capacity_teacher import teacher_targets
from .physics import load_array_npz, terminated_current_transfer, apply_tx_coupling_to_gram


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_lazy_bank(src, source, dest, n_sel, gram_basis, refine_passes,
                       refine_swap_budget, seed, *, network=None,
                       frequencies_hz=None, positions_wavelength=None,
                       source_metadata=None, pilot_sites=None):
    if network is not None or gram_basis != "statistical_proxy":
        raise ValueError("Raw-bank lazy preparation supports statistical channel-only data; no physical Z/T is present")
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}")
    if not bool(src.attrs.get("complete", False)):
        raise ValueError("Input channel bank is incomplete")
    size, nf, nr, n = src["H_real"].shape
    if (nr, n) != (16, 128) or n_sel != 16 or src["H_imag"].shape != src["H_real"].shape:
        raise ValueError("Formal lazy training path requires H[F,16,128], N_t=128, and K=16")
    snrs = np.asarray(src["snr_grid_db"][:], dtype=np.float64)
    if snrs.shape != (5,) or not np.allclose(snrs, [0, 5, 10, 15, 20], rtol=0, atol=1e-6):
        raise ValueError("Formal bank must contain the five frozen reference SNR labels")
    site_ids = src["site_id"][:]
    split_ids = src["split_id"][:]
    for site in np.unique(site_ids):
        if len(np.unique(split_ids[site_ids == site])) != 1:
            raise ValueError(f"site_id {int(site)} crosses data splits")
    eligible = np.flatnonzero(np.isin(split_ids, [0, 1]))
    if not np.any(split_ids[eligible] == 0) or not np.any(split_ids[eligible] == 1):
        raise ValueError("Train and validation site splits must both be present")
    if pilot_sites is not None:
        if pilot_sites < 10:
            raise ValueError("pilot_sites must be at least 10")
        train_sites = np.unique(site_ids[split_ids == 0])
        if len(train_sites) < pilot_sites:
            raise ValueError("Not enough official training sites for the requested pilot")
        selected_sites = np.random.default_rng(seed).permutation(train_sites)[:pilot_sites]
        n_pilot_train = max(1, int(.9 * pilot_sites))
        pilot_train_sites = set(selected_sites[:n_pilot_train].tolist())
        pilot_val_sites = set(selected_sites[n_pilot_train:].tolist())
        pilot_rows = np.flatnonzero(np.isin(site_ids, selected_sites))
        split_ids = split_ids.copy()
        split_ids[pilot_rows] = np.asarray([
            0 if int(site_ids[row]) in pilot_train_sites else 1 for row in pilot_rows
        ], dtype=np.uint8)
        eligible = pilot_rows
    rows = len(eligible) * len(snrs)
    source_hash = file_sha256(source)
    metadata = dict(
        schema="q2_physics_lazy_bank_v1",
        storage_mode="lazy_raw_bank",
        source=str(source),
        source_file_sha256=source_hash,
        upstream_source_sha256=(source_metadata or {}).get("source_file_sha256"),
        source_metadata=source_metadata or {},
        n_tx=n,
        n_sel=n_sel,
        n_frequencies=nf,
        n_rx=nr,
        gram_basis="statistical_proxy",
        physics_status="statistical-channel-only; no measured/full-wave Z/T; identity-T proxy at the adapter",
        label_splits=["train", "val"],
        preparation_role="train_sites_internal_pilot" if pilot_sites is not None else "formal_train_validation",
        pilot_sites=pilot_sites,
        pilot_split_site_counts=(
            {"train": len(pilot_train_sites), "val": len(pilot_val_sites)} if pilot_sites is not None else None
        ),
        test_policy="test teacher targets are generated on demand only after selection manifest freeze",
        snr_grid_db=snrs.tolist(),
        snr_semantics="rho=10^(snr_reference_db/10) at 10m LOS; path loss and shadow remain in H",
        capacity="mean_f log2det(I_16+(rho/16) H_S[f] H_S[f]^H), complex128 Cholesky accumulation",
        greedy_labels="deterministic capacity-greedy trajectory",
        lq_labels="Power-Top-K by frequency-averaged Gram diagonal",
        hq_labels="Greedy initialization with fixed-budget one-swap candidate search; feasible, not global optimum",
        refine_passes=refine_passes,
        refine_swap_budget=refine_swap_budget,
        seed=seed,
        n_records=rows,
        n_base_train_val=len(eligible),
        n_train_val_decisions=rows,
        source_site_split_counts={str(s): int(np.sum(split_ids == s)) for s in (0, 1, 2)},
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(f"{dest.name}.{uuid.uuid4().hex}.partial")
    try:
        with h5py.File(temporary, "x") as out:
            out.attrs["complete"] = False
            out.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
            out.attrs["source_file_sha256"] = source_hash
            scalar_types = {
                "source_row_idx": np.int64, "sample_id": np.int64, "base_sample_id": np.int64,
                "site_id": np.int32, "split_id": np.uint8, "snr_reference_db": np.float32,
                "snr_reference_linear": np.float32, "effective_snr_db": np.float32,
                "los": np.uint8, "blocked": np.uint8, "shadow_gain_db": np.float32,
                "beta_db": np.float32,
            }
            for key, dtype in scalar_types.items():
                out.create_dataset(key, (rows,), dtype=dtype)
            specs = {
                "X_rel": (n, n), "teacher_marginals": (n_sel, n),
                "teacher_order": (n_sel,), "greedy_order": (n_sel,),
                "hq_labels": (n_sel,), "lq_labels": (n_sel,),
                "greedy_capacity": (), "hq_capacity": (), "lq_capacity": (),
            }
            for key, shape in specs.items():
                dtype = (
                    np.int64 if key.endswith("labels") or key.endswith("order")
                    else np.float64 if key == "teacher_marginals" or key.endswith("capacity")
                    else np.float32
                )
                out.create_dataset(
                    key,
                    (rows,) + shape,
                    dtype=dtype,
                    compression="lzf",
                    chunks=(1,) + shape,
                )
            target_row = 0
            for count, source_row in enumerate(eligible, start=1):
                h = (
                    src["H_real"][source_row].astype(np.float64)
                    + 1j * src["H_imag"][source_row].astype(np.float64)
                )
                gram = h.conj().transpose(0, 2, 1) @ h
                targets = teacher_targets(
                    gram,
                    10.0 ** (snrs / 10.0),
                    n_sel,
                    refine_passes=refine_passes,
                    swap_budget=refine_swap_budget,
                )
                for snr_db, target in zip(snrs, targets):
                    values = dict(
                        source_row_idx=source_row,
                        sample_id=int(src["sample_id"][source_row]),
                        base_sample_id=int(src["sample_id"][source_row]),
                        site_id=int(site_ids[source_row]),
                        split_id=int(split_ids[source_row]),
                        snr_reference_db=float(snr_db),
                        snr_reference_linear=10.0 ** (float(snr_db) / 10.0),
                        effective_snr_db=float(snr_db),
                        los=int(src["los"][source_row]),
                        blocked=int(src["blocked"][source_row]),
                        shadow_gain_db=float(src["shadow_gain_db"][source_row]),
                        beta_db=float(src["beta_db"][source_row]),
                        **target,
                    )
                    for key, value in values.items():
                        out[key][target_row] = value
                    target_row += 1
                if count % max(1, len(eligible) // 100) == 0:
                    print(f"Prepared {count}/{len(eligible)} train/val base channels", flush=True)
            out.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
            out.attrs["complete"] = True
        os.replace(temporary, dest)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return metadata


def prepare(input_path, output_path, n_sel, *, array_npz=None, gram_basis="statistical_proxy",
            frequencies_hz=None, positions_wavelength=None, refine_passes=1,
            refine_swap_budget=32, pilot_sites=None, seed=42):
    source, dest = Path(input_path).resolve(), Path(output_path).resolve()
    if source == dest or dest.exists():
        raise FileExistsError("Output must be a new file, distinct from input")
    network = None if array_npz is None else load_array_npz(array_npz)
    if gram_basis not in ("unit_current", "already_loaded", "statistical_proxy"):
        raise ValueError("Unknown Gram basis")
    transfer = None
    current_transfer = None
    with h5py.File(source, "r") as src:
        source_metadata = json.loads(src.attrs.get("metadata_json", "{}"))
        if "H_real" in src and {"H_imag", "snr_grid_db", "site_id", "split_id"}.issubset(src):
            return _prepare_lazy_bank(
                src, source, dest, n_sel, gram_basis, refine_passes, refine_swap_budget, seed,
                network=network, frequencies_hz=frequencies_hz,
                positions_wavelength=positions_wavelength, source_metadata=source_metadata,
                pilot_sites=pilot_sites,
            )
        for key in ("G_tx_real", "G_tx_imag", "effective_snr_db"):
            if key not in src:
                raise KeyError(f"Required input missing: {key}")
        size, nf, n, n2 = src["G_tx_real"].shape
        if n != n2 or not 1 <= n_sel <= n:
            raise ValueError("Invalid Gram or n_sel dimensions")
        if src["G_tx_imag"].shape != src["G_tx_real"].shape:
            raise ValueError("Real/imaginary Gram dimensions must agree")
        freq = frequencies_hz
        if freq is None and "frequencies_hz" in src:
            freq = src["frequencies_hz"][:]
        if freq is not None:
            freq = np.asarray(freq, dtype=np.float64)
            if freq.shape != (nf,) or not np.isfinite(freq).all() or np.any(freq <= 0) or np.any(np.diff(freq) <= 0):
                raise ValueError("frequencies_hz must be positive, increasing, and match F")
        coupling_edge = np.zeros((n, n), np.float32)
        if network is not None:
            if freq is None:
                raise ValueError("Supply actual channel frequencies; frequency index matching is insufficient")
            if network.z_tx.shape != (nf, n, n) or not np.allclose(freq, network.frequencies_hz.numpy(), rtol=0, atol=1):
                raise ValueError("Array network frequencies/ports do not match channel; interpolate externally with validation")
            src_z = network.source_impedance
            if not torch.allclose(src_z, src_z[0].expand_as(src_z)):
                raise ValueError("This equal-source-budget evaluator requires identical source impedances at all ports")
            z = network.z_tx
            denom = z.diagonal(dim1=-2, dim2=-1).abs().clamp_min(1e-12)
            edge = (z.abs()/(denom.unsqueeze(-1)*denom.unsqueeze(-2)).sqrt()).mean(0)
            coupling_edge = edge.numpy().astype(np.float32)
            np.fill_diagonal(coupling_edge, 0)
            current_transfer = terminated_current_transfer(z, network.source_impedance, network.reference_impedance)
            if gram_basis != "already_loaded":
                transfer = current_transfer
        if positions_wavelength is not None:
            positions_wavelength = np.asarray(positions_wavelength, np.float32)
            if positions_wavelength.shape != (n, 3) or not np.isfinite(positions_wavelength).all():
                raise ValueError("positions_wavelength must be finite [N,3], using common center wavelength")
        valid = np.arange(size)
        if "error_flag" in src and np.any(src["error_flag"][:] != 0):
            raise ValueError("Input contains failed records; refusing to silently drop samples")
        if len(valid) < 3:
            raise ValueError("At least three valid samples are required")
        if "split_id" in src:
            split = src["split_id"][:][valid]
            if not np.isin(split, [0, 1, 2]).all() or any(not np.any(split == s) for s in (0, 1, 2)):
                raise ValueError("Input split_id must contain nonempty train=0, val=1, test=2")
            if "site_id" in src:
                sites = src["site_id"][:][valid]
                for site in np.unique(sites):
                    if len(np.unique(split[sites == site])) != 1:
                        raise ValueError(f"site_id {int(site)} crosses data splits")
        else:
            if "site_id" not in src:
                raise ValueError("Input must contain site_id and split_id for group-safe preparation")
            sites = src["site_id"][:][valid]
            assigned = {}
            unique_sites = np.unique(sites)
            site_order = np.random.default_rng(seed).permutation(unique_sites)
            n_train = min(len(site_order) - 2, max(1, int(.8 * len(site_order))))
            n_val = min(len(site_order) - n_train - 1, max(1, int(.1 * len(site_order))))
            train_sites = set(site_order[:n_train])
            val_sites = set(site_order[n_train:n_train + n_val])
            for site in unique_sites:
                assigned[int(site)] = 0 if site in train_sites else 1 if site in val_sites else 2
            split = np.asarray([assigned[int(site)] for site in sites], dtype=np.uint8)
        provenance = "No imported array network; channel-only study"
        if network is not None:
            provenance = network.provenance
        metadata = dict(schema="q2_physics_v1", source=str(source), n_tx=n, n_sel=n_sel,
            n_frequencies=nf, gram_basis=gram_basis, array_source=provenance,
            array_npz=None if array_npz is None else str(Path(array_npz).resolve()),
            physics_status=(
                "statistical-channel-only; no measured/full-wave Z/T; identity-T proxy at the adapter"
                if network is None and gram_basis == "statistical_proxy" else
                "statistical circuit sensitivity only; not measured/full-wave coupling"
                if network is not None and gram_basis == "statistical_proxy" else
                "user-supplied basis; external EM calibration still requires verification"
            ),
            load_policy="fixed", power_convention="fixed equal source-coordinate budget; no per-sample renormalization",
            teacher="exact greedy-prefix marginal gains with fixed rho/n_sel",
            greedy_labels="deterministic capacity-greedy trajectory",
            lq_labels="Power-Top-K by frequency-averaged Gram diagonal",
            hq_labels="capacity-greedy initialization with fixed-budget one-swap candidate search; feasible, not global optimum",
            refine_passes=refine_passes, refine_swap_budget=refine_swap_budget, seed=seed,
            split_granularity="site_id" if "site_id" in src else "sample_id-unverified",
            snr_semantics="snr_reference_db is rho at the 10 m LOS reference; effective_snr_db is a compatibility alias",
            source_file_sha256=file_sha256(source),
            upstream_source_sha256=source_metadata.get("source_file_sha256"),
            upstream_source_metadata=source_metadata.get("source_metadata", {}),
            snr_grid_db=(np.asarray(src["snr_grid_db"][:], dtype=float).tolist()
                         if "snr_grid_db" in src else source_metadata.get("snr_grid_db")))
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_name(f"{dest.name}.{uuid.uuid4().hex}.partial")
        with h5py.File(temporary, "x") as out:
            out.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
            out.attrs["complete"] = False
            out.attrs["source_file_sha256"] = metadata["source_file_sha256"]
            out["split_id"] = split
            metadata_keys = ("sample_id", "base_sample_id", "site_id", "los", "blocked",
                             "snr_reference_db", "snr_reference_linear", "shadow_gain_db", "beta_db")
            for key in metadata_keys:
                if key == "sample_id":
                    values = src[key][:][valid] if key in src else valid
                elif key == "base_sample_id":
                    values = src[key][:][valid] if key in src else (src["sample_id"][:][valid] if "sample_id" in src else valid)
                elif key == "snr_reference_db":
                    values = src[key][:][valid] if key in src else src["effective_snr_db"][:][valid]
                elif key == "snr_reference_linear":
                    values = src[key][:][valid] if key in src else np.power(10.0, src["effective_snr_db"][:][valid] / 10.0)
                elif key in src:
                    values = src[key][:][valid]
                elif key == "site_id":
                    values = np.full(len(valid), -1, np.int32)
                elif key in ("los", "blocked"):
                    values = np.full(len(valid), 0, np.uint8)
                else:
                    values = np.full(len(valid), np.nan, np.float32)
                out[key] = values
            if "snr_reference_db" in src and "effective_snr_db" in src:
                if not np.allclose(src["snr_reference_db"][:][valid], src["effective_snr_db"][:][valid], rtol=0, atol=1e-6):
                    raise ValueError("effective_snr_db must remain an alias of snr_reference_db")
            out["effective_snr_db"] = out["snr_reference_db"][:]
            out["coupling_edge"] = coupling_edge
            if current_transfer is not None:
                # Keep all physical ports, including unexcited port currents.
                out["T_tx_real"] = current_transfer.real.numpy().astype(np.float32)
                out["T_tx_imag"] = current_transfer.imag.numpy().astype(np.float32)
                induced = current_transfer-torch.diag_embed(current_transfer.diagonal(dim1=-2,dim2=-1))
                kernel = induced.mH@induced
                kd = kernel.diagonal(dim1=-2,dim2=-1).real.clamp_min(0)
                kernel = (kernel/(kd.unsqueeze(-1)*kd.unsqueeze(-2)).clamp_min(1e-24).sqrt()).mean(0)
                out["current_kernel_real"] = kernel.real.numpy().astype(np.float32)
                out["current_kernel_imag"] = kernel.imag.numpy().astype(np.float32)
                resistance = (network.z_tx+network.z_tx.mH)/2
                accepted = .5*(current_transfer.mH@resistance@current_transfer).diagonal(dim1=-2,dim2=-1).real
                out["accepted_power_per_source"] = accepted.clamp_min(0).numpy().astype(np.float32)
            if freq is not None:
                out["frequencies_hz"] = freq
            if positions_wavelength is not None:
                out["positions_wavelength"] = positions_wavelength
            specs = {"G_tx_real": (nf,n,n), "G_tx_imag": (nf,n,n), "X_rel": (n,n),
                     "teacher_marginals": (n_sel,n), "teacher_order": (n_sel,),
                     "greedy_order": (n_sel,), "hq_labels": (n_sel,), "lq_labels": (n_sel,),
                     "greedy_capacity": (), "hq_capacity": (), "lq_capacity": ()}
            for key, shape in specs.items():
                dtype = (
                    np.int64 if key.endswith("labels") or key.endswith("order")
                    else np.float64 if key == "teacher_marginals" or key.endswith("capacity")
                    else np.float32
                )
                out.create_dataset(key, (len(valid),)+shape, dtype=dtype, compression="lzf")
            source_snr = src["snr_reference_db"][:] if "snr_reference_db" in src else src["effective_snr_db"][:]
            base_ids = src["base_sample_id"][:][valid] if "base_sample_id" in src else (
                src["sample_id"][:][valid] if "sample_id" in src else valid
            )
            groups = {}
            for row, idx in enumerate(valid):
                groups.setdefault(int(base_ids[row]), []).append((row, int(idx)))
            metadata["unique_base_samples"] = len(groups)
            for group_number, pairs in enumerate(groups.values(), start=1):
                first_idx = pairs[0][1]
                gram = src["G_tx_real"][first_idx].astype(np.float64) + 1j * src["G_tx_imag"][first_idx].astype(np.float64)
                for _, idx in pairs[1:]:
                    other = src["G_tx_real"][idx].astype(np.float64) + 1j * src["G_tx_imag"][idx].astype(np.float64)
                    if not np.allclose(other, gram, rtol=1e-6, atol=1e-9):
                        raise ValueError("A base_sample_id has different Grams across SNR replicas")
                if len({round(float(source_snr[idx]), 7) for _, idx in pairs}) != len(pairs):
                    raise ValueError("A base_sample_id contains duplicate SNR labels")
                if len({int(split[row]) for row, _ in pairs}) != 1:
                    raise ValueError("A base_sample_id crosses data splits")
                if "site_id" in src and len({int(src["site_id"][idx]) for _, idx in pairs}) != 1:
                    raise ValueError("A base_sample_id maps to multiple site_id values")
                expected_grid = metadata.get("snr_grid_db")
                if expected_grid is not None:
                    observed_grid = sorted(float(source_snr[idx]) for _, idx in pairs)
                    if len(observed_grid) != len(expected_grid) or not np.allclose(
                        observed_grid, sorted(expected_grid), rtol=0, atol=1e-6
                    ):
                        raise ValueError("A base_sample_id is missing or duplicates an expected SNR label")
                if transfer is not None:
                    # This explicitly remains a proxy unless the source Z/T is calibrated.
                    gram = apply_tx_coupling_to_gram(torch.from_numpy(gram)[None], transfer,
                                                     gram_basis="unit_current")[0].numpy()
                targets = teacher_targets(
                    gram,
                    [float(source_snr[idx]) for _, idx in pairs],
                    n_sel,
                    refine_passes=refine_passes,
                    swap_budget=refine_swap_budget,
                )
                for (row, idx), target in zip(pairs, targets):
                    out["G_tx_real"][row] = gram.real
                    out["G_tx_imag"][row] = gram.imag
                    for key, value in target.items():
                        out[key][row] = value
                    if (row + 1) % 100 == 0:
                        print(f"Prepared {row+1}/{len(valid)}", flush=True)
            out.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
            out.attrs["complete"] = True
        os.replace(temporary, dest)
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_h5", required=True)
    p.add_argument("--output_h5", required=True)
    p.add_argument("--n_sel", type=int, default=16)
    p.add_argument("--array_npz")
    p.add_argument("--gram_basis", choices=["unit_current", "already_loaded", "statistical_proxy"], default="statistical_proxy")
    p.add_argument("--frequencies_npy")
    p.add_argument("--positions_wavelength_npy")
    p.add_argument("--refine_passes", type=int, default=1)
    p.add_argument("--refine_swap_budget", type=int, default=32)
    p.add_argument("--pilot_sites", type=int, help="group-held-out pilot using only official train sites")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    print(json.dumps(prepare(a.input_h5, a.output_h5, a.n_sel, array_npz=a.array_npz,
        gram_basis=a.gram_basis, frequencies_hz=None if a.frequencies_npy is None else np.load(a.frequencies_npy, allow_pickle=False),
        positions_wavelength=None if a.positions_wavelength_npy is None else np.load(a.positions_wavelength_npy, allow_pickle=False),
        refine_passes=a.refine_passes, refine_swap_budget=a.refine_swap_budget,
        pilot_sites=a.pilot_sites, seed=a.seed), indent=2, ensure_ascii=False))
