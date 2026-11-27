"""Fixed-termination multiport array operators; no built-in full-wave solver.

See docs/physics_model.md before using these operators in physical evaluation.
In particular, arbitrary normalized statistical channel Grams are NOT guaranteed
to be unit-current array responses, and loaded embedded patterns must not have
mutual coupling applied a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _complex(value, *, like: Optional[torch.Tensor] = None) -> torch.Tensor:
    value = torch.as_tensor(value, device=None if like is None else like.device)
    if like is not None:
        return value.to(like.dtype)
    if value.is_complex():
        return value
    return value.to(torch.complex128 if value.dtype == torch.float64 else torch.complex64)


def _square(value: torch.Tensor, name: str) -> None:
    if value.ndim < 2 or value.shape[-1] != value.shape[-2] or value.shape[-1] == 0:
        raise ValueError(f"{name} must end in nonempty [N,N], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")


def _psd(value: torch.Tensor, name: str, rtol: float, atol: float) -> None:
    scale = value.abs().amax(dim=(-2, -1)).clamp_min(1.0)
    tolerance = atol + rtol * scale
    if torch.any((value - value.mH).abs().amax(dim=(-2, -1)) > tolerance):
        raise ValueError(f"{name} must be Hermitian")
    if torch.any(torch.linalg.eigvalsh((value + value.mH) / 2).amin(dim=-1) < -tolerance):
        raise ValueError(f"{name} must be positive semidefinite")


def validate_impedance(
    z_tx: torch.Tensor,
    *,
    require_reciprocal: bool = True,
    radiation_resistance: Optional[torch.Tensor] = None,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> None:
    """Check finite, sampled-frequency passivity, reciprocity, and power matrices.

    Sampled PSD checks do not establish broadband causality or realizability.
    For reciprocal Z, Hermitian(Z) equals its entrywise real part. R_rad, if
    supplied, must satisfy 0 <= R_rad <= Hermitian(Z) as matrix inequalities.
    """
    z = _complex(z_tx)
    _square(z, "z_tx")
    if z.ndim not in (3, 4):
        raise ValueError("z_tx must be [F,N,N] or [B,F,N,N]")
    scale = z.abs().amax(dim=(-2, -1)).clamp_min(1.0)
    if require_reciprocal and torch.any(
        (z - z.transpose(-2, -1)).abs().amax(dim=(-2, -1)) > atol + rtol * scale
    ):
        raise ValueError("z_tx is not reciprocal: expected Z = Z^T (not Z = Z^H)")
    accepted = (z + z.mH) / 2
    _psd(accepted, "Hermitian(z_tx): passive accepted-power resistance", rtol, atol)
    if radiation_resistance is not None:
        radiation = _complex(radiation_resistance, like=z)
        _square(radiation, "radiation_resistance")
        if radiation.shape != z.shape:
            raise ValueError("radiation_resistance must have the same shape as z_tx")
        _psd(radiation, "radiation_resistance", rtol, atol)
        _psd(accepted - radiation, "ohmic-loss resistance", rtol, atol)


def _source_vector(source_impedance, z: torch.Tensor) -> torch.Tensor:
    source = _complex(source_impedance, like=z)
    if source.ndim == 0:
        source = source.expand(z.shape[-1])
    if source.ndim != 1 or source.shape[0] != z.shape[-1]:
        raise ValueError("source_impedance must be scalar or length N; fixed across samples/frequencies")
    if not torch.isfinite(source).all() or torch.any(source.real <= 0):
        raise ValueError("source_impedance must be finite and have strictly positive real parts")
    return source


def terminated_current_transfer(
    z_tx: torch.Tensor,
    source_impedance=50.0,
    reference_impedance: float = 50.0,
    *,
    load_policy: str = "fixed",
    validate: bool = True,
) -> torch.Tensor:
    """Return T = solve(Z + diag(Zs), diag(Z0 + Zs)).

    All physical elements remain present. Inactive sources are set to zero but
    retain exactly the same source impedances as active ports. If x is the
    reference-array current coordinate, V_s = diag(Z0 + Zs) x and I = T x.
    T becomes identity for the uncoupled reference array Z = Z0 I. It is not a
    radiation-efficiency normalization or an automatically power-normalized H.
    """
    if load_policy != "fixed":
        raise NotImplementedError("Subset-dependent/open/switched loads need a separate network solve per subset")
    z = _complex(z_tx)
    _square(z, "z_tx")
    if z.ndim not in (3, 4):
        raise ValueError("z_tx must be [F,N,N] or [B,F,N,N]")
    if not np.isfinite(reference_impedance) or reference_impedance <= 0:
        raise ValueError("reference_impedance must be a finite positive real resistance")
    if validate:
        validate_impedance(z)
    source = _source_vector(source_impedance, z)
    return torch.linalg.solve(z + torch.diag(source), torch.diag(source + reference_impedance))


def apply_tx_coupling_to_gram(
    gram: torch.Tensor,
    transfer: torch.Tensor,
    *,
    gram_basis: str = "unit_current",
) -> torch.Tensor:
    """Compute T^H G T without per-sample/per-port normalization.

    G must be H_I^H H_I for the full-array *unit port-current basis*, not a
    magnitude/normalized correlation matrix or an already loaded channel Gram.
    Accepted dimensions are G [B,F,N,N], T [F,N,N] or [B,F,N,N].
    """
    if gram_basis != "unit_current":
        raise ValueError("Coupling needs a unit_current Gram; loaded embedded channels must not be coupled twice")
    g = _complex(gram)
    t = _complex(transfer, like=g)
    _square(g, "gram")
    _square(t, "transfer")
    if g.ndim != 4 or t.ndim not in (3, 4):
        raise ValueError("Expected gram [B,F,N,N] and transfer [F,N,N] or [B,F,N,N]")
    if g.shape[-3:] != t.shape[-3:] or (t.ndim == 4 and t.shape[0] not in (1, g.shape[0])):
        raise ValueError("Gram and transfer frequency/port/batch dimensions do not match")
    return t.mH @ g @ t


def apply_gain_phase_errors(
    gram: torch.Tensor, gain_db=0.0, phase_rad=0.0
) -> torch.Tensor:
    """Source-side amplitude/phase imbalance: D^H G D, D=10^(g/20) exp(j phi).

    Parameters may be scalar, [N], [F,N], or [B,F,N]. They represent source
    errors after the nominal power budget; no corrective renormalization occurs.
    This is not a position-error model: Gram alone lacks angle/path information.
    """
    g = _complex(gram)
    _square(g, "gram")
    if g.ndim != 4:
        raise ValueError("gram must be [B,F,N,N]")
    real_dtype = g.real.dtype
    gain = torch.as_tensor(gain_db, device=g.device, dtype=real_dtype)
    phase = torch.as_tensor(phase_rad, device=g.device, dtype=real_dtype)
    target = g.shape[:-1]
    try:
        gain = torch.broadcast_to(gain, target)
        phase = torch.broadcast_to(phase, target)
    except RuntimeError as exc:
        raise ValueError("gain_db/phase_rad must broadcast to [B,F,N]") from exc
    if not torch.isfinite(gain).all() or not torch.isfinite(phase).all():
        raise ValueError("gain/phase parameters must be finite")
    diagonal = torch.pow(10.0, gain / 20) * torch.exp(1j * phase)
    if not torch.isfinite(diagonal).all():
        raise ValueError("gain values overflow the chosen dtype")
    return diagonal.conj().unsqueeze(-1) * g * diagonal.unsqueeze(-2)


@dataclass(frozen=True)
class ArrayNetworkData:
    """Imported physical/surrogate data with explicit excitation convention."""

    frequencies_hz: torch.Tensor
    z_tx: torch.Tensor
    source_impedance: torch.Tensor
    reference_impedance: float
    radiation_resistance: Optional[torch.Tensor]
    pattern_fields: Optional[torch.Tensor]
    pattern_basis: Optional[str]
    pattern_normalization: Optional[str]
    provenance: str


def load_array_npz(path: str | Path, *, device="cpu") -> ArrayNetworkData:
    """Load a trusted local NPZ without pickle; schema is in docs/physics_model.md.

    Required: frequencies_hz [F], z_tx [F,N,N], provenance (string).
    Optional complex arrays can instead be stored as *_real and *_imag pairs.
    Source impedance defaults to 50 ohms. Pattern fields require explicit basis
    and normalization text. This validates numbers, not source authenticity.
    """
    with np.load(path, allow_pickle=False) as data:
        def array(key: str, required: bool = False):
            if key in data:
                return np.array(data[key])
            if key + "_real" in data and key + "_imag" in data:
                return np.array(data[key + "_real"]) + 1j * np.array(data[key + "_imag"])
            if required:
                raise ValueError(f"Missing NPZ field {key}")
            return None

        def string(key: str, required: bool = False):
            value = array(key, required)
            if value is None:
                return None
            if value.ndim != 0 or value.dtype.kind not in "US":
                raise ValueError(f"{key} must be a scalar string")
            result = str(value.item())
            if not result.strip():
                raise ValueError(f"{key} must not be empty")
            return result

        freq = torch.as_tensor(array("frequencies_hz", True), device=device, dtype=torch.float64)
        if freq.ndim != 1 or freq.numel() == 0 or not torch.isfinite(freq).all() or torch.any(freq <= 0):
            raise ValueError("frequencies_hz must be a finite positive nonempty vector")
        if torch.any(freq[1:] <= freq[:-1]):
            raise ValueError("frequencies_hz must be strictly increasing")
        z = _complex(torch.as_tensor(array("z_tx", True), device=device))
        if z.ndim != 3 or z.shape[0] != freq.numel():
            raise ValueError("Imported z_tx must be [F,N,N] matching frequencies_hz")
        rad_np = array("radiation_resistance")
        radiation = None if rad_np is None else _complex(torch.as_tensor(rad_np, device=device), like=z)
        validate_impedance(z, radiation_resistance=radiation)
        src_np = array("source_impedance")
        source = _source_vector(50.0 if src_np is None else src_np, z)
        ref_np = array("reference_impedance")
        reference = 50.0 if ref_np is None else float(ref_np.item())
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("reference_impedance must be positive and finite")
        provenance = string("provenance", True)
        fields_np = array("pattern_fields")
        basis = string("pattern_basis")
        normalization = string("pattern_normalization")
        fields = None if fields_np is None else _complex(torch.as_tensor(fields_np, device=device), like=z)
        if fields is not None:
            if fields.ndim != 3 or fields.shape[0] != freq.numel() or fields.shape[-1] != z.shape[-1]:
                raise ValueError("pattern_fields must be [F,L,N]; L flattens direction/polarization bins")
            if fields.shape[1] == 0 or not torch.isfinite(fields).all():
                raise ValueError("pattern_fields must be finite and have nonempty angular/polarization bins")
            if basis not in ("unit_current", "fixed_loaded_source_voltage") or normalization is None:
                raise ValueError("pattern_fields require explicit valid pattern_basis and pattern_normalization")
            if basis == "fixed_loaded_source_voltage" and src_np is None:
                raise ValueError("Loaded embedded patterns require explicit source_impedance metadata")
        elif basis is not None or normalization is not None:
            raise ValueError("Pattern metadata supplied without pattern_fields")
    return ArrayNetworkData(freq, z, source, reference, radiation, fields, basis, normalization, provenance)


def channel_from_patterns(network: ArrayNetworkData, propagation: torch.Tensor) -> torch.Tensor:
    """Contract imported complex fields with known propagation: P [B,F,R,L].

    P must include angular quadrature/polarization/receiver/path-loss conventions
    consistent with pattern_fields [F,L,N]. Output is H [B,F,R,N] per x, where
    V_s=diag(Z0+Zs)x. No propagation, angles, or absolute calibration are invented.
    Loaded fields are defined per 1 V Thevenin source and are NOT multiplied by T.
    """
    if network.pattern_fields is None:
        raise ValueError("No pattern_fields were imported")
    fields = network.pattern_fields
    p = _complex(propagation, like=fields)
    if p.ndim != 4 or p.shape[1] != fields.shape[0] or p.shape[-1] != fields.shape[1]:
        raise ValueError("propagation must be [B,F,R,L] matching pattern_fields [F,L,N]")
    if not torch.isfinite(p).all():
        raise ValueError("propagation contains nonfinite values")
    h = p @ fields
    if network.pattern_basis == "unit_current":
        t = terminated_current_transfer(network.z_tx, network.source_impedance,
                                        network.reference_impedance, validate=False)
        return h @ t
    if network.pattern_basis == "fixed_loaded_source_voltage":
        return h * (network.reference_impedance + network.source_impedance)
    raise ValueError("Unsupported pattern basis")


def synthetic_passive_impedance(
    positions_m: torch.Tensor,
    frequencies_hz: torch.Tensor,
    coupling_strength: float = 0.25,
    resistance_ohm: float = 50.0,
    reactance_scale: float = 0.25,
) -> torch.Tensor:
    """DIAGNOSTIC ONLY: sampled-passive reciprocal synthetic Z [F,N,N].

    R = R0[(1-c)I+c exp(-distance/lambda)] is PSD for 0<=c<1.
    A symmetric synthetic reactive part is added. This is not a dipole formula,
    a full-wave solution, measurement, or a causal broadband antenna model.
    It cannot establish realistic mutual-coupling robustness for a paper.
    """
    pos = torch.as_tensor(positions_m, dtype=torch.float64)
    freq = torch.as_tensor(frequencies_hz, device=pos.device, dtype=pos.dtype)
    if pos.ndim != 2 or pos.shape[1] not in (1, 2, 3) or pos.shape[0] == 0 or not torch.isfinite(pos).all():
        raise ValueError("positions_m must be finite [N,1/2/3]")
    if freq.ndim != 1 or freq.numel() == 0 or not torch.isfinite(freq).all() or torch.any(freq <= 0):
        raise ValueError("frequencies_hz must be a finite positive vector")
    if not np.isfinite(coupling_strength) or not 0 <= coupling_strength < 1:
        raise ValueError("coupling_strength must lie in [0,1)")
    if not np.isfinite(resistance_ohm) or resistance_ohm <= 0 or not np.isfinite(reactance_scale):
        raise ValueError("resistance_ohm must be positive; reactance_scale must be finite")
    distance = torch.cdist(pos, pos)
    scaled = distance.unsqueeze(0) * freq[:, None, None] / 299792458.0
    eye = torch.eye(pos.shape[0], device=pos.device, dtype=pos.dtype)
    resistance = resistance_ohm * ((1 - coupling_strength) * eye + coupling_strength * torch.exp(-scaled))
    kd = 2 * torch.pi * scaled
    reactance = resistance_ohm * coupling_strength * reactance_scale * torch.sin(kd) / (1 + kd)
    return torch.complex(resistance, reactance)
