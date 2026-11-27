"""Offline capacity teachers for a fixed effective, frequency-resolved Gram.

The input is G[f] = H[f].conj().T @ H[f], with receiver noise already
whitened if required by the physical model. Capacity is equal-power log-det
in bit/s/Hz averaged across frequencies, not a water-filling optimization.

During a construction trajectory gamma = snr_linear / final_k MUST stay
fixed. If port activation changes loads, the effective channel, or power
normalization, construct a different physical teacher: the Schur update
here assumes that the same full Gram applies to every subset.

Each public entry validates finite Hermitian positive-semidefinite input
once, including O(F*N**3) eigenvalue validation. The greedy routine reuses
this validation. At a step with t selected ports, the Schur implementation
costs O(F*(t**3 + N*t**2)) for the solve and products (plus O(F*N)
diagonal/log work), and stores O(F*N*t) intermediates. It does not compute
N candidate determinants and does not yet use Cholesky rank-one updates.
Offline best-swap refinement costs O(P*K*(N-K)*F*K**3), in addition to
validation. These are offline label-generation costs.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


def _validated_gram(gram: np.ndarray) -> np.ndarray:
    raw = np.asarray(gram)
    if raw.ndim != 3 or raw.shape[1] != raw.shape[2] or min(raw.shape) < 1:
        raise ValueError("gram must have nonempty shape [F,N,N]")
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError("gram must contain numeric real or complex values")
    g = np.asarray(raw, dtype=np.complex128)
    if not np.isfinite(g).all():
        raise ValueError("gram must contain only finite values")
    # Permit input-storage roundoff, but not a materially non-Hermitian input.
    scale = np.maximum(1.0, np.max(np.abs(g), axis=(1, 2)))
    residual = np.max(np.abs(g - g.conj().transpose(0, 2, 1)), axis=(1, 2))
    if np.any(residual > 1e-6 * scale):
        raise ValueError("gram must be Hermitian at every frequency")
    g = 0.5 * (g + g.conj().transpose(0, 2, 1))
    eig_min = np.linalg.eigvalsh(g)[:, 0]
    # Generator Gram files store real/imaginary parts in float32. Their
    # rank-deficient H^H H matrices can have eigensolver residuals near 2e-5
    # of max(1, |G|); this matches the source validator's documented bound.
    if np.any(eig_min < -3e-5 * scale):
        raise ValueError("gram must be positive semidefinite at every frequency")
    # No PSD projection: tolerance only admits storage/eigensolver roundoff.
    return g


def _validated_snr(snr_linear: float) -> float:
    value = np.asarray(snr_linear)
    if value.ndim != 0 or np.iscomplexobj(value):
        raise ValueError("snr_linear must be a finite nonnegative real scalar")
    try:
        rho = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("snr_linear must be a finite nonnegative real scalar") from exc
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("snr_linear must be a finite nonnegative real scalar")
    return rho


def _validated_subset(subset: Iterable[int], n: int) -> np.ndarray:
    values = np.asarray(list(subset))
    if values.ndim != 1:
        raise ValueError("subset must be a one-dimensional sequence of indices")
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("subset indices must be integers")
    if np.any(values < 0) or np.any(values >= n):
        raise ValueError("subset contains an out-of-range antenna index")
    if np.unique(values).size != values.size:
        raise ValueError("subset contains duplicate antenna indices")
    return values.astype(np.int64, copy=False)


def _validated_k(final_k: int, n: int, selected_count: int = 0) -> int:
    if isinstance(final_k, (bool, np.bool_)) or not isinstance(final_k, (int, np.integer)):
        raise ValueError("final_k must be an integer in [1,N]")
    k = int(final_k)
    if k < 1 or k > n or selected_count > k:
        raise ValueError("final_k must be in [1,N] and at least the selected count")
    return k


def _capacity_validated(g: np.ndarray, rho: float, subset: np.ndarray, k: int) -> float:
    if subset.size == 0:
        return 0.0
    selected_g = g[:, subset, :][:, :, subset]
    matrix = np.eye(subset.size, dtype=np.complex128)[None] + (rho / k) * selected_g
    if not np.isfinite(matrix).all():
        raise ValueError("capacity matrix overflowed; rescale the physical Gram/SNR")
    try:
        chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("I + gamma*G_subset must be positive definite; check Gram/SNR") from exc
    logdet = 2.0 * np.log2(np.diagonal(chol, axis1=1, axis2=2).real).sum(axis=1)
    return float(logdet.mean())


def capacity(
    gram: np.ndarray,
    snr_linear: float,
    subset: Iterable[int],
    final_k: Optional[int] = None,
) -> float:
    """Return average log2 det(I + gamma G[S,S]); empty subset returns zero.

    gamma is rho/final_k when supplied, otherwise rho/len(subset). The
    latter convention is for evaluating a completed subset; it must not
    be used to generate intermediate fixed-power trajectory increments.
    """
    g = _validated_gram(gram)
    rho = _validated_snr(snr_linear)
    selected = _validated_subset(subset, g.shape[1])
    if final_k is None:
        if selected.size == 0:
            return 0.0
        k = selected.size
    else:
        k = _validated_k(final_k, g.shape[1], selected.size)
    return _capacity_validated(g, rho, selected, k)


def capacity_validated(
    gram: np.ndarray,
    snr_linear: float,
    subset: Iterable[int],
    final_k: int,
) -> float:
    """Evaluate a Gram already validated once by ``teacher_targets``.

    This avoids repeating the O(F*N^3) full-matrix PSD check for each model
    prediction and SNR replica. Shape, finite values, subset, rho and K are
    still checked; callers must ensure the same base Gram passed validation.
    """
    raw = np.asarray(gram)
    if raw.ndim != 3 or raw.shape[1] != raw.shape[2] or not np.isfinite(raw).all():
        raise ValueError("validated gram must be finite with shape [F,N,N]")
    g = np.asarray(raw, dtype=np.complex128)
    # Match _validated_gram: float32 storage can leave a tiny anti-Hermitian
    # residual, so evaluate the same Hermitian matrix used for teacher labels.
    g = 0.5 * (g + g.conj().transpose(0, 2, 1))
    rho = _validated_snr(snr_linear)
    selected = _validated_subset(subset, g.shape[1])
    k = _validated_k(final_k, g.shape[1], selected.size)
    return _capacity_validated(g, rho, selected, k)


def _marginal_validated(g: np.ndarray, rho: float, selected: np.ndarray, k: int) -> np.ndarray:
    n = g.shape[1]
    gamma = rho / k
    diag = np.diagonal(g, axis1=1, axis2=2).real
    if selected.size == 0:
        schur = 1.0 + gamma * diag
    else:
        g_ss = g[:, selected, :][:, :, selected]
        g_si = g[:, selected, :]
        g_is = g[:, :, selected]
        a = np.eye(selected.size, dtype=np.complex128)[None] + gamma * g_ss
        try:
            # Cholesky validates definiteness, solve reuses the dense batched
            # interface. Both costs remain O(F*t**3); no inverse is formed.
            np.linalg.cholesky(a)
            solved = np.linalg.solve(a, g_si)
        except np.linalg.LinAlgError as exc:
            raise ValueError("selected capacity matrix is not positive definite") from exc
        quadratic = np.einsum("fnt,ftn->fn", g_is, solved)
        if not np.allclose(quadratic.imag, 0.0, atol=1e-8 * max(1.0, float(np.abs(quadratic).max()))):
            raise ValueError("Schur complement acquired a non-negligible imaginary part")
        schur = 1.0 + gamma * diag - gamma * gamma * quadratic.real
    available = np.ones(n, dtype=bool)
    available[selected] = False
    if not np.isfinite(schur[:, available]).all() or np.any(schur[:, available] <= 0):
        raise ValueError("remaining-candidate Schur complements must be positive and finite")
    gains = np.zeros(n, dtype=np.float64)
    gains[available] = np.log2(schur[:, available]).mean(axis=0)
    return gains


def marginal_gains(
    gram: np.ndarray, snr_linear: float, selected: Iterable[int], final_k: int
) -> np.ndarray:
    """All remaining exact fixed-gamma increments; selected positions are 0.

    Uses Schur complements, with one small batched solve for all candidates.
    No determinant is evaluated separately for each remaining antenna.
    """
    g = _validated_gram(gram)
    rho = _validated_snr(snr_linear)
    indices = _validated_subset(selected, g.shape[1])
    k = _validated_k(final_k, g.shape[1], indices.size)
    return _marginal_validated(g, rho, indices, k)


def greedy_teacher(
    gram: np.ndarray, snr_linear: float, final_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic greedy order[K] and pre-selection gains[K,N].

    Ties use the lowest available antenna index. This is exact greedy for
    the specified fixed-Gram objective, not exhaustive global optimization.
    """
    g = _validated_gram(gram)
    rho = _validated_snr(snr_linear)
    k = _validated_k(final_k, g.shape[1])
    return _greedy_validated(g, rho, k)


def _greedy_validated(g: np.ndarray, rho: float, k: int) -> Tuple[np.ndarray, np.ndarray]:
    order = np.empty(k, dtype=np.int64)
    trace = np.empty((k, g.shape[1]), dtype=np.float64)
    for step in range(k):
        gains = _marginal_validated(g, rho, order[:step], k)
        trace[step] = gains
        scores = gains.copy()
        scores[order[:step]] = -np.inf
        order[step] = int(np.argmax(scores))
    return order, trace


def refine_subset(
    gram: np.ndarray,
    snr_linear: float,
    subset: Iterable[int],
    max_passes: int = 2,
    max_swap_evaluations: Optional[int] = None,
) -> np.ndarray:
    """Apply a fixed-budget best-improving one-port-swap search.

    A budget of ``None`` examines all K*(N-K) swaps. A finite budget samples
    evenly across the deterministic position/candidate ordering. The output
    is never worse than the input (up to arithmetic roundoff); this is not a
    global optimality certificate.
    """
    g = _validated_gram(gram)
    rho = _validated_snr(snr_linear)
    selected = _validated_subset(subset, g.shape[1]).copy()
    return _refine_validated(g, rho, selected, max_passes, max_swap_evaluations)


def _refine_validated(g, rho, selected, max_passes, max_swap_evaluations):
    if isinstance(max_passes, (bool, np.bool_)) or not isinstance(max_passes, (int, np.integer)) or max_passes < 0:
        raise ValueError("max_passes must be a nonnegative integer")
    if max_swap_evaluations is not None and (
        isinstance(max_swap_evaluations, (bool, np.bool_))
        or not isinstance(max_swap_evaluations, (int, np.integer))
        or max_swap_evaluations < 1
    ):
        raise ValueError("max_swap_evaluations must be a positive integer or None")
    k = selected.size
    if k == 0 or k == g.shape[1] or max_passes == 0:
        return selected
    value = _capacity_validated(g, rho, selected, k)
    for _ in range(int(max_passes)):
        remaining = np.setdiff1d(np.arange(g.shape[1]), selected)
        pairs = [(pos, int(candidate)) for pos in range(k) for candidate in remaining]
        if max_swap_evaluations is not None and len(pairs) > max_swap_evaluations:
            positions = np.linspace(0, len(pairs) - 1, int(max_swap_evaluations), dtype=np.int64)
            pairs = [pairs[i] for i in positions]
        best_value = value
        best = None
        for pos, candidate in pairs:
            trial = selected.copy()
            trial[pos] = candidate
            score = _capacity_validated(g, rho, trial, k)
            if score > best_value + 1e-12 * max(1.0, abs(best_value)):
                best_value, best = score, trial
        if best is None:
            break
        selected, value = best, best_value
    return selected


def teacher_targets(
    gram: np.ndarray,
    snr_values: Iterable[float],
    final_k: int,
    *,
    refine_passes: int = 1,
    swap_budget: Optional[int] = 32,
) -> list[dict]:
    """Build Greedy, Power-Top-K, and budget-refined HQ targets for one Gram.

    The expensive Hermitian/PSD validation is performed once and shared across
    the SNR replicas of this base channel.
    """
    g = _validated_gram(gram)
    k = _validated_k(final_k, g.shape[1])
    if isinstance(refine_passes, (bool, np.bool_)) or not isinstance(refine_passes, (int, np.integer)) or refine_passes < 0:
        raise ValueError("refine_passes must be a nonnegative integer")
    if swap_budget is not None and (
        isinstance(swap_budget, (bool, np.bool_)) or not isinstance(swap_budget, (int, np.integer)) or swap_budget < 1
    ):
        raise ValueError("swap_budget must be a positive integer or None")
    rho_values = [_validated_snr(value) for value in snr_values]
    power = np.maximum(np.diagonal(g, axis1=-2, axis2=-1).real, 0).mean(axis=0)
    lq = np.argsort(-power, kind="stable")[:k]
    denominator = np.sqrt((power[:, None] * power[None, :]).clip(1e-24))
    relation = np.clip(np.abs(g) / denominator[None, :, :], 0, 1).mean(0)
    result = []
    for rho in rho_values:
        order, gains = _greedy_validated(g, rho, k)
        greedy_capacity = _capacity_validated(g, rho, order, k)
        hq = _refine_validated(g, rho, order.copy(), refine_passes, swap_budget)
        hq_capacity = _capacity_validated(g, rho, hq, k)
        if hq_capacity + 1e-10 < greedy_capacity:
            raise ValueError("HQ refinement must not reduce the Greedy reference capacity")
        result.append(dict(
            X_rel=relation,
            teacher_order=order,
            greedy_order=order,
            teacher_marginals=gains,
            hq_labels=hq,
            lq_labels=lq,
            greedy_capacity=greedy_capacity,
            hq_capacity=hq_capacity,
            lq_capacity=_capacity_validated(g, rho, lq, k),
        ))
    return result
