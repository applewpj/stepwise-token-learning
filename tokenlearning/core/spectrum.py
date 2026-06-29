from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tokenlearning.core.sigmoid import learning_pulse
from tokenlearning.types import TokenEventFit


@dataclass
class Spectrum:
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    counts: np.ndarray
    mass: np.ndarray
    density: np.ndarray
    pos_mass: np.ndarray | None = None
    neg_mass: np.ndarray | None = None
    diff_mass: np.ndarray | None = None


@dataclass
class BinnedPulses:
    offset_grid: np.ndarray
    tau_bin_edges: np.ndarray
    tau_bin_centers: np.ndarray
    pulse_by_bin: np.ndarray
    pulse_mean: np.ndarray
    counts: np.ndarray
    aligned_to_tau: bool = True


@dataclass
class SpectrumReconstruction:
    x_dense: np.ndarray
    x_rate_dense: np.ndarray
    u_grid: np.ndarray
    positive_profile: np.ndarray
    negative_abs_profile: np.ndarray
    reconstructed_rate_dense: np.ndarray


def filter_tokens(fit: TokenEventFit, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = config or {}
    mask = np.asarray(fit.valid_mask, dtype=bool).copy()
    finite = (
        np.isfinite(fit.tau)
        & np.isfinite(fit.lambda_)
        & np.isfinite(fit.delta)
        & np.isfinite(fit.r2)
        & np.isfinite(fit.start)
        & np.isfinite(fit.end)
    )
    mask &= finite
    mode = str(cfg.get("mode", "default")).lower()
    if mode in {"plot", "plot_compatible", "plot-compatible"}:
        drop = np.zeros_like(mask, dtype=bool)
        min_abs_delta = cfg.get("min_abs_delta")
        min_r2 = cfg.get("min_r2")
        if min_abs_delta is not None and min_r2 is not None:
            drop |= (np.abs(fit.delta) < float(min_abs_delta)) & (fit.r2 < float(min_r2))
        elif min_abs_delta is not None:
            drop |= np.abs(fit.delta) < float(min_abs_delta)
        elif min_r2 is not None:
            drop |= fit.r2 < float(min_r2)
        if cfg.get("min_lambda") is not None:
            drop |= fit.lambda_ < float(cfg["min_lambda"])
        if cfg.get("max_lambda") is not None:
            drop |= fit.lambda_ > float(cfg["max_lambda"])
        mask &= ~drop
        if cfg.get("require_positive_delta", False):
            mask &= fit.delta > 0
        if cfg.get("tau_min") is not None:
            mask &= fit.tau >= float(cfg["tau_min"])
        if cfg.get("tau_max") is not None:
            mask &= fit.tau <= float(cfg["tau_max"])
        return mask
    if cfg.get("require_positive_delta", False):
        mask &= fit.delta > 0
    if cfg.get("min_delta") is not None:
        mask &= fit.delta >= float(cfg["min_delta"])
    if cfg.get("min_abs_delta") is not None:
        mask &= np.abs(fit.delta) >= float(cfg["min_abs_delta"])
    if cfg.get("max_abs_delta") is not None:
        mask &= np.abs(fit.delta) <= float(cfg["max_abs_delta"])
    if cfg.get("min_r2") is not None:
        mask &= fit.r2 >= float(cfg["min_r2"])
    if cfg.get("min_lambda") is not None:
        mask &= fit.lambda_ >= float(cfg["min_lambda"])
    if cfg.get("max_lambda") is not None:
        mask &= fit.lambda_ <= float(cfg["max_lambda"])
    if cfg.get("tau_min") is not None:
        mask &= fit.tau >= float(cfg["tau_min"])
    if cfg.get("tau_max") is not None:
        mask &= fit.tau <= float(cfg["tau_max"])
    return mask


def _bin_edges(values: np.ndarray, n_bins: int, binning: str, value_range: tuple[float, float] | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot bin empty values.")
    if value_range is None:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    else:
        lo, hi = map(float, value_range)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    if binning == "quantile":
        q = np.linspace(0.0, 1.0, int(n_bins) + 1)
        edges = np.quantile(values, q)
        edges[0], edges[-1] = lo, hi
        edges = np.maximum.accumulate(edges)
        for idx in range(1, len(edges)):
            if edges[idx] <= edges[idx - 1]:
                edges[idx] = np.nextafter(edges[idx - 1], np.inf)
        return edges.astype(np.float64)
    return np.linspace(lo, hi, int(n_bins) + 1, dtype=np.float64)


def estimate_spectrum(
    tau: np.ndarray,
    *,
    bins: int = 80,
    binning: str = "uniform",
    value_range: tuple[float, float] | None = None,
) -> Spectrum:
    tau_arr = np.asarray(tau, dtype=np.float64)
    tau_arr = tau_arr[np.isfinite(tau_arr)]
    if tau_arr.size == 0:
        raise ValueError("No finite tau values for spectrum.")
    edges = _bin_edges(tau_arr, bins, binning, value_range)
    counts, edges = np.histogram(tau_arr, bins=edges)
    widths = np.diff(edges)
    mass = counts.astype(np.float64) / max(float(np.sum(counts)), 1.0)
    density = mass / np.maximum(widths, 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return Spectrum(edges, centers, counts.astype(np.int64), mass, density)


def estimate_signed_spectrum(
    tau: np.ndarray,
    delta: np.ndarray,
    *,
    bins: int = 80,
    binning: str = "uniform",
    value_range: tuple[float, float] | None = None,
) -> Spectrum:
    tau_arr = np.asarray(tau, dtype=np.float64)
    delta_arr = np.asarray(delta, dtype=np.float64)
    valid = np.isfinite(tau_arr) & np.isfinite(delta_arr)
    tau_arr = tau_arr[valid]
    delta_arr = delta_arr[valid]
    if tau_arr.size == 0:
        raise ValueError("No finite tau values for spectrum.")
    edges = _bin_edges(tau_arr, bins, binning, value_range)
    counts, edges = np.histogram(tau_arr, bins=edges)
    pos = np.histogram(tau_arr[delta_arr > 0.0], bins=edges)[0].astype(np.float64)
    neg = np.histogram(tau_arr[delta_arr < 0.0], bins=edges)[0].astype(np.float64)
    widths = np.diff(edges)
    denom = max(float(tau_arr.size), 1.0)
    mass = counts.astype(np.float64) / denom
    density = mass / np.maximum(widths, 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return Spectrum(
        edges,
        centers,
        counts.astype(np.int64),
        mass,
        density,
        pos_mass=pos / denom,
        neg_mass=neg / denom,
        diff_mass=(pos - neg) / denom,
    )


def compute_binned_learning_pulses(
    axis_values: np.ndarray,
    fit: TokenEventFit,
    mask: np.ndarray,
    config: dict[str, Any] | None = None,
) -> BinnedPulses:
    cfg = config or {}
    n_tau_bins = int(cfg.get("tau_bins", cfg.get("n_tau_bins", 10)))
    binning = str(cfg.get("binning", "quantile"))
    grid_points = int(cfg.get("offset_grid_points", 401))
    chunk_size = int(cfg.get("chunk_size", 50000))
    align_to_tau = bool(cfg.get("align_to_tau", False))
    x = np.asarray(axis_values, dtype=np.float64)
    span = float(np.max(x) - np.min(x))
    if align_to_tau:
        offset_range = cfg.get("offset_range")
        if offset_range is None:
            frac = float(cfg.get("offset_range_fraction", 1.0))
            lo, hi = -frac * span, frac * span
        else:
            lo, hi = map(float, offset_range)
        pulse_grid = np.linspace(lo, hi, grid_points, dtype=np.float64)
    else:
        pulse_grid = np.linspace(float(np.min(x)), float(np.max(x)), grid_points, dtype=np.float64)

    selected = np.flatnonzero(mask & (fit.delta > 0.0))
    if selected.size == 0:
        raise ValueError("No selected tokens for learning pulse computation.")
    selected_tau = fit.tau[selected].astype(np.float64)
    tau_range_cfg = cfg.get("tau_range", cfg.get("value_range"))
    tau_range = tuple(tau_range_cfg) if tau_range_cfg is not None else None
    tau_edges = _bin_edges(selected_tau, n_tau_bins, binning, tau_range)
    tau_centers = 0.5 * (tau_edges[:-1] + tau_edges[1:])
    pulse_by_bin = np.zeros((n_tau_bins, grid_points), dtype=np.float64)
    counts = np.zeros(n_tau_bins, dtype=np.int64)
    mean_total = np.zeros(grid_points, dtype=np.float64)
    mean_count = 0

    for bin_idx in range(n_tau_bins):
        if bin_idx == n_tau_bins - 1:
            in_bin = selected[(selected_tau >= tau_edges[bin_idx]) & (selected_tau <= tau_edges[bin_idx + 1])]
        else:
            in_bin = selected[(selected_tau >= tau_edges[bin_idx]) & (selected_tau < tau_edges[bin_idx + 1])]
        counts[bin_idx] = int(in_bin.size)
        if in_bin.size == 0:
            pulse_by_bin[bin_idx] = np.nan
            continue
        total = np.zeros(grid_points, dtype=np.float64)
        for start in range(0, in_bin.size, chunk_size):
            idx = in_bin[start : start + chunk_size]
            if align_to_tau:
                pulse_axis = pulse_grid
            else:
                pulse_axis = pulse_grid - tau_centers[bin_idx]
            values = learning_pulse(pulse_axis, fit.lambda_[idx], fit.delta[idx])
            total += np.sum(values, axis=0)
        pulse_by_bin[bin_idx] = total / max(int(in_bin.size), 1)
        mean_total += total
        mean_count += int(in_bin.size)

    pulse_mean = mean_total / max(mean_count, 1)
    return BinnedPulses(pulse_grid, tau_edges, tau_centers, pulse_by_bin, pulse_mean, counts, align_to_tau)


def build_u_grid(axis_values: np.ndarray, num_points: int = 999) -> np.ndarray:
    x = np.asarray(axis_values, dtype=np.float64).ravel()
    num_points = max(int(num_points), 257)
    span = float(np.max(x) - np.min(x))
    if span <= 0.0:
        return np.zeros(num_points, dtype=np.float64)
    return np.linspace(-span / 2.0, span / 2.0, num_points, dtype=np.float64)


def mean_profile(
    delta: np.ndarray,
    lambda_: np.ndarray,
    u_grid: np.ndarray,
    mask: np.ndarray,
    *,
    absolute: bool = False,
) -> np.ndarray:
    delta_arr = np.asarray(delta, dtype=np.float64).ravel()
    lam_arr = np.asarray(lambda_, dtype=np.float64).ravel()
    u = np.asarray(u_grid, dtype=np.float64).ravel()
    mask_arr = np.asarray(mask, dtype=bool).ravel()
    if not np.any(mask_arr):
        return np.zeros_like(u)
    delta_use = np.abs(delta_arr[mask_arr]) if absolute else delta_arr[mask_arr]
    lam_use = lam_arr[mask_arr]
    z = np.clip(lam_use[:, None] * u[None, :] / 2.0, -20.0, 20.0)
    profiles = (delta_use[:, None] * lam_use[:, None] / 4.0) / (np.cosh(z) ** 2)
    return profiles.mean(axis=0)


def _sum_shifted_profile(
    profile_u: np.ndarray,
    tau: np.ndarray,
    x_eval: np.ndarray,
    u_grid: np.ndarray,
    *,
    chunk_size: int = 50000,
) -> np.ndarray:
    profile = np.asarray(profile_u, dtype=np.float64).ravel()
    tau_arr = np.asarray(tau, dtype=np.float64).ravel()
    x = np.asarray(x_eval, dtype=np.float64).ravel()
    u = np.asarray(u_grid, dtype=np.float64).ravel()
    if tau_arr.size == 0:
        return np.zeros_like(x)
    u_lo = float(u[0])
    u_hi = float(u[-1])
    total = np.zeros_like(x, dtype=np.float64)
    for start in range(0, tau_arr.size, int(chunk_size)):
        tau_chunk = tau_arr[start : start + int(chunk_size)]
        u_eval = x[None, :] - tau_chunk[:, None]
        oob = (u_eval < u_lo) | (u_eval > u_hi)
        if np.any(oob):
            u_eval = u_eval.copy()
            u_eval[oob] = -u_eval[oob]
        u_eval = np.clip(u_eval, u_lo, u_hi)
        shifted = np.interp(u_eval.reshape(-1), u, profile).reshape(tau_chunk.size, x.size)
        total += shifted.sum(axis=0)
    return total


def reconstruct_rate_spectrum_dominated(
    axis_values: np.ndarray,
    fit: TokenEventFit,
    mask: np.ndarray,
    *,
    dense_points: int = 999,
    denominator: int | None = None,
    tau_range_for_g_mean: tuple[float, float] | None = None,
    chunk_size: int = 50000,
) -> SpectrumReconstruction:
    x = np.asarray(axis_values, dtype=np.float64)
    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), int(dense_points), dtype=np.float64)
    if np.all(x_dense[:-1] > 0):
        x_rate_dense = np.sqrt(x_dense[:-1] * x_dense[1:])
    else:
        x_rate_dense = 0.5 * (x_dense[:-1] + x_dense[1:])
    u_grid = build_u_grid(x, int(dense_points))
    valid = np.asarray(mask, dtype=bool) & np.isfinite(fit.tau) & np.isfinite(fit.delta) & np.isfinite(fit.lambda_)
    pos_mask = valid & (fit.delta > 0.0)
    neg_mask = valid & (fit.delta < 0.0)
    mean_tau_mask = np.ones_like(valid, dtype=bool)
    if tau_range_for_g_mean is not None:
        lo, hi = map(float, tau_range_for_g_mean)
        mean_tau_mask = (fit.tau >= lo) & (fit.tau <= hi)
    profile_pos = mean_profile(fit.delta, fit.lambda_, u_grid, pos_mask & mean_tau_mask, absolute=False)
    profile_neg_abs = mean_profile(fit.delta, fit.lambda_, u_grid, neg_mask & mean_tau_mask, absolute=True)
    pos_sum = _sum_shifted_profile(profile_pos, fit.tau[pos_mask], x_rate_dense, u_grid, chunk_size=chunk_size)
    neg_sum = _sum_shifted_profile(profile_neg_abs, fit.tau[neg_mask], x_rate_dense, u_grid, chunk_size=chunk_size)
    denom = float(max(int(denominator if denominator is not None else fit.n_tokens), 1))
    recon = (pos_sum - neg_sum) / denom
    return SpectrumReconstruction(x_dense, x_rate_dense, u_grid, profile_pos, profile_neg_abs, recon)


def reconstruct_rate(
    eval_axis: np.ndarray,
    spectrum: Spectrum,
    offset_grid: np.ndarray,
    pulse_mean: np.ndarray,
) -> np.ndarray:
    x = np.asarray(eval_axis, dtype=np.float64)
    offsets = x[:, None] - spectrum.bin_centers[None, :]
    flat_offsets = offsets.reshape(-1)
    flat_values = np.interp(flat_offsets, offset_grid, pulse_mean, left=0.0, right=0.0)
    values = flat_values.reshape(offsets.shape)
    return values @ spectrum.mass
