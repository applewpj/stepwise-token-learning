from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tokenlearning.config import deep_get, load_config
from tokenlearning.core.fitting import fit_token_events, loss_snr_keep_mask
from tokenlearning.core.rates import loss_change_rate
from tokenlearning.core.sigmoid import mean_sigmoid_loss
from tokenlearning.core.spectrum import (
    compute_binned_learning_pulses,
    estimate_signed_spectrum,
    filter_tokens,
    reconstruct_rate_spectrum_dominated,
)
from tokenlearning.io.loss_matrix import load_loss_npz, load_public_fit_cache, save_fit_cache
from tokenlearning.types import LossMatrix, TokenEventFit
from tokenlearning.viz.plots import (
    plot_binned_pulses,
    plot_grouped_mean_loss,
    plot_mean_loss,
    plot_rate_reconstruction,
    plot_rate_vs_spectrum,
    plot_spectrum,
    plot_token_examples,
    plot_token_examples_by_label,
)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _compute_loss_snr_filter_if_requested(
    loss_data: LossMatrix | None,
    filter_cfg: dict[str, Any],
    cache_dir: Path,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if loss_data is None:
        return None, None
    loss_snr_cfg = filter_cfg.get("loss_snr") or filter_cfg.get("loss_snr_filter")
    if loss_snr_cfg is None:
        return None, None
    filter_cfg = loss_snr_cfg
    if not filter_cfg or not bool(filter_cfg.get("enabled", False)):
        return None, None
    keep_mask = loss_snr_keep_mask(loss_data.loss_matrix, filter_cfg)
    np.save(cache_dir / "loss_snr_mask.npy", keep_mask)
    summary = {
        "method": str(filter_cfg.get("method", "snr")),
        "min_snr": float(filter_cfg.get("min_snr", 1.0)),
        "smooth_window": int(filter_cfg.get("smooth_window", 3)),
        "edge_points": int(filter_cfg.get("edge_points", min(3, loss_data.loss_matrix.shape[1]))),
        "raw_tokens": int(loss_data.loss_matrix.shape[0]),
        "kept_tokens": int(np.sum(keep_mask)),
        "dropped_tokens": int(np.sum(~keep_mask)),
    }
    print(
        "[loss-snr-filter] "
        f"raw={summary['raw_tokens']} kept={summary['kept_tokens']} dropped={summary['dropped_tokens']}",
        flush=True,
    )
    return keep_mask, summary


def _finite_fit_mask(fit: TokenEventFit) -> np.ndarray:
    return (
        np.isfinite(fit.tau)
        & np.isfinite(fit.lambda_)
        & np.isfinite(fit.delta)
        & np.isfinite(fit.start)
        & np.isfinite(fit.end)
        & np.isfinite(fit.r2)
        & (fit.lambda_ > 0.0)
    )


def _load_inputs(
    config: dict[str, Any],
    out_dir: Path,
) -> tuple[np.ndarray, TokenEventFit, np.ndarray | None, np.ndarray | None, LossMatrix | None]:
    input_cfg = config.get("input", {})
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    public_cache_path = cache_dir / "fit_params.npz"
    reuse_cache = bool(deep_get(config, "output.reuse_cache", True))
    losses_npz = input_cfg.get("losses_npz")
    loss_data = load_loss_npz(losses_npz, axis_name=input_cfg.get("axis_name")) if losses_npz else None

    if reuse_cache and public_cache_path.exists():
        axis_values, fit, raw_mean, fitted_mean = load_public_fit_cache(public_cache_path)
        if loss_data is not None:
            if loss_data.axis_values.shape != axis_values.shape or not np.allclose(loss_data.axis_values, axis_values):
                raise ValueError(f"Cached axis_values in {public_cache_path} do not match {losses_npz}")
            if fit.n_tokens != loss_data.loss_matrix.shape[0]:
                raise ValueError(
                    f"Cached fit token count in {public_cache_path} does not match loaded losses: "
                    f"{fit.n_tokens} vs {loss_data.loss_matrix.shape[0]}"
                )
            if raw_mean is None:
                raw_mean = np.mean(loss_data.loss_matrix.astype(np.float64), axis=0)
        return axis_values, fit, raw_mean, fitted_mean, loss_data

    if loss_data is not None:
        raw_mean = np.mean(loss_data.loss_matrix.astype(np.float64), axis=0)
        fit = fit_token_events(loss_data.axis_values, loss_data.loss_matrix, config.get("fit", {}))
        fitted_mean = mean_sigmoid_loss(
            loss_data.axis_values,
            fit,
            mask=_finite_fit_mask(fit),
            chunk_size=int(deep_get(config, "runtime.mean_curve_chunk_size", 50000)),
        )
        save_fit_cache(public_cache_path, axis_values=loss_data.axis_values, fit=fit, raw_loss_mean=raw_mean, fitted_loss_mean=fitted_mean)
        return loss_data.axis_values, fit, raw_mean, fitted_mean, loss_data

    raise ValueError("Core analysis requires input.losses_npz.")


def _get_label_array(loss_data: LossMatrix | None, field: str | None) -> np.ndarray | None:
    if loss_data is None or not field:
        return None
    if field == "sample_id":
        if loss_data.sample_id is None:
            raise ValueError("Label field 'sample_id' was requested, but losses.npz has no sample_id array")
        return loss_data.sample_id
    if field == "token_pos":
        if loss_data.token_pos is None:
            raise ValueError("Label field 'token_pos' was requested, but losses.npz has no token_pos array")
        return loss_data.token_pos
    if loss_data.metadata is not None and field in loss_data.metadata:
        return loss_data.metadata[field]
    available = []
    if loss_data.sample_id is not None:
        available.append("sample_id")
    if loss_data.token_pos is not None:
        available.append("token_pos")
    if loss_data.metadata is not None:
        available.extend(sorted(loss_data.metadata))
    raise ValueError(f"Label field {field!r} not found in losses.npz. Available fields: {available}")


def _range_tuple(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Range config must be a length-2 list, got {value!r}")
    return float(value[0]), float(value[1])


def _loss_decrease_rate(
    axis_values: np.ndarray,
    mean_loss: np.ndarray,
    *,
    normalize_by_axis: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if normalize_by_axis:
        return loss_change_rate(axis_values, mean_loss)
    x = np.asarray(axis_values, dtype=np.float64)
    y = np.asarray(mean_loss, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"axis_values and mean_loss must be matching 1D arrays, got {x.shape}, {y.shape}")
    if np.any(np.diff(x) <= 0):
        raise ValueError("axis_values must be strictly increasing.")
    return x[1:], -np.diff(y)


def _apply_token_example_filters(mask: np.ndarray, fit: TokenEventFit, cfg: dict[str, Any], default_min_r2: Any = None) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    threshold = cfg.get("min_r2", cfg.get("plot_min_r2", default_min_r2))
    if threshold is None:
        threshold = default_min_r2
    if threshold is not None:
        result &= np.isfinite(fit.r2) & (fit.r2 > float(threshold))

    direction = cfg.get("direction", cfg.get("trend", "down"))
    if direction is not None:
        direction_text = str(direction).lower()
        if direction_text in {"down", "decrease", "decreasing", "drop", "loss_decrease"}:
            result &= np.isfinite(fit.delta) & (fit.delta > 0.0)
        elif direction_text in {"up", "increase", "increasing", "rise", "loss_increase"}:
            result &= np.isfinite(fit.delta) & (fit.delta < 0.0)
        elif direction_text not in {"any", "all", "none", "null"}:
            raise ValueError(f"Unsupported token_examples.direction={direction!r}")

    min_delta = cfg.get("min_delta", cfg.get("delta_min"))
    if min_delta is not None:
        min_delta_value = float(min_delta)
        direction_text = "down" if direction is None else str(direction).lower()
        if direction_text in {"up", "increase", "increasing", "rise", "loss_increase"}:
            result &= -fit.delta > min_delta_value
        elif direction_text in {"any", "all", "none", "null"}:
            result &= np.abs(fit.delta) > min_delta_value
        else:
            result &= fit.delta > min_delta_value
    return result


def run_analysis(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    out_dir = Path(deep_get(config, "output.dir", "runs/default")).resolve()
    fig_dir = out_dir / "figures"
    cache_dir = out_dir / "cache"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    axis_name = str(deep_get(config, "input.axis_name", "axis"))

    axis_values, fit, raw_mean, fitted_mean, loss_data = _load_inputs(config, out_dir)
    if axis_name == "axis" and loss_data is not None:
        axis_name = loss_data.axis_name
    plot_mean_loss(fig_dir / "mean_loss_raw_vs_sigmoid.png", axis_values, raw_mean, fitted_mean, axis_name=axis_name)

    label_cfg = config.get("labels", {}) or {}
    label_field = label_cfg.get("field")
    label_prefix = str(label_cfg.get("prefix", ""))
    label_values = _get_label_array(loss_data, str(label_field) if label_field else None)
    grouped_loss_arrays: dict[str, np.ndarray] | None = None
    if label_values is not None and loss_data is not None and bool(label_cfg.get("plot_grouped_loss", True)):
        grouped_loss_arrays = plot_grouped_mean_loss(
            fig_dir / f"mean_loss_by_{label_field}.png",
            axis_values,
            loss_data.loss_matrix,
            label_values,
            label_prefix=label_prefix,
            label_name=str(label_field),
            axis_name=axis_name,
            max_groups=label_cfg.get("max_groups"),
            include_overall=bool(label_cfg.get("include_overall", True)),
        )

    filter_cfg = config.get("filters", {})
    selected_mask = filter_tokens(fit, filter_cfg)
    loss_filter_summary = None
    loss_filter_mask, loss_filter_summary = _compute_loss_snr_filter_if_requested(loss_data, filter_cfg, cache_dir)
    if loss_filter_mask is not None:
        if loss_filter_mask.shape[0] != selected_mask.shape[0]:
            raise ValueError(
                f"loss_snr filter length must match fitted tokens: {loss_filter_mask.shape[0]} vs {selected_mask.shape[0]}"
            )
        selected_mask &= loss_filter_mask
    np.save(cache_dir / "filtered_mask.npy", selected_mask)

    token_cfg = config.get("token_examples", {})
    title_field = token_cfg.get("title_field")
    title_values = _get_label_array(loss_data, str(title_field) if title_field else None)
    token_example_mask = _apply_token_example_filters(selected_mask, fit, token_cfg)
    try:
        sampled_indices = plot_token_examples(
            fig_dir / "token_examples_by_tau_decile.png",
            axis_values,
            fit,
            token_example_mask,
            loss_matrix=loss_data.loss_matrix if loss_data is not None else None,
            labels=label_values,
            titles=title_values,
            label_prefix=label_prefix,
            tau_bins=int(token_cfg.get("tau_bins", 10)),
            samples_per_bin=int(token_cfg.get("samples_per_bin", 5)),
            tau_binning=str(token_cfg.get("binning", token_cfg.get("tau_binning", "quantile"))),
            seed=int(token_cfg.get("seed", 1234)),
            axis_name=axis_name,
            label_mean_window=int(token_cfg.get("label_mean_window", 3)),
            max_title_chars=int(token_cfg.get("max_title_chars", 52)),
        )
    except ValueError:
        sampled_indices = []

    label_sampled_indices: list[int] = []
    label_examples_cfg = config.get("label_examples", {}) or {}
    if (
        label_values is not None
        and loss_data is not None
        and bool(label_examples_cfg.get("enabled", label_cfg.get("plot_token_examples_by_label", False)))
    ):
        try:
            if bool(label_examples_cfg.get("use_filtered_mask", True)):
                label_example_mask = selected_mask
            else:
                label_filter_cfg = {k: v for k, v in filter_cfg.items() if k not in {"tau_min", "tau_max"}}
                label_example_mask = filter_tokens(fit, label_filter_cfg)
            label_example_mask = _apply_token_example_filters(
                label_example_mask,
                fit,
                label_examples_cfg,
                default_min_r2=token_cfg.get("min_r2", token_cfg.get("plot_min_r2")),
            )
            label_sampled_indices = plot_token_examples_by_label(
                fig_dir / f"token_examples_by_{label_field}.png",
                axis_values,
                fit,
                label_example_mask,
                label_values,
                loss_matrix=loss_data.loss_matrix,
                titles=title_values,
                label_prefix=label_prefix,
                samples_per_label=int(label_examples_cfg.get("samples_per_label", token_cfg.get("samples_per_bin", 5))),
                max_labels=label_examples_cfg.get("max_labels", label_cfg.get("max_groups")),
                axis_name=axis_name,
                label_mean_window=int(label_examples_cfg.get("label_mean_window", token_cfg.get("label_mean_window", 3))),
                max_title_chars=int(label_examples_cfg.get("max_title_chars", token_cfg.get("max_title_chars", 52))),
            )
        except ValueError:
            label_sampled_indices = []

    spectrum_cfg = config.get("spectrum", {})
    spectrum = estimate_signed_spectrum(
        fit.tau[selected_mask],
        fit.delta[selected_mask],
        bins=int(spectrum_cfg.get("bins", 80)),
        binning=str(spectrum_cfg.get("binning", "uniform")),
        value_range=_range_tuple(spectrum_cfg.get("range", spectrum_cfg.get("value_range"))),
    )
    pulse_cfg = config.get("pulse", {})
    pulses = compute_binned_learning_pulses(axis_values, fit, selected_mask, pulse_cfg)
    plot_binned_pulses(fig_dir / "binned_learning_pulses.png", pulses, axis_name=axis_name)

    rate_source = raw_mean if raw_mean is not None else fitted_mean
    empirical_x, empirical_rate = (None, None)
    reconstructed_rate = None
    reconstructed_rate_dense = None
    x_rate_dense = None
    sigmoid_rate = None
    if rate_source is not None:
        rate_cfg = config.get("rate", {}) or {}
        normalize_by_axis = bool(rate_cfg.get("normalize_by_axis", True))
        empirical_x, empirical_rate = _loss_decrease_rate(
            axis_values,
            rate_source,
            normalize_by_axis=normalize_by_axis,
        )
        rate_scale = 1.0
        if not normalize_by_axis:
            rate_scale = float(rate_cfg.get("axis_scale", np.median(np.diff(axis_values.astype(np.float64)))))
        recon_cfg = config.get("reconstruction", {})
        tau_range = recon_cfg.get("tau_range_for_g_mean")
        tau_range_tuple = tuple(tau_range) if tau_range is not None else None
        reconstruction = reconstruct_rate_spectrum_dominated(
            axis_values,
            fit,
            selected_mask,
            dense_points=int(recon_cfg.get("dense_points", 999)),
            denominator=int(recon_cfg.get("denominator", getattr(fit, "raw_token_count", fit.n_tokens))),
            tau_range_for_g_mean=tau_range_tuple,
            chunk_size=int(recon_cfg.get("chunk_size", 50000)),
        )
        x_rate_dense = reconstruction.x_rate_dense
        reconstructed_rate_dense = reconstruction.reconstructed_rate_dense * rate_scale
        reconstructed_rate = np.interp(empirical_x, x_rate_dense, reconstructed_rate_dense)
        if fitted_mean is not None:
            _, sigmoid_rate = _loss_decrease_rate(
                axis_values,
                fitted_mean,
                normalize_by_axis=normalize_by_axis,
            )
        plot_rate_vs_spectrum(
            fig_dir / "learning_time_spectrum.png",
            spectrum,
            empirical_x,
            empirical_rate,
            axis_name=axis_name,
            curve=str(spectrum_cfg.get("curve", "pos_mass")),
        )
        plot_rate_reconstruction(
            fig_dir / "rate_reconstruction.png",
            x_rate_dense,
            empirical_rate,
            reconstructed_rate_dense,
            sigmoid_rate=sigmoid_rate,
            empirical_axis=empirical_x,
            sigmoid_axis=empirical_x,
            axis_name=axis_name,
        )
        if bool(rate_cfg.get("plot_loglog", rate_cfg.get("loglog", False))):
            plot_rate_reconstruction(
                fig_dir / "rate_reconstruction_loglog.png",
                x_rate_dense,
                empirical_rate,
                reconstructed_rate_dense,
                sigmoid_rate=sigmoid_rate,
                empirical_axis=empirical_x,
                sigmoid_axis=empirical_x,
                axis_name=axis_name,
                loglog=True,
            )
    else:
        plot_spectrum(
            fig_dir / "learning_time_spectrum.png",
            spectrum,
            axis_name=axis_name,
            curve=str(spectrum_cfg.get("curve", "pos_mass")),
        )

    summary_path = fig_dir / "reconstruction_summary_1x3.png"
    if summary_path.exists():
        summary_path.unlink()

    analysis_payload: dict[str, np.ndarray] = {
        "axis_values": axis_values,
        "raw_loss_mean": raw_mean if raw_mean is not None else np.array([], dtype=np.float64),
        "fitted_loss_mean": fitted_mean if fitted_mean is not None else np.array([], dtype=np.float64),
        "spectrum_bin_edges": spectrum.bin_edges,
        "spectrum_bin_centers": spectrum.bin_centers,
        "spectrum_density": spectrum.density,
        "spectrum_mass": spectrum.mass,
        "spectrum_pos_mass": spectrum.pos_mass if spectrum.pos_mass is not None else np.array([], dtype=np.float64),
        "spectrum_neg_mass": spectrum.neg_mass if spectrum.neg_mass is not None else np.array([], dtype=np.float64),
        "spectrum_diff_mass": spectrum.diff_mass if spectrum.diff_mass is not None else np.array([], dtype=np.float64),
        "pulse_offset_grid": pulses.offset_grid,
        "pulse_aligned_to_tau": np.asarray([pulses.aligned_to_tau], dtype=bool),
        "pulse_by_bin": pulses.pulse_by_bin,
        "pulse_mean": pulses.pulse_mean,
        "pulse_tau_bin_edges": pulses.tau_bin_edges,
        "rate_axis": empirical_x if empirical_x is not None else np.array([], dtype=np.float64),
        "empirical_rate": empirical_rate if empirical_rate is not None else np.array([], dtype=np.float64),
        "reconstructed_rate": reconstructed_rate if reconstructed_rate is not None else np.array([], dtype=np.float64),
        "rate_axis_dense": x_rate_dense if x_rate_dense is not None else np.array([], dtype=np.float64),
        "reconstructed_rate_dense": reconstructed_rate_dense if reconstructed_rate_dense is not None else np.array([], dtype=np.float64),
        "sigmoid_rate": sigmoid_rate if sigmoid_rate is not None else np.array([], dtype=np.float64),
    }
    if grouped_loss_arrays is not None:
        analysis_payload.update(
            {
                "group_label_values": grouped_loss_arrays["group_values"],
                "group_label_names": grouped_loss_arrays["group_names"],
                "group_label_counts": grouped_loss_arrays["group_counts"],
                "group_raw_loss_mean": grouped_loss_arrays["group_mean_loss"],
            }
        )
    np.savez_compressed(cache_dir / "analysis_arrays.npz", **analysis_payload)

    metrics: dict[str, Any] = {
        "config_path": str(Path(config_path).resolve()),
        "output_dir": str(out_dir),
        "n_tokens": fit.n_tokens,
        "n_valid_tokens": int(np.sum(fit.valid_mask)),
        "n_filtered_tokens": int(np.sum(selected_mask)),
        "axis_min": float(np.min(axis_values)),
        "axis_max": float(np.max(axis_values)),
        "tau_min_filtered": float(np.min(fit.tau[selected_mask])) if np.any(selected_mask) else None,
        "tau_max_filtered": float(np.max(fit.tau[selected_mask])) if np.any(selected_mask) else None,
        "mean_r2_filtered": float(np.mean(fit.r2[selected_mask])) if np.any(selected_mask) else None,
        "sampled_token_indices": sampled_indices,
        "sampled_token_indices_by_label": label_sampled_indices,
        "figures": {
            "mean_loss": str(fig_dir / "mean_loss_raw_vs_sigmoid.png"),
            "token_examples": str(fig_dir / "token_examples_by_tau_decile.png"),
            "spectrum": str(fig_dir / "learning_time_spectrum.png"),
            "pulses": str(fig_dir / "binned_learning_pulses.png"),
            "rate_reconstruction": str(fig_dir / "rate_reconstruction.png"),
        },
    }
    if bool(deep_get(config, "rate.plot_loglog", deep_get(config, "rate.loglog", False))):
        metrics["figures"]["rate_reconstruction_loglog"] = str(fig_dir / "rate_reconstruction_loglog.png")
    if loss_filter_summary is not None:
        metrics["loss_snr_filter"] = loss_filter_summary
    if token_cfg.get("min_r2", token_cfg.get("plot_min_r2")) is not None:
        metrics["token_examples_min_r2"] = float(token_cfg.get("min_r2", token_cfg.get("plot_min_r2")))
    if label_examples_cfg.get("min_r2", label_examples_cfg.get("plot_min_r2")) is not None:
        metrics["label_examples_min_r2"] = float(label_examples_cfg.get("min_r2", label_examples_cfg.get("plot_min_r2")))
    if label_values is not None and label_field:
        metrics["label_field"] = str(label_field)
        metrics["label_prefix"] = label_prefix
        metrics["figures"][f"mean_loss_by_{label_field}"] = str(fig_dir / f"mean_loss_by_{label_field}.png")
        if label_sampled_indices:
            metrics["figures"][f"token_examples_by_{label_field}"] = str(fig_dir / f"token_examples_by_{label_field}.png")
        unique, counts = np.unique(label_values, return_counts=True)
        metrics["label_counts"] = {str(k): int(v) for k, v in zip(unique, counts, strict=False)}
    if empirical_rate is not None and reconstructed_rate is not None:
        valid = np.isfinite(empirical_rate) & np.isfinite(reconstructed_rate)
        if np.sum(valid) >= 2:
            metrics["rate_reconstruction_corr"] = float(np.corrcoef(empirical_rate[valid], reconstructed_rate[valid])[0, 1])
            metrics["rate_reconstruction_mae"] = float(np.mean(np.abs(empirical_rate[valid] - reconstructed_rate[valid])))
    if sigmoid_rate is not None and reconstructed_rate is not None:
        valid = np.isfinite(sigmoid_rate) & np.isfinite(reconstructed_rate)
        if np.sum(valid) >= 2:
            metrics["sigmoid_rate_reconstruction_corr"] = float(
                np.corrcoef(sigmoid_rate[valid], reconstructed_rate[valid])[0, 1]
            )
            metrics["sigmoid_rate_reconstruction_mae"] = float(
                np.mean(np.abs(sigmoid_rate[valid] - reconstructed_rate[valid]))
            )
    _write_json(out_dir / "metrics.json", metrics)
    return metrics
