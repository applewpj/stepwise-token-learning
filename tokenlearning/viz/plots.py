from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import numpy as np

from tokenlearning.core.sigmoid import sigmoid_loss
from tokenlearning.core.spectrum import BinnedPulses, Spectrum
from tokenlearning.types import TokenEventFit


LOSS_COLOR = "#1F4E79"
FIT_COLOR = "#C43C39"
SPECTRUM_COLOR = "#4CAF50"
MEAN_COLOR = "#3E8E41"
SHADE_COLOR = "#D95F5F"
GRID_COLOR = "#CDD3DA"
SPINE_COLOR = "#7A8087"
FIGURE_FACE_COLOR = "#FFFFFF"
AXES_FACE_COLOR = "#FFFFFF"

_STYLE_CONFIGURED = False


def configure_plot_style() -> None:
    global _STYLE_CONFIGURED
    if _STYLE_CONFIGURED:
        return
    font_candidates = [
        "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
        "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf",
    ]
    for font_path in font_candidates:
        path = Path(font_path)
        if not path.exists():
            continue
        font_manager.fontManager.addfont(str(path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
        break
    else:
        plt.rcParams["font.family"] = "DejaVu Serif"

    plt.rcParams.update(
        {
            "figure.facecolor": FIGURE_FACE_COLOR,
            "axes.facecolor": AXES_FACE_COLOR,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
        }
    )
    _STYLE_CONFIGURED = True


def style_axis(ax: plt.Axes, *, facecolor: str = AXES_FACE_COLOR) -> None:
    ax.grid(True, alpha=0.14, color=GRID_COLOR, linewidth=0.6)
    ax.tick_params(length=3.0, width=0.75)
    ax.set_facecolor(facecolor)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color(SPINE_COLOR)


def padded_limits(y: np.ndarray, frac: float = 0.06) -> tuple[float, float]:
    values = np.asarray(y, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    scale = max(abs(ymin), abs(ymax), 1e-12)
    span = max(ymax - ymin, scale * frac)
    pad = span * frac
    return ymin - pad, ymax + pad


def _finish(fig: plt.Figure, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=FIGURE_FACE_COLOR)
    plt.close(fig)


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    values = np.asarray(y, dtype=np.float64)
    if window <= 1 or values.size == 0:
        return values
    idx = np.arange(values.shape[0], dtype=np.int64)
    half = window // 2
    left = np.clip(idx - half, 0, values.shape[0] - 1)
    right = np.clip(idx + half, 0, values.shape[0] - 1)
    csum = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(values)])
    return (csum[right + 1] - csum[left]) / (right - left + 1)


def _format_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


def _format_group_label(value: Any, prefix: str = "") -> str:
    return f"{prefix}{_format_value(value)}" if prefix else _format_value(value)


def _short_title(value: Any, max_chars: int) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    text = str(value)
    return textwrap.shorten(text, width=max(int(max_chars), 8), placeholder="...")


def _append_r2_to_title(title: str, r2: float) -> str:
    if np.isfinite(r2):
        return f"{title}\n$R^2$={r2:.2f}"
    return f"{title}\n$R^2$=nan"


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(values))
    try:
        order = np.argsort(unique.astype(np.float64), kind="mergesort")
        return unique[order]
    except (TypeError, ValueError):
        return unique


def _grouped_means(
    loss_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    max_groups: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_values = np.asarray(labels).reshape(-1)
    losses = np.asarray(loss_matrix, dtype=np.float64)
    if label_values.shape[0] != losses.shape[0]:
        raise ValueError(f"labels length must match number of tokens: {label_values.shape[0]} vs {losses.shape[0]}")
    group_values = _ordered_unique(label_values)
    if max_groups is not None:
        group_values = group_values[: int(max_groups)]
    means = []
    counts = []
    for value in group_values:
        mask = label_values == value
        counts.append(int(np.sum(mask)))
        means.append(np.mean(losses[mask], axis=0))
    return group_values, np.asarray(means, dtype=np.float64), np.asarray(counts, dtype=np.int64)


def _select_spectrum_curve(spectrum: Spectrum, curve: str) -> tuple[np.ndarray, str]:
    if curve == "diff_mass" and spectrum.diff_mass is not None:
        return spectrum.diff_mass, "Mass"
    if curve == "mass":
        return spectrum.mass, "Mass"
    if curve == "density":
        return spectrum.density, "Density"
    if spectrum.pos_mass is not None:
        return spectrum.pos_mass, "Mass"
    return spectrum.mass, "Mass"


def _sigmoid_drop_interval(tau: float, lambda_: float, axis_values: np.ndarray, frac: float = 0.1) -> tuple[float, float]:
    if not np.isfinite(tau) or not np.isfinite(lambda_) or lambda_ <= 0:
        return float(tau), float(tau)
    eps = max(min(float(frac), 1.0 - 1e-6), 1e-6)
    width = np.log((1.0 - eps) / eps) / float(lambda_)
    xmin = float(np.min(axis_values))
    xmax = float(np.max(axis_values))
    lo = tau - width
    hi = tau + width
    if hi < xmin:
        return xmin, xmin
    if lo > xmax:
        return xmax, xmax
    return max(xmin, lo), min(xmax, hi)


def plot_mean_loss(
    path: str | Path,
    axis_values: np.ndarray,
    raw_mean: np.ndarray | None,
    sigmoid_mean: np.ndarray | None,
    *,
    axis_name: str = "axis",
) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    if raw_mean is not None:
        ax.plot(axis_values, raw_mean, color=LOSS_COLOR, linewidth=2.3, label="Mean validation loss")
    if sigmoid_mean is not None:
        ax.plot(axis_values, sigmoid_mean, color=FIT_COLOR, linewidth=2.1, linestyle="--", label="Mean sigmoid fit")
    ax.set_title("Mean validation loss")
    ax.set_xlabel(axis_name)
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)
    _finish(fig, path)


def plot_grouped_mean_loss(
    path: str | Path,
    axis_values: np.ndarray,
    loss_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    label_prefix: str = "",
    label_name: str = "label",
    axis_name: str = "axis",
    max_groups: int | None = None,
    include_overall: bool = True,
) -> dict[str, np.ndarray]:
    configure_plot_style()
    group_values, group_means, group_counts = _grouped_means(loss_matrix, labels, max_groups=max_groups)
    group_names = np.asarray([_format_group_label(value, label_prefix) for value in group_values])
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, max(len(group_values), 1)))

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    if include_overall:
        ax.plot(axis_values, np.mean(loss_matrix, axis=0), color="#222222", linewidth=2.5, label="Overall")
    for name, mean_curve, color in zip(group_names, group_means, colors, strict=False):
        ax.plot(axis_values, mean_curve, color=color, linewidth=1.65, alpha=0.95, label=str(name))

    ax.set_title(f"Mean validation loss by {label_name}")
    ax.set_xlabel(axis_name)
    ax.set_ylabel("Loss")
    ncol = 2 if len(group_values) <= 10 else 3
    ax.legend(frameon=False, loc="upper right", ncol=ncol)
    style_axis(ax)
    _finish(fig, path)
    return {
        "group_values": group_values,
        "group_names": group_names,
        "group_counts": group_counts,
        "group_mean_loss": group_means,
    }


def plot_token_examples(
    path: str | Path,
    axis_values: np.ndarray,
    fit: TokenEventFit,
    mask: np.ndarray,
    *,
    loss_matrix: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    titles: np.ndarray | None = None,
    label_prefix: str = "",
    tau_bins: int = 10,
    samples_per_bin: int = 5,
    tau_binning: str = "quantile",
    seed: int = 1234,
    axis_name: str = "axis",
    label_mean_window: int = 3,
    max_title_chars: int = 52,
) -> list[int]:
    configure_plot_style()
    rng = np.random.default_rng(int(seed))
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        raise ValueError("No selected tokens for token example plot.")
    label_values = None if labels is None else np.asarray(labels).reshape(-1)
    if label_values is not None and label_values.shape[0] != fit.n_tokens:
        raise ValueError(f"labels length must match number of tokens: {label_values.shape[0]} vs {fit.n_tokens}")
    title_values = None if titles is None else np.asarray(titles).reshape(-1)
    if title_values is not None and title_values.shape[0] != fit.n_tokens:
        raise ValueError(f"titles length must match number of tokens: {title_values.shape[0]} vs {fit.n_tokens}")
    label_means: dict[Any, np.ndarray] = {}
    if label_values is not None and loss_matrix is not None:
        for value in _ordered_unique(label_values):
            label_means[value.item() if isinstance(value, np.generic) else value] = _moving_average(
                np.mean(np.asarray(loss_matrix)[label_values == value], axis=0),
                label_mean_window,
            )

    n_rows = int(tau_bins)
    n_cols = int(samples_per_bin)
    binning = str(tau_binning).lower()
    sampled_by_bin: list[list[int]] = []
    bin_labels: list[str] = []
    if binning in {"axis", "uniform_axis", "uniform"}:
        edges = np.linspace(float(np.min(axis_values)), float(np.max(axis_values)), n_rows + 1, dtype=np.float64)
        tau_selected = np.asarray(fit.tau[selected], dtype=np.float64)
        for row in range(n_rows):
            if row == n_rows - 1:
                in_bin = selected[(tau_selected >= edges[row]) & (tau_selected <= edges[row + 1])]
            else:
                in_bin = selected[(tau_selected >= edges[row]) & (tau_selected < edges[row + 1])]
            n = min(n_cols, int(in_bin.size))
            chosen = rng.choice(in_bin, size=n, replace=False).astype(int).tolist() if n > 0 else []
            sampled_by_bin.append(chosen)
            bin_labels.append(f"{edges[row]:.3g}-{edges[row + 1]:.3g}")
    else:
        order = selected[np.argsort(fit.tau[selected], kind="mergesort")]
        splits = np.array_split(order, n_rows)
        for row, part in enumerate(splits):
            n = min(n_cols, int(part.size))
            chosen = rng.choice(part, size=n, replace=False).astype(int).tolist() if n > 0 else []
            sampled_by_bin.append(chosen)
            bin_labels.append(f"Q{row + 1}")
    sampled = [idx for row_sample in sampled_by_bin for idx in row_sample]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(8.0, n_cols * 3.7), max(4.0, n_rows * 2.1)), sharex=True, squeeze=False)
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.tick_params(labelsize=7.5, length=2.5, width=0.7)
            style_axis(ax)
            if col >= len(sampled_by_bin[row]):
                ax.axis("off")
                continue
            idx = sampled_by_bin[row][col]
            fitted = sigmoid_loss(
                axis_values,
                fit.tau[idx : idx + 1],
                fit.lambda_[idx : idx + 1],
                fit.start[idx : idx + 1],
                fit.end[idx : idx + 1],
            )[0]
            start_step, end_step = _sigmoid_drop_interval(float(fit.tau[idx]), float(fit.lambda_[idx]), axis_values)
            ax.axvspan(start_step, end_step, color=SHADE_COLOR, alpha=0.12, linewidth=0)
            if loss_matrix is not None:
                ax.plot(axis_values, loss_matrix[idx], color=LOSS_COLOR, linewidth=1.35)
            ax.plot(axis_values, fitted, color=FIT_COLOR, linewidth=1.15, linestyle="--")
            title_prefix = ""
            if label_values is not None:
                value = label_values[idx].item() if isinstance(label_values[idx], np.generic) else label_values[idx]
                label_text = _format_group_label(value, label_prefix)
                title_prefix = f"{label_text}: "
                mean_curve = label_means.get(value)
                if mean_curve is not None:
                    ax.plot(axis_values, mean_curve, color=MEAN_COLOR, linewidth=1.05, linestyle="--")
            if title_values is not None:
                title = f"{title_prefix}{_short_title(title_values[idx], max_title_chars)}"
            else:
                title = f"{title_prefix}tau={fit.tau[idx]:.3g}"
            title = _append_r2_to_title(title, float(fit.r2[idx]))
            ax.set_title(title, fontsize=8.2, pad=3.0)
            if col == 0:
                ax.set_ylabel(f"{bin_labels[row]}\nLoss", fontsize=9.2)
            if row == n_rows - 1:
                ax.set_xlabel(axis_name)

    handles = [
        Line2D([0], [0], color=LOSS_COLOR, linewidth=1.6, label="True loss"),
        Line2D([0], [0], color=FIT_COLOR, linewidth=1.4, linestyle="--", label="Sigmoid fit"),
    ]
    if label_values is not None and loss_matrix is not None:
        handles.append(Line2D([0], [0], color=MEAN_COLOR, linewidth=1.4, linestyle="--", label="Mean loss of label"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.996))
    fig.tight_layout(rect=(0.012, 0.012, 0.995, 0.986), h_pad=1.2, w_pad=0.7)
    _finish(fig, path)
    return sampled


def _pick_evenly_by_tau(indices: np.ndarray, tau: np.ndarray, count: int) -> list[int]:
    ordered = np.asarray(indices, dtype=np.int64)[np.argsort(tau[indices], kind="mergesort")]
    if ordered.size <= count:
        return ordered.astype(int).tolist()
    positions = np.rint(np.linspace(0, ordered.size - 1, count)).astype(np.int64)
    selected = []
    seen = set()
    for pos in positions:
        idx = int(ordered[pos])
        selected.append(idx)
        seen.add(idx)
    if len(selected) < count:
        for idx in ordered:
            idx_int = int(idx)
            if idx_int in seen:
                continue
            selected.append(idx_int)
            if len(selected) >= count:
                break
    return selected[:count]


def plot_token_examples_by_label(
    path: str | Path,
    axis_values: np.ndarray,
    fit: TokenEventFit,
    mask: np.ndarray,
    labels: np.ndarray,
    *,
    loss_matrix: np.ndarray,
    titles: np.ndarray | None = None,
    label_prefix: str = "",
    samples_per_label: int = 5,
    max_labels: int | None = None,
    axis_name: str = "axis",
    label_mean_window: int = 3,
    max_title_chars: int = 52,
) -> list[int]:
    configure_plot_style()
    label_values = np.asarray(labels).reshape(-1)
    if label_values.shape[0] != fit.n_tokens:
        raise ValueError(f"labels length must match number of tokens: {label_values.shape[0]} vs {fit.n_tokens}")
    title_values = None if titles is None else np.asarray(titles).reshape(-1)
    if title_values is not None and title_values.shape[0] != fit.n_tokens:
        raise ValueError(f"titles length must match number of tokens: {title_values.shape[0]} vs {fit.n_tokens}")
    group_values = _ordered_unique(label_values)
    if max_labels is not None:
        group_values = group_values[: int(max_labels)]
    n_rows = int(len(group_values))
    n_cols = int(samples_per_label)
    if n_rows == 0 or n_cols <= 0:
        raise ValueError("Need at least one label and one sample per label.")

    label_means = {
        value.item() if isinstance(value, np.generic) else value: _moving_average(
            np.mean(np.asarray(loss_matrix)[label_values == value], axis=0),
            label_mean_window,
        )
        for value in group_values
    }

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(8.0, n_cols * 3.7), max(3.0, n_rows * 2.1)), sharex=True, squeeze=False)
    sampled: list[int] = []
    for row, value in enumerate(group_values):
        candidates = np.flatnonzero(mask & (label_values == value))
        chosen = _pick_evenly_by_tau(candidates, fit.tau, n_cols) if candidates.size else []
        label_key = value.item() if isinstance(value, np.generic) else value
        label_text = _format_group_label(label_key, label_prefix)
        for col in range(n_cols):
            ax = axes[row, col]
            ax.tick_params(labelsize=7.5, length=2.5, width=0.7)
            style_axis(ax)
            if col >= len(chosen):
                ax.axis("off")
                continue
            idx = int(chosen[col])
            sampled.append(idx)
            fitted = sigmoid_loss(
                axis_values,
                fit.tau[idx : idx + 1],
                fit.lambda_[idx : idx + 1],
                fit.start[idx : idx + 1],
                fit.end[idx : idx + 1],
            )[0]
            start_step, end_step = _sigmoid_drop_interval(float(fit.tau[idx]), float(fit.lambda_[idx]), axis_values)
            ax.axvspan(start_step, end_step, color=SHADE_COLOR, alpha=0.12, linewidth=0)
            ax.plot(axis_values, loss_matrix[idx], color=LOSS_COLOR, linewidth=1.35)
            ax.plot(axis_values, fitted, color=FIT_COLOR, linewidth=1.15, linestyle="--")
            ax.plot(axis_values, label_means[label_key], color=MEAN_COLOR, linewidth=1.05, linestyle="--")
            if title_values is not None:
                title = _short_title(title_values[idx], max_title_chars)
            else:
                title = f"tau={fit.tau[idx]:.3g}"
            title = _append_r2_to_title(title, float(fit.r2[idx]))
            ax.set_title(title, fontsize=8.2, pad=3.0)
            if col == 0:
                ax.set_ylabel(f"{label_text}\nLoss", fontsize=9.2)
            if row == n_rows - 1:
                ax.set_xlabel(axis_name)

    handles = [
        Line2D([0], [0], color=LOSS_COLOR, linewidth=1.6, label="True loss"),
        Line2D([0], [0], color=FIT_COLOR, linewidth=1.4, linestyle="--", label="Sigmoid fit"),
        Line2D([0], [0], color=MEAN_COLOR, linewidth=1.4, linestyle="--", label="Mean loss of label"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.996))
    fig.tight_layout(rect=(0.012, 0.012, 0.995, 0.986), h_pad=1.2, w_pad=0.7)
    _finish(fig, path)
    return sampled


def plot_spectrum(
    path: str | Path,
    spectrum: Spectrum,
    *,
    axis_name: str = "axis",
    curve: str = "pos_mass",
) -> None:
    configure_plot_style()
    y, ylabel = _select_spectrum_curve(spectrum, curve)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(
        spectrum.bin_centers,
        y,
        color=SPECTRUM_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=2.8,
        label=r"Learning-time spectrum $p(\tau)$",
    )
    ax.set_title("Learning-time spectrum")
    ax.set_xlabel(rf"Learning time $\tau$ ({axis_name})")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)
    _finish(fig, path)


def plot_rate_vs_spectrum(
    path: str | Path,
    spectrum: Spectrum,
    empirical_axis: np.ndarray,
    empirical_rate: np.ndarray,
    *,
    axis_name: str = "axis",
    curve: str = "pos_mass",
) -> None:
    configure_plot_style()
    p_tau, _ = _select_spectrum_curve(spectrum, curve)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(empirical_axis, empirical_rate, color=LOSS_COLOR, linewidth=2.2, label="True loss decrease rate")
    ax.set_title("Loss decrease rate vs. learning-time spectrum")
    ax.set_xlabel(axis_name)
    ax.set_ylabel("Loss decrease rate", color=LOSS_COLOR)
    ax.tick_params(axis="y", labelcolor=LOSS_COLOR)
    ax.set_ylim(*padded_limits(empirical_rate, frac=0.08))
    style_axis(ax)

    ax2 = ax.twinx()
    ax2.plot(
        spectrum.bin_centers,
        p_tau,
        color=SPECTRUM_COLOR,
        linewidth=2.2,
        linestyle="--",
        marker="o",
        markersize=2.5,
        label=r"Learning-time spectrum $p(\tau)$",
    )
    ax2.set_ylabel(r"Learning-time spectrum $p(\tau)$", color=SPECTRUM_COLOR)
    ax2.tick_params(axis="y", labelcolor=SPECTRUM_COLOR, length=3.0, width=0.75)
    ax2.set_ylim(*padded_limits(p_tau, frac=0.08))
    for spine in ax2.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color(SPINE_COLOR)
    ax.legend(
        handles=[
            Line2D([0], [0], color=LOSS_COLOR, linewidth=2.2, label="True loss decrease rate"),
            Line2D([0], [0], color=SPECTRUM_COLOR, linewidth=2.2, linestyle="--", label=r"Learning-time spectrum $p(\tau)$"),
        ],
        frameon=False,
        loc="upper right",
    )
    _finish(fig, path)


def plot_binned_pulses(path: str | Path, pulses: BinnedPulses, *, axis_name: str = "axis") -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    n_bins = int(pulses.pulse_by_bin.shape[0])
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, max(n_bins, 1)))
    for color, row in zip(colors, pulses.pulse_by_bin, strict=False):
        ax.plot(pulses.offset_grid, row, color=color, alpha=0.85, linewidth=0.95)
    ax.set_title(r"Binned learning pulse $g(t, \tau)$")
    if pulses.aligned_to_tau:
        ax.set_xlabel(rf"Offset from $\tau$ ({axis_name})")
    else:
        ax.set_xlabel(axis_name)
    ax.set_ylabel("Learning pulse")
    style_axis(ax)
    if n_bins > 1:
        norm = Normalize(vmin=1, vmax=n_bins)
        sm = ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.050, pad=0.035)
        cbar.set_label(r"$\tau$ bin")
        cbar.outline.set_linewidth(0.75)
    _finish(fig, path)


def plot_rate_reconstruction(
    path: str | Path,
    rate_axis: np.ndarray,
    empirical_rate: np.ndarray | None,
    reconstructed_rate: np.ndarray,
    *,
    sigmoid_rate: np.ndarray | None = None,
    empirical_axis: np.ndarray | None = None,
    sigmoid_axis: np.ndarray | None = None,
    axis_name: str = "axis",
    loglog: bool = False,
) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    y_for_limits = [np.asarray(reconstructed_rate, dtype=np.float64)]

    def plot_curve(x_values: np.ndarray, y_values: np.ndarray, **kwargs: Any) -> None:
        x_arr = np.asarray(x_values, dtype=np.float64)
        y_arr = np.asarray(y_values, dtype=np.float64)
        if loglog:
            keep = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0.0) & (y_arr > 0.0)
            if np.any(keep):
                ax.loglog(x_arr[keep], y_arr[keep], **kwargs)
        else:
            ax.plot(x_arr, y_arr, **kwargs)

    if empirical_rate is not None:
        x_emp = empirical_axis if empirical_axis is not None else rate_axis
        plot_curve(x_emp, empirical_rate, color=LOSS_COLOR, linewidth=2.1, label="True loss decrease rate")
        y_for_limits.append(np.asarray(empirical_rate, dtype=np.float64))
    if sigmoid_rate is not None:
        x_sig = sigmoid_axis if sigmoid_axis is not None else rate_axis
        plot_curve(x_sig, sigmoid_rate, color=FIT_COLOR, linewidth=1.9, linestyle="--", label="Sigmoid-implied rate")
        y_for_limits.append(np.asarray(sigmoid_rate, dtype=np.float64))
    plot_curve(rate_axis, reconstructed_rate, color=SPECTRUM_COLOR, linewidth=2.3, linestyle="-.", label=r"$p(\tau)$ dominated rate")
    ax.set_title("Loss decrease rate reconstruction" + (" (log-log)" if loglog else ""))
    ax.set_xlabel(axis_name)
    ax.set_ylabel("Loss decrease rate")
    if loglog:
        positive = np.concatenate(
            [
                np.asarray(values, dtype=np.float64)[np.isfinite(values) & (np.asarray(values, dtype=np.float64) > 0.0)]
                for values in y_for_limits
            ]
        )
        if positive.size:
            ax.set_ylim(10 ** np.floor(np.log10(np.min(positive)) - 0.05), 10 ** np.ceil(np.log10(np.max(positive)) + 0.05))
    else:
        ax.set_ylim(*padded_limits(np.concatenate(y_for_limits), frac=0.08))
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)
    _finish(fig, path)


def plot_reconstruction_summary(
    path: str | Path,
    axis_values: np.ndarray,
    raw_mean: np.ndarray | None,
    sigmoid_mean: np.ndarray | None,
    spectrum: Spectrum,
    rate_axis_dense: np.ndarray,
    empirical_rate: np.ndarray | None,
    reconstructed_rate_dense: np.ndarray,
    *,
    empirical_axis: np.ndarray | None = None,
    sigmoid_rate: np.ndarray | None = None,
    sigmoid_axis: np.ndarray | None = None,
    axis_name: str = "axis",
    spectrum_curve: str = "pos_mass",
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), constrained_layout=True)

    ax = axes[0]
    if raw_mean is not None:
        ax.plot(axis_values, raw_mean, color=LOSS_COLOR, linewidth=2.3, label="Mean validation loss")
    if sigmoid_mean is not None:
        ax.plot(axis_values, sigmoid_mean, color=FIT_COLOR, linewidth=2.1, linestyle="--", label="Mean sigmoid fit")
    ax.set_title("Mean validation loss")
    ax.set_xlabel(axis_name)
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    if empirical_rate is not None:
        x_emp = empirical_axis if empirical_axis is not None else rate_axis_dense
        ax.plot(x_emp, empirical_rate, color=LOSS_COLOR, linewidth=2.2, label="True loss decrease rate")
        ax.set_ylim(*padded_limits(empirical_rate, frac=0.08))
    ax.set_title("Loss decrease rate vs. learning-time spectrum")
    ax.set_xlabel(axis_name)
    ax.set_ylabel("Loss decrease rate", color=LOSS_COLOR)
    ax.tick_params(axis="y", labelcolor=LOSS_COLOR)
    p_tau, _ = _select_spectrum_curve(spectrum, spectrum_curve)
    ax2 = ax.twinx()
    ax2.plot(
        spectrum.bin_centers,
        p_tau,
        color=SPECTRUM_COLOR,
        linewidth=2.2,
        linestyle="--",
        marker="o",
        markersize=2.6,
        label=r"Learning-time spectrum $p(\tau)$",
    )
    ax2.set_ylabel(r"Learning-time spectrum $p(\tau)$", color=SPECTRUM_COLOR)
    ax2.tick_params(axis="y", labelcolor=SPECTRUM_COLOR, length=3.0, width=0.75)
    ax2.set_ylim(*padded_limits(p_tau, frac=0.08))
    for spine in ax2.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color(SPINE_COLOR)
    ax.legend(
        handles=[
            Line2D([0], [0], color=LOSS_COLOR, linewidth=2.2, label="True loss decrease rate"),
            Line2D([0], [0], color=SPECTRUM_COLOR, linewidth=2.2, linestyle="--", label=r"Learning-time spectrum $p(\tau)$"),
        ],
        frameon=False,
        loc="upper right",
    )

    ax = axes[2]
    y_for_limits = [np.asarray(reconstructed_rate_dense, dtype=np.float64)]
    if empirical_rate is not None:
        x_emp = empirical_axis if empirical_axis is not None else rate_axis_dense
        ax.plot(x_emp, empirical_rate, color=LOSS_COLOR, linewidth=2.2, label="True loss decrease rate")
        y_for_limits.append(np.asarray(empirical_rate, dtype=np.float64))
    if sigmoid_rate is not None:
        x_sig = sigmoid_axis if sigmoid_axis is not None else rate_axis_dense
        ax.plot(x_sig, sigmoid_rate, color=FIT_COLOR, linewidth=2.0, linestyle="--", label="Sigmoid-implied rate")
        y_for_limits.append(np.asarray(sigmoid_rate, dtype=np.float64))
    ax.plot(rate_axis_dense, reconstructed_rate_dense, color=SPECTRUM_COLOR, linewidth=2.4, linestyle="-.", label=r"$p(\tau)$ dominated rate")
    ax.set_title("Loss decrease rate reconstruction")
    ax.set_xlabel(axis_name)
    ax.set_ylabel("Loss decrease rate")
    ax.set_ylim(*padded_limits(np.concatenate(y_for_limits), frac=0.08))
    ax.legend(frameon=False, loc="upper right")

    for styled_ax in axes:
        style_axis(styled_ax)
    _finish(fig, path)
