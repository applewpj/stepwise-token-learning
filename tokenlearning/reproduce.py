"""Reproduce the paper figures from the released loss matrices."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tokenlearning.config import apply_overrides, load_config
from tokenlearning.hub import DEFAULT_REPO_ID, MANIFEST_RELPATH, ensure_example, find_repo_root, load_manifest, manifest_entries

REFERENCE_RELPATH = Path("configs") / "reference_metrics.yaml"
QUICK_SETTINGS = {"max_tokens": 50_000, "epochs": 2_000}


@dataclass(frozen=True)
class Target:
    name: str
    asset: str
    title: str
    panels: tuple[tuple[str, str], ...]
    paper_figure: str | None = None
    quick: dict[str, int] | None = field(default_factory=lambda: dict(QUICK_SETTINGS))


TARGETS: dict[str, Target] = {
    "fig3": Target(
        "fig3",
        "step_axis_1m",
        "Figure 3, training steps T (1.5B dense model, 300B tokens)",
        (
            ("A", "mean_loss_raw_vs_sigmoid.png"),
            ("B", "binned_learning_pulses.png"),
            ("C", "learning_time_spectrum.png"),
            ("D", "rate_reconstruction.png"),
            ("D, log-log", "rate_reconstruction_loglog.png"),
        ),
        paper_figure="assets/t_axis.png",
    ),
    "fig4": Target(
        "fig4",
        "d_axis_1m",
        "Figure 4, data scale D (290M MoE, 1B-35B tokens)",
        (
            ("B", "mean_loss_raw_vs_sigmoid.png"),
            ("C", "rate_reconstruction_loglog.png"),
            ("C, inset", "learning_time_spectrum.png"),
        ),
        paper_figure="assets/d_axis.png",
    ),
    "fig5": Target(
        "fig5",
        "m_axis_1m",
        "Figure 5, model scale M (60 models, 0.42-5.00 GFLOPs/token)",
        (
            ("B", "mean_loss_raw_vs_sigmoid.png"),
            ("C", "rate_reconstruction_loglog.png"),
            ("C, inset", "learning_time_spectrum.png"),
        ),
        paper_figure="assets/m_axis.png",
    ),
    "synthetic": Target(
        "synthetic",
        "synthetic_mano_uniform_validation_10k",
        "Appendix, synthetic Mano with uniform difficulty",
        (
            ("token gallery", "token_examples_by_difficulty.png"),
            ("loss by level", "mean_loss_by_difficulty.png"),
            ("reconstruction", "rate_reconstruction.png"),
        ),
        quick=None,
    ),
}

ALIASES = {
    "3": "fig3",
    "t": "fig3",
    "steps": "fig3",
    "step_axis_1m": "fig3",
    "4": "fig4",
    "d": "fig4",
    "data": "fig4",
    "d_axis_1m": "fig4",
    "5": "fig5",
    "m": "fig5",
    "model": "fig5",
    "m_axis_1m": "fig5",
    "mano": "synthetic",
    "appendix": "synthetic",
    "synthetic_mano_uniform_validation_10k": "synthetic",
}

def _number(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"


CHECKED_METRICS = (
    ("kept_fraction", "tokens kept by the event filter", "{:.1%}".format),
    ("tau_median_filtered", "median learning time of kept tokens", _number),
    ("mean_loss_r2", "R^2, sigmoid-implied vs measured mean loss", "{:.4f}".format),
    ("sigmoid_rate_reconstruction_corr", "corr, spectrum-dominated vs sigmoid-implied L'", "{:.3f}".format),
)
DEFAULT_TOLERANCE = {"abs": 0.02}


def resolve_targets(names: list[str]) -> list[Target]:
    resolved: list[Target] = []
    for raw in names:
        key = raw.lower()
        if key == "all":
            candidates = list(TARGETS.values())
        elif key in TARGETS or key in ALIASES:
            candidates = [TARGETS[ALIASES.get(key, key)]]
        else:
            raise SystemExit(f"Unknown target {raw!r}. Choose from: {', '.join(TARGETS)}, all.")
        resolved.extend(t for t in candidates if t not in resolved)
    return resolved


def _display(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def load_reference(repo_root: Path) -> dict[str, Any]:
    path = repo_root / REFERENCE_RELPATH
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def summary_values(metrics: dict[str, Any]) -> dict[str, float]:
    values = {key: float(metrics[key]) for key, _, _ in CHECKED_METRICS if metrics.get(key) is not None}
    if metrics.get("n_tokens"):
        values["kept_fraction"] = metrics["n_filtered_tokens"] / metrics["n_tokens"]
    return values


def _allowed_difference(reference_value: float, spec: dict[str, float]) -> tuple[float, str | None]:
    if "rel" in spec:
        return abs(reference_value) * float(spec["rel"]), f"{float(spec['rel']):.0%}"
    return float(spec["abs"]), None


def compare_with_reference(
    metrics: dict[str, Any],
    reference: dict[str, Any] | None,
    tolerance: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    if not reference:
        return []
    values = summary_values(metrics)
    rows = []
    for key, label, fmt in CHECKED_METRICS:
        if key not in reference or values.get(key) is None:
            continue
        ref = float(reference[key])
        allowed, relative = _allowed_difference(ref, tolerance.get(key, DEFAULT_TOLERANCE))
        rows.append(
            {
                "key": key,
                "label": label,
                "value": values[key],
                "reference": ref,
                "allowed": allowed,
                "text": f"{fmt(values[key]):>8}  reference {fmt(ref)} +/- {relative or fmt(allowed)}",
                "ok": abs(values[key] - ref) <= allowed,
            }
        )
    return rows


def _print_result(target: Target, mode: str, metrics: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    print(f"\nresult    {metrics['n_filtered_tokens']:,} of {metrics['n_tokens']:,} fitted tokens pass the event filter")
    by_key = {row["key"]: row for row in checks}
    values = summary_values(metrics)
    for key, label, fmt in CHECKED_METRICS:
        if key in by_key:
            row = by_key[key]
            print(f"check     {label:<48} {row['text']}  {'ok' if row['ok'] else 'DIFFERS'}")
        elif key in values:
            print(f"          {label:<48} {fmt(values[key]):>8}")
    if not checks:
        print("reference " + ("not checked: custom settings" if mode == "custom" else "none recorded for this setting"))
    fit_time = "reused from cache" if metrics.get("fit_reused_cache") else _format_seconds(metrics.get("fit_seconds"))
    print(f"time      fit {fit_time}, total {_format_seconds(metrics.get('total_seconds'))}")
    fig_dir = Path(metrics["output_dir"]) / "figures"
    print(f"figures   {_display(fig_dir)}{os.sep}")
    for panel, filename in target.panels:
        if (fig_dir / filename).exists():
            print(f"          {panel:<15} {filename}")
    if target.paper_figure:
        print(f"paper     {target.paper_figure}")


def _run_settings(target: Target, quick: bool, epochs: int | None, max_tokens: int | None) -> tuple[str, dict[str, int]]:
    settings: dict[str, int] = {}
    mode = "full"
    if quick and target.quick:
        settings.update(target.quick)
        mode = "quick"
    if epochs is not None:
        settings["epochs"] = epochs
    if max_tokens is not None:
        settings["max_tokens"] = max_tokens
    if (epochs is not None or max_tokens is not None) and settings != (target.quick if mode == "quick" else {}):
        mode = "custom"
    return mode, settings


def reproduce(
    names: list[str],
    *,
    quick: bool = False,
    device: str | None = None,
    epochs: int | None = None,
    max_tokens: int | None = None,
    subset_seed: int | None = None,
    output_dir: str | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    from tokenlearning.pipeline import resolve_device, run_analysis

    root = find_repo_root()
    entries = manifest_entries(load_manifest(root / MANIFEST_RELPATH))
    reference = load_reference(root)
    tolerance = reference.get("tolerance", {})
    targets = resolve_targets(names)
    results: list[dict[str, Any]] = []

    for target in targets:
        entry = entries[target.asset]
        mode, settings = _run_settings(target, quick, epochs, max_tokens)
        config = load_config(root / str(entry["config"]))
        if output_dir is not None:
            out_dir = Path(output_dir).resolve() / (target.asset if len(targets) > 1 else "")
        else:
            out_dir = root / "runs" / (target.asset + ("" if mode == "full" else f"_{mode}"))
        config = apply_overrides(
            config,
            device=device,
            epochs=settings.get("epochs"),
            max_tokens=settings.get("max_tokens"),
            subset_seed=subset_seed,
            output_dir=out_dir,
        )
        config["input"]["losses_npz"] = str(root / str(entry["local_path"]))
        n_rows, n_points = (int(v) for v in entry["loss_matrix_shape"])
        n_fit = min(int(config["input"].get("max_tokens") or n_rows), n_rows)
        fit_device = resolve_device(config.get("fit", {}).get("device"))

        print(f"\n== {target.name}: {target.title} ==", flush=True)
        print(f"data      {entry['local_path']}  ({n_rows:,} tokens x {n_points} points, {float(entry['size_bytes']) / 1e6:,.1f} MB)")
        subset_note = "all tokens" if n_fit == n_rows else f"random {n_fit:,} of {n_rows:,} tokens"
        print(f"setting   {mode}: {subset_note}, {int(config['fit'].get('epochs', 0)):,} epochs, device {fit_device}")
        print(f"output    {_display(out_dir)}", flush=True)
        if mode == "full" and fit_device == "cpu" and target.quick:
            print("note      the paper setting takes hours on CPU; --quick draws the same panels from 50,000 tokens in about a minute")
        if dry_run:
            local = root / str(entry["local_path"])
            print(f"status    {'data present' if local.exists() else 'data will be downloaded'}; dry run, nothing fitted")
            continue

        ensure_example(entry, root, repo_id=repo_id, revision=revision)
        metrics = run_analysis(config)
        checks = compare_with_reference(metrics, reference.get(target.asset, {}).get(mode), tolerance)

        _print_result(target, mode, metrics, checks)
        results.append({"target": target.name, "mode": mode, "metrics": metrics, "checks": checks})

    if len(results) > 1:
        print("\nsummary")
        for item in results:
            if not item["checks"]:
                status = "no reference values"
            elif all(c["ok"] for c in item["checks"]):
                status = f"matches reference ({len(item['checks'])} checks)"
            else:
                status = "DIFFERS from reference: " + ", ".join(c["key"] for c in item["checks"] if not c["ok"])
            print(f"  {item['target']:<10} {item['mode']:<7} {_format_seconds(item['metrics'].get('total_seconds')):>7}  {status}")
    return results


def describe_targets(repo_root: Path | None = None) -> str:
    entries: dict[str, dict[str, Any]] = {}
    try:
        root = repo_root or find_repo_root()
        entries = manifest_entries(load_manifest(root / MANIFEST_RELPATH))
    except FileNotFoundError:
        root = None
    lines = ["Targets:"]
    for target in TARGETS.values():
        entry = entries.get(target.asset, {})
        size = f"{float(entry['size_bytes']) / 1e6:,.0f} MB" if entry else ""
        present = ""
        if root is not None and entry:
            present = "downloaded" if (root / str(entry["local_path"])).exists() else "not downloaded"
        lines.append(f"  {target.name:<10} {target.title:<62} {size:>7}  {present}")
    lines.append("  all        every target above")
    return "\n".join(lines)
