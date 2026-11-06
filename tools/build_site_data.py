#!/usr/bin/env python3
"""Export the data behind the project page's live figures to docs/static/data/.

    python tools/build_site_data.py              # t.json, d.json, m.json and synthetic.json
    python tools/build_site_data.py --only D     # a single file

Real axes read `runs/<asset>` (paper setting) when it exists and `runs/<asset>_quick`
otherwise; the synthetic file reads `runs/synthetic_mano_uniform_validation_10k`. Run
`tokenlearning reproduce fig3 fig4 fig5 synthetic` (optionally with `--quick`) first.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import operator
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tokenlearning.core.sigmoid import sigmoid_loss  # noqa: E402
from tokenlearning.io.loss_matrix import load_loss_npz  # noqa: E402

# Grouped views and clear-drop tokens use only the interior of each axis: tokens already learned
# at the first point pile up there, and steps cut off by the last point are fitted from half a
# transition. E is the irreducible loss of the power law in the paper's figures.
AXES = {
    "T": {"asset": "step_axis_1m", "window": (25000, 60000), "E": 2.21},
    "D": {"asset": "d_axis_1m", "window": (2.5, 25), "E": 2.50},
    "M": {"asset": "m_axis_1m", "window": (0.6, 3.5), "E": 2.11},
}
SYNTHETIC = "synthetic_mano_uniform_validation_10k"

GROUPS = 18
# Each group takes the middle half of its slot of the window, so neighbouring groups stay apart.
BIN_FRACTION = 0.5
CURVE_POINTS = 101
PULSE_POOL = 3
DENSE_STRIDE = 3

TOKEN_INTERVALS = 10
TOKENS_PER_INTERVAL = 16
RANDOM_TOKENS = 160
# Clear drops follow the appendix gallery: R^2 >= 0.5, a drop of at least 0.3 nats, and the
# sharpest tenth of each learning-time interval.
CLEAR_MIN_R2 = 0.5
CLEAR_MIN_DELTA = 0.3
CLEAR_SHARPEST = 0.1

MANO_MOD = 23
MANO_EDGES = (5000, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000)
MANO_PER_COLUMN = 30
MANO_RANDOM = 120


def _round(values, digits: int = 5) -> list:
    return [float(f"{v:.{digits}g}") if np.isfinite(v) else None for v in np.asarray(values, dtype=np.float64).ravel()]


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _quantize_rows(rows: np.ndarray) -> dict:
    """Every row as uint8 between its own minimum and maximum."""
    rows = np.asarray(rows, dtype=np.float64)
    lo = np.round(rows.min(axis=1), 4)
    hi = np.round(rows.max(axis=1), 4)
    span = np.maximum(hi - lo, 1e-6)
    q = np.clip(np.round((rows - lo[:, None]) / span[:, None] * 255), 0, 255).astype(np.uint8)
    return {"lo": lo.tolist(), "hi": hi.tolist(), "u8": _b64(q)}


def _run_dir(asset: str) -> Path:
    for name in (asset, f"{asset}_quick"):
        run = REPO_ROOT / "runs" / name
        if all((run / p).exists() for p in ("metrics.json", "cache/fit_params.npz", "cache/analysis_arrays.npz")):
            return run
    raise SystemExit(f"No finished run for {asset}. Run `tokenlearning reproduce` for it first.")


def fit_power_law(x: np.ndarray, y: np.ndarray, E: float) -> dict:
    """L(a) = E + k a^-alpha for a given irreducible loss E, by least squares on log(L - E)."""
    design = np.vstack([np.ones_like(x), np.log(x)]).T
    coef = np.linalg.lstsq(design, np.log(y - E), rcond=None)[0]
    return {"E": E, "k": float(f"{np.exp(coef[0]):.6g}"), "alpha": round(float(-coef[1]), 4)}


def fit_spectrum_power(x: np.ndarray, mass: np.ndarray) -> dict:
    """p(tau) = c tau^-beta in log-log, leaving out the two edge bins where learning times pile up."""
    keep = mass > 0
    keep[0] = keep[-1] = False
    slope, intercept = np.polyfit(np.log(x[keep]), np.log(mass[keep]), 1)
    return {"c": float(f"{np.exp(intercept):.6g}"), "beta": round(float(-slope), 4)}


def export_pulses(arrays) -> dict:
    """The 40 binned learning pulses, averaged over runs of PULSE_POOL grid points, each as uint8 of its peak."""
    grid = np.asarray(arrays["pulse_offset_grid"], dtype=np.float64)
    pulses = np.clip(np.asarray(arrays["pulse_by_bin"], dtype=np.float64), 0.0, None)
    n = grid.size // PULSE_POOL * PULSE_POOL
    x = grid[:n].reshape(-1, PULSE_POOL).mean(axis=1)
    pooled = pulses[:, :n].reshape(pulses.shape[0], -1, PULSE_POOL).mean(axis=2)
    peak = np.maximum(pooled.max(axis=1), 1e-30)
    q = np.round(pooled / peak[:, None] * 255).astype(np.uint8)
    return {"x": _round(x, 6), "edges": _round(arrays["pulse_tau_bin_edges"], 6), "peak": _round(peak, 4), "u8": _b64(q)}


def export_groups(x_all, losses, tau, lam, start, end, chosen, window) -> dict:
    """Mean loss of tokens in narrow learning-time bins, as measured and shifted to each group's tau."""
    lo, hi = window
    cols = (x_all >= lo) & (x_all <= hi)
    slot = (hi - lo) / GROUPS
    centres = lo + slot * (np.arange(GROUPS) + 0.5)
    fit_x = np.linspace(lo, hi, CURVE_POINTS)
    rel = np.linspace(-(hi - lo), hi - lo, CURVE_POINTS)
    out = {"x": _round(x_all[cols], 7), "fit_x": _round(fit_x, 7), "rel": _round(rel, 6),
           "bins": [], "tau": [], "count": [], "mid": [], "raw": [], "fit": [], "aligned_fit": []}
    for c in centres:
        a, b = c - 0.5 * BIN_FRACTION * slot, c + 0.5 * BIN_FRACTION * slot
        members = np.flatnonzero(chosen & (tau >= a) & (tau < b))
        if members.size == 0:
            continue
        t, l, s, e = tau[members], lam[members], start[members], end[members]
        centre = float(np.median(t))
        mid = float(np.mean(0.5 * (s + e)))
        out["bins"].append(_round([a, b], 7))
        out["tau"].append(centre)
        out["count"].append(int(members.size))
        out["mid"].append(round(mid, 5))
        out["raw"].append(_round(losses[members][:, cols].mean(axis=0, dtype=np.float64), 4))
        out["fit"].append(_round(sigmoid_loss(fit_x, t, l, s, e).mean(axis=0), 4))
        out["aligned_fit"].append(_round(sigmoid_loss(centre + rel, t, l, s, e).mean(axis=0) - mid, 4))
    return out


def export_tokens(losses, tau, lam, start, end, r2, valid, window, rng) -> dict:
    """A pool of clear drops per learning-time interval of the window, and a pool of random drops."""
    delta = start - end
    width = 2.0 * np.log(9.0) / lam
    edges = np.linspace(window[0], window[1], TOKEN_INTERVALS + 1)
    eligible = valid & (delta >= CLEAR_MIN_DELTA) & (r2 >= CLEAR_MIN_R2)
    clear: list[int] = []
    interval: list[int] = []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        members = np.flatnonzero(eligible & (tau >= a) & (tau < b))
        keep = max(TOKENS_PER_INTERVAL, int(np.ceil(CLEAR_SHARPEST * members.size)))
        sharp = members[np.argsort(width[members], kind="stable")[:keep]]
        drawn = rng.choice(sharp, min(TOKENS_PER_INTERVAL, sharp.size), replace=False)
        clear.extend(drawn.tolist())
        interval.extend([k] * drawn.size)
    drops = np.setdiff1d(np.flatnonzero(valid & (delta > 0)), clear)
    random = rng.choice(drops, min(RANDOM_TOKENS, drops.size), replace=False)
    idx = np.concatenate([np.asarray(clear, dtype=np.int64), random])
    return {
        "edges": _round(edges, 7),
        "n": int(idx.size),
        "pool": [1] * len(clear) + [0] * int(random.size),
        "interval": interval + [-1] * int(random.size),
        "tau": _round(tau[idx], 6),
        "lambda": _round(lam[idx], 5),
        "start": _round(start[idx], 5),
        "end": _round(end[idx], 5),
        "r2": _round(r2[idx], 3),
        "loss": _quantize_rows(losses[idx]),
        "eligible": int(np.sum(eligible & (tau >= window[0]) & (tau < window[1]))),
    }


def export_axis(key: str, spec: dict, rng: np.random.Generator) -> dict:
    run = _run_dir(spec["asset"])
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    fit = np.load(run / "cache" / "fit_params.npz")
    arrays = np.load(run / "cache" / "analysis_arrays.npz")
    tau, lam, start, end, r2 = (np.asarray(fit[k], dtype=np.float64) for k in ("tau", "lambda_", "start", "end", "r2"))
    valid = np.asarray(fit["valid_mask"], dtype=bool) & np.isfinite(tau) & np.isfinite(lam) & np.isfinite(r2)
    x_all = np.asarray(fit["axis_values"], dtype=np.float64)

    data = load_loss_npz(REPO_ROOT / "examples" / spec["asset"] / "losses.npz")
    subset = metrics.get("subset")
    if subset:
        rows = np.sort(np.random.default_rng(subset["seed"]).choice(subset["n_total"], subset["n_fitted"], replace=False))
        losses = data.loss_matrix[rows]
    else:
        losses = data.loss_matrix
    assert losses.shape[0] == tau.size

    # Every token whose fitted loss goes down, as in Figure 1 of the paper.
    drops = valid & (start > end)
    mean_x = np.asarray(arrays["axis_values"], dtype=np.float64)
    mean_raw = np.asarray(arrays["raw_loss_mean"], dtype=np.float64)
    spec_x = np.asarray(arrays["spectrum_bin_centers"], dtype=np.float64)
    spec_mass = np.asarray(arrays["spectrum_pos_mass"], dtype=np.float64)
    return {
        "key": key,
        "setting": "quick" if subset else "full",
        "window": list(spec["window"]),
        "stats": {"n_total": int(data.loss_matrix.shape[0]), "n_fitted": int(tau.size), "n_drops": int(np.sum(drops))},
        "x": _round(x_all, 7),
        "mean": {
            "x": _round(mean_x, 7),
            "empirical": _round(mean_raw, 6),
            "sigmoid": _round(arrays["fitted_loss_mean"], 6),
            "power": fit_power_law(mean_x, mean_raw, spec["E"]),
        },
        "rate": {
            "x": _round(arrays["rate_axis"], 7),
            "empirical": _round(arrays["empirical_rate"], 4),
            "sigmoid": _round(arrays["sigmoid_rate"], 4),
            "spectrum_x": _round(arrays["rate_axis_dense"][::DENSE_STRIDE], 7),
            "spectrum": _round(arrays["reconstructed_rate_dense"][::DENSE_STRIDE], 4),
        },
        "spectrum": {
            "x": _round(spec_x, 7),
            "edges": _round(arrays["spectrum_bin_edges"], 7),
            "mass": _round(spec_mass, 4),
            "power": fit_spectrum_power(spec_x, spec_mass),
        },
        "pulses": export_pulses(arrays),
        "groups": export_groups(x_all, losses, tau, lam, start, end, drops, spec["window"]),
        "tokens": export_tokens(losses, tau, lam, start, end, r2, valid, spec["window"], rng),
    }


def mano_answer(expression: str) -> int:
    """Value of a Mano expression modulo 23."""
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}

    def value(node):
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        raise ValueError(f"Unexpected Mano expression: {expression}")

    return value(ast.parse(expression, mode="eval").body) % MANO_MOD


def export_synthetic(rng: np.random.Generator) -> dict:
    run = _run_dir(SYNTHETIC)
    fit = np.load(run / "cache" / "fit_params.npz")
    src = np.load(REPO_ROOT / "examples" / SYNTHETIC / "losses.npz")
    tau, lam, start, end, r2 = (np.asarray(fit[k], dtype=np.float64) for k in ("tau", "lambda_", "start", "end", "r2"))
    x = np.asarray(src["axis_values"], dtype=np.float64)
    difficulty = np.asarray(src["difficulty"], dtype=np.int64)
    questions = [str(q) for q in src["question"]]

    # Columns: learned before the first checkpoint, eight intervals of the axis, not learned by the last one.
    edges = np.asarray(MANO_EDGES, dtype=np.float64)
    column = np.searchsorted(edges, tau, side="right")
    n_columns = edges.size + 1
    levels = np.arange(1, difficulty.max() + 1)
    counts = [[int(np.sum((difficulty == d) & (column == c))) for c in range(n_columns)] for d in levels]

    picked: list[int] = []
    for c in range(n_columns):
        members = np.flatnonzero(column == c)
        picked.extend(rng.choice(members, min(MANO_PER_COLUMN, members.size), replace=False).tolist())
    rest = np.setdiff1d(np.arange(tau.size), picked)
    random = rng.choice(rest, MANO_RANDOM, replace=False)
    idx = np.concatenate([np.asarray(picked, dtype=np.int64), random])
    return {
        "x": _round(x, 7),
        "start": round(float(np.median(start)), 5),
        "end": round(float(np.median(end)), 5),
        "edges": _round(edges, 7),
        "levels": levels.tolist(),
        "counts": counts,
        "tokens": {
            "n": int(idx.size),
            "pool": [1] * len(picked) + [0] * int(random.size),
            "column": column[idx].tolist(),
            "difficulty": difficulty[idx].tolist(),
            "question": [questions[i] for i in idx],
            "answer": [mano_answer(questions[i]) for i in idx],
            "tau": _round(tau[idx], 6),
            "lambda": _round(lam[idx], 5),
            "r2": _round(r2[idx], 3),
            "loss": _quantize_rows(np.asarray(src["loss_matrix"], dtype=np.float64)[idx]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "static" / "data")
    parser.add_argument("--only", nargs="*", choices=[*AXES, "synthetic"], default=[*AXES, "synthetic"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for key in args.only:
        rng = np.random.default_rng(2026)
        if key == "synthetic":
            payload, path = export_synthetic(rng), args.out / "synthetic.json"
            note = f"{payload['tokens']['n']} problems"
        else:
            payload, path = export_axis(key, AXES[key], rng), args.out / f"{key.lower()}.json"
            note = (f"{payload['setting']} run, {len(payload['groups']['tau'])} groups, "
                    f"{payload['tokens']['n']} tokens, E = {payload['mean']['power']['E']}")
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        print(f"{path.relative_to(REPO_ROOT)}: {path.stat().st_size / 1024:.0f} KB, {note}")


if __name__ == "__main__":
    main()
