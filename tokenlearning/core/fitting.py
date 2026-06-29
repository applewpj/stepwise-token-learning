from __future__ import annotations

from typing import Any

import numpy as np

from tokenlearning.types import TokenEventFit


def smooth_loss_matrix(loss_matrix: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return np.asarray(loss_matrix, dtype=np.float64)
    y = np.asarray(loss_matrix, dtype=np.float64)
    n_points = y.shape[1]
    idx = np.arange(n_points, dtype=np.int64)
    half = window // 2
    left = np.clip(idx - half, 0, n_points - 1)
    right = np.clip(idx + half, 0, n_points - 1)
    counts = (right - left + 1).astype(np.float64)
    csum = np.concatenate([np.zeros((y.shape[0], 1), dtype=np.float64), np.cumsum(y, axis=1)], axis=1)
    return (csum[:, right + 1] - csum[:, left]) / counts[None, :]


def compute_loss_snr_stats(loss_matrix: np.ndarray, config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    cfg = config or {}
    y = np.asarray(loss_matrix, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"loss_matrix must be 2D, got {y.shape}")
    n_points = y.shape[1]
    smooth_window = int(cfg.get("smooth_window", 3))
    edge_points = int(cfg.get("edge_points", min(3, n_points)))
    edge_points = max(1, min(edge_points, n_points))
    y_smooth = smooth_loss_matrix(y, smooth_window)
    start_mean = np.mean(y_smooth[:, :edge_points], axis=1)
    end_mean = np.mean(y_smooth[:, -edge_points:], axis=1)
    delta = start_mean - end_mean
    residual = y - y_smooth
    noise_std = np.std(residual, axis=1)
    snr = np.abs(delta) / np.maximum(noise_std, 1e-8)
    return {
        "snr": snr.astype(np.float64),
        "delta": delta.astype(np.float64),
        "noise_std": noise_std.astype(np.float64),
    }


def loss_snr_keep_mask(loss_matrix: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = config or {}
    if not cfg or not bool(cfg.get("enabled", False)):
        return np.ones(np.asarray(loss_matrix).shape[0], dtype=bool)
    method = str(cfg.get("method", "snr")).lower()
    if method != "snr":
        raise ValueError(f"Unsupported loss_snr.method={method!r}")
    stats = compute_loss_snr_stats(loss_matrix, cfg)
    min_snr = float(cfg.get("min_snr", 1.0))
    return np.isfinite(stats["snr"]) & (stats["snr"] >= min_snr)


def _initial_tau_scaled(x_scaled: np.ndarray, y: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    delta = start - end
    safe_delta = np.where(np.abs(delta) > 1e-8, delta, np.where(delta >= 0.0, 1e-8, -1e-8))
    progress = (start[:, None] - y) / safe_delta[:, None]
    crossed = progress >= 0.5
    first = np.argmax(crossed, axis=1)
    has_cross = np.any(crossed, axis=1)
    tau = np.full(y.shape[0], 0.5, dtype=np.float32)
    tau[has_cross] = x_scaled[first[has_cross]]
    return np.clip(tau, 1e-4, 1.0 - 1e-4)


def _initial_tau_scaled_fixed_endpoint(
    axis_values: np.ndarray,
    y: np.ndarray,
    *,
    fixed_start: float,
    fixed_end: float,
) -> np.ndarray:
    x = np.asarray(axis_values, dtype=np.float64)
    x_min = float(np.min(x))
    x_range = float(np.max(x) - x_min)
    step = x_range / max(int(x.shape[0]) - 1, 1)
    delta = float(fixed_start) - float(fixed_end)
    safe_delta = delta if abs(delta) > 1e-8 else 1e-8
    midpoint = float(fixed_end) + 0.5 * safe_delta
    crossed = np.asarray(y, dtype=np.float32) < midpoint
    first = np.argmax(crossed, axis=1).astype(np.float64)
    has_cross = np.any(crossed, axis=1)
    already_below = crossed[:, 0]
    never_dropped = ~has_cross
    tau_axis = x_min + first * step
    tau_axis[already_below] = x_min - 0.5 * float(x.shape[0]) * step
    tau_axis[never_dropped] = x_min + 1.5 * float(x.shape[0]) * step
    return ((tau_axis - x_min) / x_range).astype(np.float32)


def _fit_chunk_torch(
    axis_values: np.ndarray,
    y_np: np.ndarray,
    *,
    epochs: int,
    lr: float,
    lambda_init: float,
    device: str,
    edge_points: int,
    constrain_tau: bool,
    max_grad_norm: float | None,
    fixed_start: float | None,
    fixed_end: float | None,
    step_mean_loss_weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(int(seed))
    x = np.asarray(axis_values, dtype=np.float64)
    x_min = float(np.min(x))
    x_range = float(np.max(x) - x_min)
    if x_range <= 0:
        raise ValueError("axis_values must span a positive range.")
    x_scaled_np = ((x - x_min) / x_range).astype(np.float32)
    y_np = np.asarray(y_np, dtype=np.float32)
    n_tokens, n_points = y_np.shape
    edge = max(1, min(int(edge_points), n_points))
    use_fixed_endpoints = fixed_start is not None and fixed_end is not None
    if use_fixed_endpoints:
        start_init = np.full(n_tokens, float(fixed_start), dtype=np.float32)
        end_init = np.full(n_tokens, float(fixed_end), dtype=np.float32)
        delta_init = (start_init - end_init).astype(np.float32)
        tau_init = _initial_tau_scaled_fixed_endpoint(
            axis_values,
            y_np,
            fixed_start=float(fixed_start),
            fixed_end=float(fixed_end),
        )
    else:
        start_init = np.mean(y_np[:, :edge], axis=1).astype(np.float32)
        end_init = np.mean(y_np[:, -edge:], axis=1).astype(np.float32)
        delta_init = (start_init - end_init).astype(np.float32)
        tau_init = _initial_tau_scaled(x_scaled_np, y_np, start_init, end_init)
    lam_init = max(float(lambda_init), 1e-4)

    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    x_scaled = torch.tensor(x_scaled_np, dtype=torch.float32, device=device)[None, :]
    if use_fixed_endpoints:
        end_fixed = torch.tensor(end_init[:, None], dtype=torch.float32, device=device)
        delta_fixed = torch.tensor(delta_init[:, None], dtype=torch.float32, device=device)
        fit_params: list[torch.nn.Parameter] = []
    else:
        end_param = torch.nn.Parameter(torch.tensor(end_init[:, None], dtype=torch.float32, device=device))
        delta_param = torch.nn.Parameter(torch.tensor(delta_init[:, None], dtype=torch.float32, device=device))
        fit_params = [end_param, delta_param]
    if constrain_tau:
        tau_init = np.clip(tau_init, 1e-4, 1.0 - 1e-4)
        tau_logit_init = np.log(tau_init / (1.0 - tau_init)).astype(np.float32)
        tau_param = torch.nn.Parameter(torch.tensor(tau_logit_init[:, None], dtype=torch.float32, device=device))
    else:
        tau_param = torch.nn.Parameter(torch.tensor(tau_init[:, None], dtype=torch.float32, device=device))
    inv_softplus = np.log(np.expm1(lam_init)).astype(np.float32) if lam_init < 20 else np.float32(lam_init)
    lam_param = torch.nn.Parameter(torch.full((n_tokens, 1), float(inv_softplus), dtype=torch.float32, device=device))
    fit_params.extend([tau_param, lam_param])

    optimizer = torch.optim.Adam(fit_params, lr=float(lr))
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        lam = F.softplus(lam_param) + 1e-8
        tau = torch.sigmoid(tau_param) if constrain_tau else tau_param
        if use_fixed_endpoints:
            pred = end_fixed + delta_fixed * (1.0 - torch.sigmoid(lam * (x_scaled - tau)))
        else:
            pred = end_param + delta_param * (1.0 - torch.sigmoid(lam * (x_scaled - tau)))
        loss = F.mse_loss(pred, y)
        if step_mean_loss_weight > 0.0:
            err_step_mean = (pred - y).mean(dim=0)
            loss = loss + float(step_mean_loss_weight) * torch.mean(err_step_mean**2)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(fit_params, float(max_grad_norm))
        optimizer.step()

    with torch.no_grad():
        lam_scaled = (F.softplus(lam_param) + 1e-8).detach().cpu().numpy().reshape(-1)
        tau_scaled = (
            torch.sigmoid(tau_param) if constrain_tau else tau_param
        ).detach().cpu().numpy().reshape(-1)
        if use_fixed_endpoints:
            end = end_fixed.detach().cpu().numpy().reshape(-1)
            delta = delta_fixed.detach().cpu().numpy().reshape(-1)
        else:
            end = end_param.detach().cpu().numpy().reshape(-1)
            delta = delta_param.detach().cpu().numpy().reshape(-1)
        start = end + delta
        pred_np = (
            end[:, None]
            + delta[:, None] * (1.0 - 1.0 / (1.0 + np.exp(-lam_scaled[:, None] * (x_scaled_np[None, :] - tau_scaled[:, None]))))
        )
    rss = np.sum((y_np - pred_np) ** 2, axis=1)
    centered = y_np - np.mean(y_np, axis=1, keepdims=True)
    tss = np.sum(centered**2, axis=1)
    r2 = 1.0 - rss / np.maximum(tss, 1e-8)
    tau = tau_scaled * x_range + x_min
    lambda_axis = lam_scaled / x_range
    return (
        tau.astype(np.float32),
        lambda_axis.astype(np.float32),
        delta.astype(np.float32),
        start.astype(np.float32),
        end.astype(np.float32),
        r2.astype(np.float32),
    )


def _first_crossing_time(t_scaled: np.ndarray, y_smooth: np.ndarray, start: np.ndarray, end: np.ndarray, threshold: float) -> np.ndarray:
    delta = start - end
    delta_safe = np.where(np.abs(delta) > 1e-8, delta, np.ones_like(delta))
    progress = (start - y_smooth) / delta_safe
    progress = np.where(np.abs(delta) > 1e-8, progress, np.zeros_like(progress))
    crossed = progress >= float(threshold)
    first = np.argmax(crossed, axis=1)
    never = ~np.any(crossed, axis=1)
    t_hit = t_scaled[first].astype(np.float64)
    t_hit[never] = float(t_scaled[-1]) * 1.5
    return t_hit.reshape(-1, 1)


def _fit_chunk_plot_compatible(
    axis_values: np.ndarray,
    y_np: np.ndarray,
    *,
    epochs: int,
    lr: float,
    lambda_init: float,
    device: str,
    seed: int,
    max_grad_norm: float | None,
    constrain_sigmoid: dict[str, Any] | None,
    t_div: float,
    log_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    y_np = np.asarray(y_np, dtype=np.float32)
    x = np.asarray(axis_values, dtype=np.float64).ravel()
    n_tokens, n_points = y_np.shape
    if n_points < 2:
        raise ValueError("Need at least two axis points for sigmoid fitting.")

    constrain_sigmoid = constrain_sigmoid or {}
    quantile = float(constrain_sigmoid.get("quantile", 0.5))
    constrain_t_start = dict(constrain_sigmoid.get("constrain_t_start", {}))
    use_t_start = bool(constrain_t_start.get("enabled", False))
    t_start_mode = str(constrain_t_start.get("mode", "in_range"))

    edge = min(3, n_points)
    y_smooth = smooth_loss_matrix(y_np, 3)
    start_init = np.mean(y_smooth[:, :edge], axis=1, keepdims=True).astype(np.float64)
    end_init = np.mean(y_smooth[:, -edge:], axis=1, keepdims=True).astype(np.float64)
    t_scaled = x / float(t_div)
    tau_init = _first_crossing_time(t_scaled, y_smooth, start_init, end_init, 0.5)
    tstart_init = _first_crossing_time(t_scaled, y_smooth, start_init, end_init, quantile)

    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    t = torch.tensor(t_scaled.astype(np.float32), dtype=torch.float32, device=device)[None, :]
    end_t = torch.tensor(end_init, dtype=torch.float32, device=device)
    start_t = torch.tensor(np.clip(start_init, 1e-6, None), dtype=torch.float32, device=device)
    lam_t = torch.full((n_tokens, 1), float(lambda_init), dtype=torch.float32, device=device)

    eps = 1e-6
    end_param = torch.nn.Parameter(torch.log(torch.expm1(torch.clamp(end_t, min=eps))))
    start_param = torch.nn.Parameter(torch.log(torch.expm1(torch.clamp(start_t, min=eps))))
    lam_param = torch.nn.Parameter(torch.log(torch.expm1(torch.clamp(lam_t, min=eps))))

    t_min = torch.min(t).detach()
    t_max = torch.max(t).detach()
    if use_t_start:
        tstart = torch.tensor(tstart_init, dtype=torch.float32, device=device)
        tstart = torch.clamp(tstart, min=t_min, max=t_max)
        if t_start_mode == "in_range":
            denom = torch.clamp(t_max - t_min, min=eps)
            p0 = torch.clamp((tstart - t_min) / denom, min=1e-6, max=1.0 - 1e-6)
            t_param = torch.nn.Parameter(torch.log(p0 / (1.0 - p0)))
        else:
            t_param = torch.nn.Parameter(torch.log(torch.expm1(torch.clamp(tstart, min=eps))))
    else:
        t_param = torch.nn.Parameter(torch.tensor(tau_init, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam([end_param, start_param, lam_param, t_param], lr=float(lr))
    for epoch_idx in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        end = F.softplus(end_param)
        start = F.softplus(start_param)
        delta = start - end
        lam = F.softplus(lam_param)
        if use_t_start:
            if t_start_mode == "in_range":
                tstart = t_min + (t_max - t_min) * torch.sigmoid(t_param)
            else:
                tstart = F.softplus(t_param)
            q = torch.tensor(quantile, dtype=torch.float32, device=device)
            z = lam * (t - tstart) + torch.log(q / (1.0 - q))
        else:
            z = lam * (t - t_param)
        pred = end + delta * (1.0 - torch.sigmoid(torch.clamp(z, -50.0, 50.0)))
        err = pred - y
        loss = err.mean(dim=0).abs().mean() + F.mse_loss(pred, y)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_([end_param, start_param, lam_param, t_param], float(max_grad_norm))
        optimizer.step()
        if log_every > 0 and (epoch_idx + 1) % int(log_every) == 0:
            print(f"[plot-compatible-fit] epoch {epoch_idx + 1}/{epochs} loss={loss.item():.6f}", flush=True)

    with torch.no_grad():
        end = F.softplus(end_param).detach().cpu().numpy().reshape(-1)
        start = F.softplus(start_param).detach().cpu().numpy().reshape(-1)
        lam_scaled = F.softplus(lam_param).detach().cpu().numpy().reshape(-1)
        if use_t_start:
            if t_start_mode == "in_range":
                tstart = (t_min + (t_max - t_min) * torch.sigmoid(t_param)).detach().cpu().numpy().reshape(-1)
            else:
                tstart = F.softplus(t_param).detach().cpu().numpy().reshape(-1)
            z_q = float(np.log(quantile / (1.0 - quantile)))
            tau_scaled = tstart - z_q / np.maximum(lam_scaled, 1e-6)
        else:
            tau_scaled = t_param.detach().cpu().numpy().reshape(-1)

    lambda_axis = lam_scaled / float(t_div)
    tau = tau_scaled * float(t_div)
    delta = start - end
    pred_np = np.empty((n_tokens, n_points), dtype=np.float32)
    for row_start in range(0, n_tokens, 65536):
        row_end = min(row_start + 65536, n_tokens)
        z = lambda_axis[row_start:row_end, None] * (x[None, :] - tau[row_start:row_end, None])
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        pred_np[row_start:row_end] = (
            end[row_start:row_end, None] + delta[row_start:row_end, None] * (1.0 - sig)
        ).astype(np.float32)
    rss = np.sum((y_np - pred_np) ** 2, axis=1)
    centered = y_np - np.mean(y_np, axis=1, keepdims=True)
    tss = np.sum(centered**2, axis=1)
    r2 = 1.0 - rss / np.maximum(tss, 1e-8)
    return (
        tau.astype(np.float32),
        lambda_axis.astype(np.float32),
        delta.astype(np.float32),
        start.astype(np.float32),
        end.astype(np.float32),
        r2.astype(np.float32),
    )


def _plot_postfit_keep_mask(fit: TokenEventFit, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = config or {}
    finite = (
        np.isfinite(fit.tau)
        & np.isfinite(fit.lambda_)
        & np.isfinite(fit.delta)
        & np.isfinite(fit.start)
        & np.isfinite(fit.end)
        & np.isfinite(fit.r2)
        & (fit.lambda_ > 0.0)
    )
    drop = np.zeros_like(finite, dtype=bool)
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
    return finite & (~drop)


def _fit_token_events_plot_compatible(
    axis_values: np.ndarray,
    loss_matrix: np.ndarray,
    config: dict[str, Any],
) -> TokenEventFit:
    epochs = int(config.get("epochs", 10000))
    lr = float(config.get("lr", 0.02))
    chunk_size = int(config.get("chunk_size", -1))
    if chunk_size <= 0:
        chunk_size = int(loss_matrix.shape[0])
    device = str(config.get("device", "cuda"))
    lambda_init = float(config.get("lambda_init", 0.1))
    max_grad_norm = config.get("max_grad_norm")
    t_div = float(config.get("t_div", 1.0))
    constrain_sigmoid = config.get("constrain_sigmoid")
    seed = int(config.get("seed", 1234))
    log_every = int(config.get("log_every", 0))

    losses = np.asarray(loss_matrix, dtype=np.float32)
    pieces: dict[str, list[np.ndarray]] = {
        "tau": [],
        "lambda_": [],
        "delta": [],
        "start": [],
        "end": [],
        "r2": [],
    }
    chunk_count = (losses.shape[0] + chunk_size - 1) // chunk_size
    for chunk_idx, row_start in enumerate(range(0, losses.shape[0], chunk_size), start=1):
        row_end = min(row_start + chunk_size, losses.shape[0])
        print(f"[plot-compatible-fit] chunk {chunk_idx}/{chunk_count} rows={row_start}:{row_end}", flush=True)
        result = _fit_chunk_plot_compatible(
            axis_values,
            losses[row_start:row_end],
            epochs=epochs,
            lr=lr,
            lambda_init=lambda_init,
            device=device,
            seed=seed + chunk_idx - 1,
            max_grad_norm=max_grad_norm,
            constrain_sigmoid=constrain_sigmoid,
            t_div=t_div,
            log_every=log_every,
        )
        for key, value in zip(pieces, result):
            pieces[key].append(value)
    merged = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
    finite = (
        np.isfinite(merged["tau"])
        & np.isfinite(merged["lambda_"])
        & np.isfinite(merged["delta"])
        & np.isfinite(merged["start"])
        & np.isfinite(merged["end"])
        & np.isfinite(merged["r2"])
        & (merged["lambda_"] > 0.0)
    )
    fit = TokenEventFit(valid_mask=finite, **merged)
    fit.validate()
    return fit


def fit_token_events(
    axis_values: np.ndarray,
    loss_matrix: np.ndarray,
    config: dict[str, Any] | None = None,
) -> TokenEventFit:
    cfg = config or {}
    mode = str(cfg.get("mode", cfg.get("fit_mode", "scaled"))).lower()
    if mode in {"plot", "plot_compatible", "plot-compatible"}:
        return _fit_token_events_plot_compatible(axis_values, loss_matrix, cfg)

    epochs = int(cfg.get("epochs", 2000))
    lr = float(cfg.get("lr", 0.03))
    chunk_size = int(cfg.get("chunk_size", 8192))
    if chunk_size <= 0:
        chunk_size = int(loss_matrix.shape[0])
    device = str(cfg.get("device", "cuda"))
    lambda_init = float(cfg.get("lambda_init_scaled", cfg.get("lambda_init", 10.0)))
    edge_points = int(cfg.get("edge_points", 3))
    constrain_tau = bool(cfg.get("constrain_tau", True))
    max_grad_norm = cfg.get("max_grad_norm")
    fixed_start = cfg.get("fixed_start")
    fixed_end = cfg.get("fixed_end")
    fixed_start = None if fixed_start is None else float(fixed_start)
    fixed_end = None if fixed_end is None else float(fixed_end)
    step_mean_loss_weight = float(cfg.get("step_mean_loss_weight", 0.0))
    seed = int(cfg.get("seed", 1234))

    losses = np.asarray(loss_matrix, dtype=np.float32)
    if losses.ndim != 2:
        raise ValueError(f"loss_matrix must be 2D, got {losses.shape}")
    pieces: dict[str, list[np.ndarray]] = {
        "tau": [],
        "lambda_": [],
        "delta": [],
        "start": [],
        "end": [],
        "r2": [],
    }
    for chunk_idx, row_start in enumerate(range(0, losses.shape[0], chunk_size)):
        row_end = min(row_start + chunk_size, losses.shape[0])
        result = _fit_chunk_torch(
            axis_values,
            losses[row_start:row_end],
            epochs=epochs,
            lr=lr,
            lambda_init=lambda_init,
            device=device,
            edge_points=edge_points,
            constrain_tau=constrain_tau,
            max_grad_norm=max_grad_norm,
            fixed_start=fixed_start,
            fixed_end=fixed_end,
            step_mean_loss_weight=step_mean_loss_weight,
            seed=seed + chunk_idx,
        )
        for key, value in zip(pieces, result):
            pieces[key].append(value)
    merged = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
    valid = (
        np.isfinite(merged["tau"])
        & np.isfinite(merged["lambda_"])
        & np.isfinite(merged["delta"])
        & np.isfinite(merged["r2"])
        & (merged["lambda_"] > 0)
    )
    fit = TokenEventFit(valid_mask=valid, **merged)
    fit.validate()
    return fit
