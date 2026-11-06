import numpy as np
import pytest

from tokenlearning.cli import main
from tokenlearning.config import apply_overrides
from tokenlearning.core.sigmoid import sigmoid_loss
from tokenlearning.io.loss_matrix import save_loss_npz
from tokenlearning.pipeline import run_analysis
from tokenlearning.reproduce import _run_settings, resolve_targets
from tokenlearning.types import LossMatrix


def _write_event_matrix(path, n_tokens=400, n_points=16, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(1.0, 16.0, n_points)
    start = rng.uniform(4.0, 6.0, n_tokens)
    y = sigmoid_loss(
        x,
        tau=rng.uniform(4.0, 13.0, n_tokens),
        lambda_=rng.uniform(0.8, 2.0, n_tokens),
        start=start,
        end=start - rng.uniform(1.5, 3.0, n_tokens),
    )
    y = y + rng.normal(0.0, 0.02, y.shape)
    save_loss_npz(path, LossMatrix(axis_values=x, loss_matrix=y.astype(np.float32), axis_name="step"))


def _config(npz, out_dir, **input_overrides):
    return {
        "input": {"losses_npz": str(npz), "axis_name": "step", **input_overrides},
        "output": {"dir": str(out_dir), "reuse_cache": True},
        "fit": {
            "mode": "plot",
            "epochs": 300,
            "lr": 0.05,
            "chunk_size": 1000,
            "device": "cpu",
            "lambda_init": 0.1,
            "t_div": 1.0,
            "log_every": 0,
            "constrain_sigmoid": {"quantile": 0.5, "constrain_t_start": {"enabled": True, "mode": "in_range"}},
        },
        "token_examples": {"tau_bins": 2, "samples_per_bin": 1},
        "filters": {"mode": "plot", "min_abs_delta": 0.3, "min_r2": 0.5},
        "spectrum": {"bins": 8},
        "pulse": {"tau_bins": 4, "offset_grid_points": 99},
        "reconstruction": {"dense_points": 99, "denominator": 400},
    }


def _reconstruction_scale(out_dir):
    arrays = np.load(out_dir / "cache" / "analysis_arrays.npz")
    return float(np.mean(np.abs(arrays["reconstructed_rate"])))


def test_target_aliases():
    assert [t.name for t in resolve_targets(["t", "fig3", "5"])] == ["fig3", "fig5"]
    assert [t.name for t in resolve_targets(["all"])] == ["fig3", "fig4", "fig5", "synthetic"]
    with pytest.raises(SystemExit):
        resolve_targets(["fig9"])


def test_run_settings_modes():
    fig3, synthetic = resolve_targets(["fig3", "synthetic"])
    assert _run_settings(fig3, True, None, None) == ("quick", {"max_tokens": 50_000, "epochs": 2_000})
    assert _run_settings(fig3, False, None, None) == ("full", {})
    assert _run_settings(fig3, True, 500, None)[0] == "custom"
    assert _run_settings(synthetic, True, None, None) == ("full", {})


def test_apply_overrides_returns_copy():
    config = {"fit": {"device": "cuda", "epochs": 10}, "input": {}}
    updated = apply_overrides(config, device="cpu", epochs=5, max_tokens=7)
    assert config["fit"] == {"device": "cuda", "epochs": 10}
    assert updated["fit"]["device"] == "cpu"
    assert updated["fit"]["epochs"] == 5
    assert updated["input"]["max_tokens"] == 7


def test_subset_run_rescales_denominator_and_invalidates_cache(tmp_path):
    npz = tmp_path / "losses.npz"
    _write_event_matrix(npz)

    full = run_analysis(_config(npz, tmp_path / "full"))
    subset = run_analysis(_config(npz, tmp_path / "subset", max_tokens=100))
    assert subset["subset"] == {"n_total": 400, "n_fitted": 100, "seed": 0}
    assert subset["n_tokens"] == 100
    ratio = _reconstruction_scale(tmp_path / "subset") / _reconstruction_scale(tmp_path / "full")
    assert 0.6 < ratio < 1.6
    assert full["rate_reconstruction_corr"] > 0.8

    again = run_analysis(_config(npz, tmp_path / "subset", max_tokens=100))
    assert again["fit_reused_cache"]
    changed = _config(npz, tmp_path / "subset", max_tokens=100)
    changed["fit"]["epochs"] = 200
    assert not run_analysis(changed)["fit_reused_cache"]
    assert (tmp_path / "subset" / "config.yaml").is_file()


def test_download_list_and_dry_run(capsys):
    main(["download", "--list"])
    listing = capsys.readouterr().out
    assert "step_axis_1m" in listing and "Figure 3" in listing

    main(["reproduce", "fig3", "--quick", "--dry-run", "--device", "cpu"])
    plan = capsys.readouterr().out
    assert "random 50,000 of 1,000,000 tokens" in plan
    assert "2,000 epochs, device cpu" in plan
    assert "dry run" in plan
