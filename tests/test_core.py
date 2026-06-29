import numpy as np

from tokenlearning.core.fitting import fit_token_events, loss_snr_keep_mask
from tokenlearning.core.rates import loss_change_rate
from tokenlearning.core.sigmoid import learning_pulse, sigmoid_loss
from tokenlearning.core.spectrum import estimate_spectrum, reconstruct_rate
from tokenlearning.io.loss_matrix import load_loss_npz, save_loss_npz
from tokenlearning.types import LossMatrix


def test_sigmoid_loss_shape():
    x = np.linspace(1.0, 10.0, 8)
    y = sigmoid_loss(
        x,
        tau=np.array([3.0, 7.0]),
        lambda_=np.array([1.0, 2.0]),
        start=np.array([5.0, 4.0]),
        end=np.array([1.0, 2.0]),
    )
    assert y.shape == (2, 8)
    assert y[0, 0] > y[0, -1]


def test_pulse_and_reconstruction_shapes():
    offsets = np.linspace(-5.0, 5.0, 101)
    pulse = learning_pulse(offsets, np.array([1.0, 2.0]), np.array([1.0, 0.5]))
    assert pulse.shape == (2, 101)
    spectrum = estimate_spectrum(np.linspace(1.0, 10.0, 100), bins=10)
    rate = reconstruct_rate(np.linspace(1.0, 10.0, 20), spectrum, offsets, pulse.mean(axis=0))
    assert rate.shape == (20,)
    assert np.all(np.isfinite(rate))


def test_loss_change_rate():
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([4.0, 3.0, 2.0])
    rate_x, rate = loss_change_rate(x, y)
    assert rate_x.shape == (2,)
    np.testing.assert_allclose(rate, np.array([1.0, 0.5]))


def test_plot_compatible_fit_smoke():
    x = np.linspace(1.0, 8.0, 12)
    tau = np.array([3.0, 5.0], dtype=np.float32)
    lam = np.array([2.0, 1.5], dtype=np.float32)
    start = np.array([4.0, 3.0], dtype=np.float32)
    end = np.array([1.0, 1.2], dtype=np.float32)
    y = sigmoid_loss(x, tau, lam, start, end).astype(np.float32)
    fit = fit_token_events(
        x,
        y,
        {
            "mode": "plot",
            "epochs": 5,
            "lr": 0.05,
            "chunk_size": 2,
            "device": "cpu",
            "lambda_init": 0.1,
            "t_div": 1.0,
            "constrain_sigmoid": {
                "quantile": 0.5,
                "constrain_t_start": {"enabled": True, "mode": "in_range"},
            },
        },
    )
    assert fit.n_tokens == 2
    assert fit.valid_mask.shape == (2,)
    assert np.all(np.isfinite(fit.tau))


def test_loss_snr_keep_mask():
    losses = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 2.01, 1.99, 2.0],
        ],
        dtype=np.float32,
    )
    mask = loss_snr_keep_mask(losses, {"enabled": True, "method": "snr", "min_snr": 0.1})
    assert mask.shape == (2,)
    assert mask[0]


def test_loss_npz_roundtrips_metadata(tmp_path):
    loss_data = LossMatrix(
        axis_values=np.array([1.0, 2.0, 3.0]),
        loss_matrix=np.array([[3.0, 2.0, 1.0], [4.0, 3.0, 2.0]], dtype=np.float32),
        axis_name="step",
        sample_id=np.array([10, 11]),
        token_pos=np.array([0, 0]),
        metadata={"difficulty": np.array([1, 2]), "split_seed": np.array([100, 101])},
    )
    out = tmp_path / "losses.npz"
    save_loss_npz(out, loss_data)
    loaded = load_loss_npz(out)
    assert loaded.axis_name == "step"
    assert loaded.metadata is not None
    np.testing.assert_array_equal(loaded.metadata["difficulty"], np.array([1, 2]))
    np.testing.assert_array_equal(loaded.metadata["split_seed"], np.array([100, 101]))
