import importlib.util
from pathlib import Path

import numpy as np
import pytest

pq = pytest.importorskip("pyarrow.parquet")

TOOL = Path(__file__).resolve().parents[1] / "tools" / "export_hf_previews.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("export_hf_previews", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_previews_samples_rows_and_keeps_metadata(tmp_path):
    tool = _load_tool()
    loss_matrix = np.random.default_rng(0).random((50, 6), dtype=np.float32)
    np.savez_compressed(
        tmp_path / "big.npz",
        axis_values=np.arange(1.0, 7.0),
        loss_matrix=loss_matrix,
        axis_name=np.asarray(["training step"]),
    )
    np.savez_compressed(
        tmp_path / "small.npz",
        axis_values=np.arange(1.0, 4.0),
        loss_matrix=loss_matrix[:5, :3],
        axis_name=np.asarray(["training step"]),
        question=np.asarray([f"q{i}" for i in range(5)]),
        difficulty=np.arange(5),
    )
    manifest = {
        "files": [
            {"id": "big", "local_path": "big.npz"},
            {"id": "small", "local_path": "small.npz"},
        ]
    }

    written = tool.export_previews(manifest, data_root=tmp_path, out_dir=tmp_path / "previews", max_rows=10, seed=0)
    assert sorted(path.name for path in written) == ["axes.parquet", "big.parquet", "small.parquet"]

    big = pq.read_table(tmp_path / "previews" / "big.parquet").to_pydict()
    assert len(big["row"]) == 10
    assert big["row"] == sorted(big["row"])
    for row, losses in zip(big["row"], big["loss"]):
        np.testing.assert_array_equal(np.asarray(losses, dtype=np.float32), loss_matrix[row])

    small = pq.read_table(tmp_path / "previews" / "small.parquet").to_pydict()
    assert list(small) == ["row", "question", "difficulty", "loss"]
    assert small["row"] == list(range(5))
    assert small["question"] == [f"q{i}" for i in range(5)]

    axes = pq.read_table(tmp_path / "previews" / "axes.parquet").to_pydict()
    assert axes["asset"] == ["big", "small"]
    assert axes["n_tokens"] == [50, 5]
    assert axes["preview_rows"] == [10, 5]
    assert axes["axis_values"][1] == [1.0, 2.0, 3.0]
