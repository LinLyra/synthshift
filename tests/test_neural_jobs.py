import json

import pytest
import torch

from synthshift.train.jobs import (
    ARCHITECTURES,
    SEEDS,
    NeuralJob,
    build_jobs,
    completed_metadata,
    resolve_device,
)


def _config():
    return {
        "primary_devices": ["phone_a", "phone_b"],
        "folds": [{"name": "fold_0"}, {"name": "fold_1"}],
    }


def test_job_grid_is_complete_unique_and_stable():
    jobs = build_jobs(_config())
    assert len(jobs) == 2 * 2 * len(ARCHITECTURES) * len(SEEDS)
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert jobs[0] == NeuralJob(
        "fold_0__phone_a__compact_cnn__seed0",
        "fold_0",
        "phone_a",
        "compact_cnn",
        0,
    )


def test_unavailable_explicit_accelerator_is_rejected():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")


def test_incomplete_run_is_not_resumed(tmp_path):
    assert completed_metadata(tmp_path) is None
    (tmp_path / "metadata.json").write_text(json.dumps({"status": "complete"}))
    assert completed_metadata(tmp_path) is None
