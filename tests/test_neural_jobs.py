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
    run_source_job,
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


def test_checksum_mismatch_is_rejected(tmp_path):
    (tmp_path / "checkpoint.pt").write_bytes(b"checkpoint")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"status": "complete", "checkpoint_sha256": "wrong"})
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        completed_metadata(tmp_path)


def test_formal_job_cannot_override_frozen_optimization(tmp_path):
    job = NeuralJob("id", "fold_0", "phone", "compact_cnn", 0)
    with pytest.raises(ValueError, match="Formal jobs"):
        run_source_job(
            job,
            features_path=tmp_path / "missing-features",
            raw_cache_path=tmp_path / "missing-cache",
            split_path=tmp_path / "missing-split",
            output_root=tmp_path,
            max_epochs=1,
        )
