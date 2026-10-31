import hashlib
import json

import numpy as np
import pytest
import torch

from synthshift.eval.neural import _predict_indices, audit_formal_checkpoints, score_predictions
from synthshift.train.jobs import PROTOCOL_COMMIT, NeuralJob
from synthshift.train.neural import ChannelNormalization


class _FirstAccelerationClassifier(torch.nn.Module):
    def forward(self, values):
        score = values[:, 0].mean(dim=1)
        return torch.stack([-score, score], dim=1)


def test_pooled_and_subject_macro_are_distinct_and_exact():
    truth = np.asarray(["a", "a", "b", "a", "b", "b"])
    prediction = np.asarray(["a", "a", "a", "b", "b", "b"])
    users = np.asarray(["u1", "u1", "u1", "u2", "u2", "u2"])
    scores = score_predictions(truth, prediction, users)
    assert scores["pooled_macro_f1"] == pytest.approx(2 / 3)
    assert scores["subject_macro_f1"] == pytest.approx(0.4)


def test_empty_scores_are_rejected():
    empty = np.asarray([])
    with pytest.raises(ValueError, match="empty"):
        score_predictions(empty, empty, empty)


def test_acceleration_gain_is_applied_in_physical_units_before_normalization():
    windows = np.zeros((1, 4, 6), dtype=np.float32)
    windows[:, :, 0] = 6.0
    normalization = ChannelNormalization(
        mean=np.asarray([10, 0, 0, 0, 0, 0], dtype=np.float32),
        std=np.asarray([2, 1, 1, 1, 1, 1], dtype=np.float32),
    )
    prediction = _predict_indices(
        _FirstAccelerationClassifier(),
        windows,
        normalization,
        torch.device("cpu"),
        acceleration_ratio=2.0,
    )
    np.testing.assert_array_equal(prediction, [1])


def test_integrity_gate_requires_every_expected_checkpoint(tmp_path):
    first = NeuralJob("first", "fold", "phone", "compact_cnn", 0)
    second = NeuralJob("second", "fold", "phone", "compact_cnn", 1)
    directory = tmp_path / "formal" / first.job_id
    directory.mkdir(parents=True)
    checkpoint = b"valid-checkpoint"
    (directory / "checkpoint.pt").write_bytes(checkpoint)
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "smoke": False,
                "protocol_commit": PROTOCOL_COMMIT,
                "job": first.__dict__,
                "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
            }
        )
    )
    audit = audit_formal_checkpoints([first, second], tmp_path)
    assert audit["complete"] == 1
    assert audit["missing"] == ["second"]
    assert not audit["ready_for_target_evaluation"]
