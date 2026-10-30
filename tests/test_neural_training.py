import numpy as np
import pytest
import torch

from synthshift.train.neural import (
    channel_normalization,
    checkpoint_sha256,
    encode_labels,
    inverse_frequency_weights,
    refit_exact_epochs,
    select_best_epoch,
)


def test_channel_normalization_is_channelwise_and_clamped():
    windows = np.zeros((2, 3, 6), dtype=np.float32)
    windows[:, :, 1] = np.arange(6).reshape(2, 3)
    result = channel_normalization(windows)
    assert result.mean.shape == (6,)
    assert result.mean[1] == pytest.approx(2.5)
    assert result.std[0] == pytest.approx(1e-6)
    assert result.std[1] == pytest.approx(np.std(np.arange(6)))


def test_labels_and_inverse_frequency_weights():
    classes = np.asarray(["bike", "walk"])
    encoded = encode_labels(np.asarray(["walk", "bike", "bike"]), classes)
    np.testing.assert_array_equal(encoded, [1, 0, 0])
    weights = inverse_frequency_weights(encoded, 2)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[1] == pytest.approx(2 * weights[0])


def test_missing_training_class_is_rejected():
    with pytest.raises(ValueError, match="Every class"):
        inverse_frequency_weights(np.asarray([0, 0]), 2)


def test_selection_and_fresh_refit_complete_on_tiny_data():
    rng = np.random.default_rng(4)
    windows = rng.normal(size=(12, 24, 6)).astype(np.float32)
    labels = np.asarray(["a", "b"] * 6)
    classes = np.asarray(["a", "b"])
    result = select_best_epoch(
        "compact_cnn",
        windows[:8],
        labels[:8],
        windows[8:],
        labels[8:],
        classes,
        seed=0,
        device=torch.device("cpu"),
        batch_size=4,
        max_epochs=2,
        patience=2,
    )
    assert 1 <= result.best_epoch <= 2
    model, normalization, losses = refit_exact_epochs(
        "compact_cnn",
        windows,
        labels,
        classes,
        seed=0,
        epochs=result.best_epoch,
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert len(losses) == result.best_epoch
    assert normalization.mean.shape == (6,)
    assert len(checkpoint_sha256(model)) == 64
