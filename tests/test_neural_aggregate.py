import numpy as np
import pytest

from synthshift.eval.neural_aggregate import aggregate_neural_seeds, combined_ranking_summary


def _row(configuration, fold, source, target, value, seed=None):
    row = {
        "configuration": configuration,
        "fold": fold,
        "source_device": source,
        "target_device": target,
    }
    if seed is not None:
        row.update({"architecture": configuration, "seed": seed})
    for score in ("pooled", "subject"):
        row[f"source_{score}_macro_f1"] = 0.8
        row[f"real_target_{score}_macro_f1"] = 0.8 - value
        row[f"synthetic_target_{score}_macro_f1"] = 0.8 - value / 2
        row[f"real_{score}_degradation"] = value
        row[f"synthetic_{score}_degradation"] = value / 2
    return row


def test_seed_aggregation_requires_exact_fixed_seeds():
    rows = [_row("compact_cnn", "f", "a", "b", seed + 1, seed) for seed in (0, 1, 2)]
    result = aggregate_neural_seeds(rows)
    assert len(result) == 1
    assert result[0]["configuration"] == "neural_compact_cnn"
    assert result[0]["real_pooled_degradation"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="Expected seeds"):
        aggregate_neural_seeds(rows[:2])


def test_combined_ranking_requires_ten_models_and_sixty_cells():
    configurations = [f"model_{index}" for index in range(10)]
    rows = []
    for fold in range(3):
        for source in range(5):
            for target in range(5):
                if source == target:
                    continue
                for model, configuration in enumerate(configurations):
                    rows.append(
                        _row(configuration, f"f{fold}", f"d{source}", f"d{target}", model / 100)
                    )
    result = combined_ranking_summary(rows[: 7 * 60], rows[7 * 60 :])
    assert result["configuration_count"] == 10
    assert result["cell_count"] == 60
    assert np.isfinite(result["mean"]["pooled_performance"]["kendall_tau_b"])
