import numpy as np
import pytest

from synthshift.eval.performance_baseline import _paired_summary


def test_paired_summary_counts_wins_ties_losses_and_delta():
    source = [
        {"kendall_tau_b": 0.2, "spearman_r": 0.3, "top2_overlap": 0.5,
         "pairwise_ordering_accuracy": 0.6},
        {"kendall_tau_b": 0.4, "spearman_r": 0.5, "top2_overlap": 1.0,
         "pairwise_ordering_accuracy": 0.7},
        {"kendall_tau_b": 0.6, "spearman_r": 0.7, "top2_overlap": 0.0,
         "pairwise_ordering_accuracy": 0.8},
    ]
    synthetic = [
        {"kendall_tau_b": 0.3, "spearman_r": 0.3, "top2_overlap": 1.0,
         "pairwise_ordering_accuracy": 0.7},
        {"kendall_tau_b": 0.4, "spearman_r": 0.4, "top2_overlap": 0.5,
         "pairwise_ordering_accuracy": 0.7},
        {"kendall_tau_b": 0.5, "spearman_r": 0.9, "top2_overlap": 0.0,
         "pairwise_ordering_accuracy": 0.7},
    ]

    result = _paired_summary(source, synthetic)

    assert result["kendall_tau_b"]["wins"] == 1
    assert result["kendall_tau_b"]["ties"] == 1
    assert result["kendall_tau_b"]["losses"] == 1
    assert result["kendall_tau_b"]["mean_delta"] == pytest.approx(0.0)
    assert result["spearman_r"]["mean_delta"] == pytest.approx(1 / 30)
    assert np.isfinite(result["top2_overlap"]["median_delta"])
