import numpy as np
import pytest

from synthshift.eval.performance_baseline import _fold_uncertainty, _paired_summary


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


def test_fold_uncertainty_reports_leave_one_fold_out_and_three_cluster_warning():
    cells = []
    for fold, difference in (("fold_0", 0.1), ("fold_1", 0.2), ("fold_2", 0.6)):
        cells.append(
            {
                "fold": fold,
                "subject": {
                    "source_baseline": {metric: 0.0 for metric in (
                        "kendall_tau_b", "spearman_r", "top2_overlap",
                        "pairwise_ordering_accuracy",
                    )},
                    "synthetic_target": {metric: difference for metric in (
                        "kendall_tau_b", "spearman_r", "top2_overlap",
                        "pairwise_ordering_accuracy",
                    )},
                },
            }
        )

    result = _fold_uncertainty(cells, "subject")

    assert result["leave_one_fold_out"]["fold_2"]["kendall_tau_b"] == pytest.approx(0.15)
    bootstrap = result["fold_cluster_bootstrap"]
    assert bootstrap["cluster_count"] == 3
    assert "descriptive" in bootstrap["warning"]
    interval = bootstrap["metrics"]["kendall_tau_b"]["percentile_95_interval"]
    assert interval[0] <= bootstrap["metrics"]["kendall_tau_b"]["estimate"] <= interval[1]
