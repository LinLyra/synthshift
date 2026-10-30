import unittest

import numpy as np

from synthshift.eval.cpu_multimodel import _pairwise_accuracy, _ranking_metrics
from synthshift.models.cpu_multimodel import feature_indices, model_runs


class CpuMultimodelTest(unittest.TestCase):
    def test_feature_subsets_are_disjoint_and_cover_sensor_features(self) -> None:
        acceleration = feature_indices("logistic_acc_only")
        gyroscope = feature_indices("logistic_gyro_only")
        self.assertEqual(len(acceleration), 39)
        self.assertEqual(len(gyroscope), 39)
        self.assertEqual(len(set(acceleration) & set(gyroscope)), 0)
        self.assertEqual(set(acceleration) | set(gyroscope), set(range(78)))

    def test_protocol_has_seven_configurations_and_nested_tree_seeds(self) -> None:
        runs = model_runs()
        self.assertEqual(len({run.configuration for run in runs}), 7)
        self.assertEqual(sum(run.configuration == "random_forest" for run in runs), 3)
        self.assertEqual(sum(run.configuration == "extra_trees" for run in runs), 3)

    def test_ranking_metrics_distinguish_match_and_reverse(self) -> None:
        real = np.array([0.1, 0.2, 0.3, 0.4])
        matched = _ranking_metrics(real, real, lower_better=True)
        reversed_metrics = _ranking_metrics(real, real[::-1], lower_better=True)
        self.assertAlmostEqual(matched["kendall_tau_b"], 1.0)
        self.assertAlmostEqual(matched["pairwise_ordering_accuracy"], 1.0)
        self.assertAlmostEqual(reversed_metrics["kendall_tau_b"], -1.0)
        self.assertAlmostEqual(_pairwise_accuracy(real, real[::-1], lower_better=True), 0.0)


if __name__ == "__main__":
    unittest.main()
