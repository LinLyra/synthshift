import unittest

import torch

from synthshift.models.neural import build_neural_model, parameter_count


class NeuralModelTest(unittest.TestCase):
    def test_all_frozen_architectures_forward_and_backward(self) -> None:
        values = torch.randn(4, 6, 250)
        for name in ("compact_cnn", "deepconvlstm", "resnet1d"):
            with self.subTest(name=name):
                model = build_neural_model(name, classes=6)
                logits = model(values)
                self.assertEqual(tuple(logits.shape), (4, 6))
                logits.sum().backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
                self.assertGreater(parameter_count(model), 10_000)

    def test_unknown_architecture_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_neural_model("unknown", classes=6)


if __name__ == "__main__":
    unittest.main()
