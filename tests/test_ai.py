"""Check full inference, persistence, and AI algorithm boundaries."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from ai import MLP, synthetic_dataset, export_native  # noqa: E402


class AITests(unittest.TestCase):
    def test_probabilities_are_normalized(self):
        model = MLP((7, 5, 3))
        inputs = np.random.default_rng(10).normal(size=(3, 7))
        actual = model.predict_proba(inputs)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-6)

    def test_native_export_contract(self):
        # The header and bytes must match the C++ reader exactly.
        model = MLP((4, 3))
        inputs = np.arange(8, dtype=np.float32).reshape(2, 4)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model.save(folder / "ai_model.npz")
            np.savez(folder / "ai_test.npz", inputs=inputs, labels=[0, 1])
            native = folder / "native"
            export_native(folder, native)
            payload = (native / "inputs.bin").read_bytes()
            np.testing.assert_array_equal(np.frombuffer(payload[:8], dtype="<u4"), [2, 4])
            np.testing.assert_array_equal(np.frombuffer(payload[8:], dtype="<f4").reshape(2, 4), inputs)

    def test_model_round_trip(self):
        model = MLP((4, 6, 3))
        inputs = np.ones((2, 4))

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            model.save(path)
            restored = MLP.load(path)
            np.testing.assert_array_equal(model.predict_proba(inputs), restored.predict_proba(inputs))

    def test_different_dataset_splits(self):
        train, validation, test = synthetic_dataset(8, 3)
        self.assertEqual(len(train[0]), 3000)
        self.assertEqual(len(validation[0]), 600)
        self.assertEqual(len(test[0]), 600)
        self.assertFalse(np.array_equal(validation[0], test[0]))


if __name__ == "__main__":
    unittest.main()
