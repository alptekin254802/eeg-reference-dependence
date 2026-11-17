"""Short CPU checks of the optional EEGNet adapter, using synthetic signals."""

import importlib.util
import unittest

import numpy as np
from numpy.testing import assert_allclose


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires the neural environment")
class EEGNetAdapterTests(unittest.TestCase):
    def test_training_scale_and_network_stay_fixed_at_inference(self):
        import torch
        from reference_dependence.eegnet import EEGNetClassifier

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            rng = np.random.default_rng(12)
            eeg = rng.normal(size=(6, 2, 19, 800))
            labels = np.array([0, 1, 0, 1, 0, 1])
            before_rng = torch.random.get_rng_state().clone()
            classifier = EEGNetClassifier(random_state=11, epochs=1)
            classifier.fit(eeg[:4], labels[:4])
            self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
            self.assertAlmostEqual(classifier.scale_, np.sqrt(np.mean(eeg[:4] ** 2)))
            before_weights = {name: value.clone() for name, value in classifier.model_.state_dict().items()}
            before_scale = classifier.scale_
            original = classifier.predict_probability(eeg[4:])
            residual = eeg[4:] - eeg[4:].mean(axis=2, keepdims=True)
            removed = classifier.predict_probability(residual)
            self.assertEqual(original.shape, (2,))
            self.assertTrue(np.isfinite(removed).all())
            self.assertTrue(((original >= 0) & (original <= 1)).all())
            self.assertEqual(classifier.scale_, before_scale)
            for name, value in classifier.model_.state_dict().items():
                self.assertTrue(torch.equal(value, before_weights[name]))
            assert_allclose(original, classifier.predict_probability(eeg[4:]), atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "19, 800"):
                classifier.predict_probability(eeg[4:, :, :8])
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
