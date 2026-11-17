"""Scientific semantics and validation for the reference-dependence interface."""

import json
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.metrics import roc_auc_score

from examples.reference_dependence import simulate_eeg
from reference_dependence import BandpowerClassifier, NoCommonComponentError, assess_reference_dependence
from reference_dependence.assessment import ARMS, _paired_intervals


class ReferenceDependenceTests(unittest.TestCase):
    def setUp(self):
        self.eeg, self.labels, self.ids, self.folds, self.frequency = simulate_eeg("common")

    def assess(self, **overrides):
        arguments = dict(
            eeg=self.eeg, labels=self.labels, participant_ids=self.ids,
            sampling_frequency=self.frequency, fold_ids=self.folds, n_bootstrap=20,
        )
        arguments.update(overrides)
        return assess_reference_dependence(**arguments)

    def test_common_information_and_caller_arrays(self):
        original = self.eeg.copy()
        result = self.assess()
        auc = result.summary["auc"]
        self.assertGreater(auc["common_only"]["estimate"], 0.95)
        self.assertGreater(auc["full_input"]["estimate"], 0.95)
        self.assertAlmostEqual(auc["car_retrained"]["estimate"], 0.5, places=5)
        self.assertLess(result.summary["contrasts"]["inference_removal_minus_full"]["estimate"], -0.3)
        assert_array_equal(self.eeg, original)
        self.assertEqual(result.summary["n_fits"], 9)
        self.assertNotIn(self.ids[0], json.dumps(result.summary))

    def test_spatial_information_is_retained(self):
        eeg, *_ = simulate_eeg("spatial")
        result = self.assess(eeg=eeg)
        auc = result.summary["auc"]
        self.assertAlmostEqual(auc["common_only"]["estimate"], 0.5, places=5)
        self.assertGreater(auc["full_input"]["estimate"], 0.95)
        self.assertGreater(auc["car_retrained"]["estimate"], 0.95)
        self.assertGreater(auc["inference_removal"]["estimate"], 0.95)

    def test_car_and_zero_inputs_are_not_applicable(self):
        car = (self.eeg - self.eeg.mean(axis=2, keepdims=True)).astype(np.float32)
        for eeg in (car, car * 1e-9, car * 1e9, np.zeros_like(car)):
            with self.subTest(scale=float(np.max(np.abs(eeg)))):
                with self.assertRaises(NoCommonComponentError):
                    self.assess(eeg=eeg)

    def test_participant_and_fold_validation(self):
        repeated = self.ids.copy()
        repeated[1] = repeated[0]
        nonfinite = self.eeg.copy()
        nonfinite[0, 0, 0, 0] = np.nan
        invalid = [
            {"participant_ids": repeated}, {"eeg": nonfinite},
            {"labels": np.zeros_like(self.labels)}, {"labels": self.labels + 1},
            {"fold_ids": self.labels}, {"fold_ids": np.zeros_like(self.labels)},
            {"fold_ids": self.folds[:-1]}, {"n_bootstrap": 0},
            {"confidence": 1}, {"sampling_frequency": 0},
            {"sampling_frequency": 80}, {"eeg": self.eeg[..., :100]},
        ]
        for arguments in invalid:
            with self.subTest(arguments=list(arguments)):
                with self.assertRaises(ValueError):
                    self.assess(**arguments)

    def test_three_dimensional_input_and_generated_folds(self):
        result = self.assess(eeg=self.eeg[:, 0], fold_ids=None, n_splits=3)
        self.assertEqual(result.summary["input_shape"], [48, 1, 4, 400])
        self.assertEqual(len(np.unique(result.fold_ids)), 3)
        assert_array_equal(result.fold_ids, self.assess(eeg=self.eeg[:, 0], fold_ids=None, n_splits=3).fold_ids)

    def test_inference_uses_same_fit_and_never_sees_test_labels(self):
        models = []

        class Recorder:
            def fit(model, eeg, labels):
                model.training = eeg.copy()
                model.labels = labels.copy()
                model.calls = []
                model.scale = float(eeg.std())
                return model

            def predict_probability(model, eeg):
                model.calls.append(eeg.copy())
                return np.full(len(eeg), 0.5)

        def factory(seed):
            model = Recorder()
            models.append(model)
            return model

        self.assess(model_factory=factory)
        self.assertEqual(len(models), 9)
        for fold in range(3):
            train, test = self.folds != fold, self.folds == fold
            full, common, car = models[3 * fold : 3 * fold + 3]
            assert_array_equal(full.training, self.eeg[train])
            assert_array_equal(full.labels, self.labels[train])
            self.assertEqual(len(full.calls), 2)
            assert_array_equal(full.calls[0], self.eeg[test])
            assert_allclose(full.calls[1], self.eeg[test] - self.eeg[test].mean(axis=2, keepdims=True))
            assert_allclose(common.training, np.repeat(self.eeg[train].mean(axis=2, keepdims=True), 4, axis=2))
            assert_allclose(car.training, self.eeg[train] - self.eeg[train].mean(axis=2, keepdims=True))
            self.assertEqual(len(common.calls), 1)
            self.assertEqual(len(car.calls), 1)

    def test_invalid_custom_model_and_factory(self):
        class Invalid:
            def fit(self, eeg, labels):
                return self

            def predict_probability(self, eeg):
                return np.ones(len(eeg)) * 2

        with self.assertRaisesRegex(ValueError, "probabilities"):
            self.assess(model_factory=lambda seed: Invalid())

        class Constant(Invalid):
            def predict_probability(self, eeg):
                return np.full(len(eeg), 0.5)

        shared = Constant()
        with self.assertRaisesRegex(ValueError, "fresh model"):
            self.assess(model_factory=lambda seed: shared)

    def test_spectral_scaler_does_not_change_during_prediction(self):
        model = BandpowerClassifier(self.frequency)
        model.fit(self.eeg[:32], self.labels[:32])
        scaler = model.pipeline_.named_steps["standardscaler"]
        before = scaler.mean_.copy()
        assert_allclose(before, model.features(self.eeg[:32]).mean(axis=0))
        model.predict_probability(self.eeg[32:] * 1000)
        assert_array_equal(scaler.mean_, before)

    def test_features_match_the_study_spectral_definition(self):
        from analysis.core.spectral import spectral

        eeg = np.random.default_rng(3).normal(size=(6, 2, 19, 800))
        assert_allclose(BandpowerClassifier(200).features(eeg), spectral(eeg), rtol=1e-12, atol=1e-12)

    def test_slotted_custom_model(self):
        class Slotted:
            __slots__ = ()

            def fit(self, eeg, labels):
                return self

            def predict_probability(self, eeg):
                return np.full(len(eeg), 0.5)

        result = self.assess(model_factory=lambda seed: Slotted())
        self.assertEqual(result.summary["auc"]["full_input"]["estimate"], 0.5)

    def test_paired_bootstrap_matches_independent_manual_calculation(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        baseline = np.array([0.1, 0.2, 0.8, 0.3, 0.7, 0.9])
        removed = baseline[::-1].copy()
        predictions = {arm: baseline.copy() for arm in ARMS}
        predictions["inference_removal"] = removed
        _, contrasts = _paired_intervals(labels, predictions, 30, 0.9, 7)
        rng = np.random.default_rng(7)
        changes = []
        for _ in range(30):
            indices = np.r_[rng.choice([0, 1, 2], 3), rng.choice([3, 4, 5], 3)]
            changes.append(roc_auc_score(labels[indices], removed[indices]) - roc_auc_score(labels[indices], baseline[indices]))
        contrast = contrasts["inference_removal_minus_full"]
        assert_allclose([contrast["lower"], contrast["upper"]], np.quantile(changes, [0.05, 0.95]))
        self.assertEqual(contrasts["car_retrained_minus_full"]["lower"], 0)
        self.assertEqual(contrasts["car_retrained_minus_full"]["upper"], 0)


if __name__ == "__main__":
    unittest.main()
