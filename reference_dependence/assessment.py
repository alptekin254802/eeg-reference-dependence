"""Compare common-only learning, common removal and CAR retraining."""

from dataclasses import dataclass
from typing import Callable, Protocol
import weakref

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .spectral import BandpowerClassifier


ARMS = ("full_input", "common_only", "inference_removal", "car_retrained")
CONTRASTS = {
    "inference_removal_minus_full": ("inference_removal", "full_input"),
    "car_retrained_minus_full": ("car_retrained", "full_input"),
    "common_only_minus_full": ("common_only", "full_input"),
}


class ParticipantClassifier(Protocol):
    """Model interface; all fitted preprocessing belongs inside ``fit``.

    Inputs have shape (participants, windows, channels, samples). Predictions
    must be one probability of label 1 per participant. Any tuning must use
    training participants only. ``predict_probability`` must not update the
    fitted model, scaler, normalization statistics or batch statistics.
    """

    def fit(self, eeg: np.ndarray, labels: np.ndarray): ...

    def predict_probability(self, eeg: np.ndarray) -> np.ndarray: ...


class NoCommonComponentError(ValueError):
    """The selected channels have negligible spatial mean; assessment is undefined."""


@dataclass
class AssessmentResult:
    """Aggregate estimates and participant-level out-of-fold probabilities.

    ``summary`` contains no participant identifiers or individual predictions.
    The other fields are participant-level data and inherit the input data's
    access restrictions. Saving them is an explicit caller decision.
    """

    summary: dict
    probabilities: dict[str, np.ndarray]
    participant_ids: np.ndarray
    labels: np.ndarray
    fold_ids: np.ndarray


def _integer_at_least(value, minimum, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _validate_inputs(eeg, labels, participant_ids):
    values = np.asarray(eeg)
    if values.dtype.kind not in "fiu":
        raise ValueError("EEG must contain real numeric samples.")
    if values.ndim == 3:
        values = values[:, None, :, :]
    if values.ndim != 4 or any(size == 0 for size in values.shape):
        raise ValueError("EEG shape must be (participants, [windows,] channels, samples).")
    if values.shape[2] < 2 or values.shape[3] < 2:
        raise ValueError("At least two channels and two samples are required.")
    if not np.isfinite(values).all():
        raise ValueError("EEG contains NaN or infinity; determine eligibility before assessment.")
    target = np.asarray(labels)
    if target.shape != (len(values),) or not np.isin(target, [0, 1]).all():
        raise ValueError("labels must contain one binary value (0 or 1) per participant.")
    target = target.astype(int)
    if np.bincount(target, minlength=2).min() < 2:
        raise ValueError("At least two participants per class are required.")
    identifiers = np.asarray(participant_ids)
    if identifiers.shape != target.shape or identifiers.dtype.kind not in "iuUS":
        raise ValueError("participant_ids must be a one-dimensional string or integer array.")
    if identifiers.dtype.kind in "US" and any(not str(item).strip() for item in identifiers):
        raise ValueError("participant_ids cannot contain empty identifiers.")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("Use one row per participant; put their windows on the window axis.")
    # Float64 stabilizes subtraction even for float32 CAR input. Work on copies
    # below so neither models nor callers can silently alter the original EEG.
    return np.asarray(values, dtype=np.float64), target, identifiers.copy()


def _make_folds(labels, fold_ids, n_splits, random_state):
    if fold_ids is None:
        _integer_at_least(n_splits, 2, "n_splits")
        if np.bincount(labels).min() < n_splits:
            raise ValueError("Each class must contain at least n_splits participants.")
        assignments = np.empty(len(labels), dtype=int)
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
        for fold, (_, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            assignments[test] = fold
    else:
        assignments = np.asarray(fold_ids)
        if assignments.shape != labels.shape or assignments.dtype.kind not in "iu":
            raise ValueError("fold_ids must contain one integer assignment per participant.")
        assignments = assignments.copy()
    if len(np.unique(assignments)) < 2:
        raise ValueError("At least two outer folds are required.")
    splits = []
    for fold in np.unique(assignments):
        test = np.flatnonzero(assignments == fold)
        train = np.flatnonzero(assignments != fold)
        if len(np.unique(labels[train])) != 2 or len(np.unique(labels[test])) != 2:
            raise ValueError("Every training and test fold must contain both classes.")
        splits.append((train, test))
    return assignments, splits


def _predict(model, eeg):
    probability = np.asarray(model.predict_probability(eeg), dtype=float)
    if probability.shape != (len(eeg),):
        raise ValueError("predict_probability must return one value per participant.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("predict_probability must return finite probabilities in [0, 1].")
    return probability


def _paired_intervals(labels, predictions, n_bootstrap, confidence, random_state):
    """Resample participants within label, using identical draws for every arm."""
    estimates = {arm: float(roc_auc_score(labels, predictions[arm])) for arm in ARMS}
    draws = {arm: np.empty(n_bootstrap) for arm in ARMS}
    groups = [np.flatnonzero(labels == label) for label in (0, 1)]
    rng = np.random.default_rng(random_state)
    for draw in range(n_bootstrap):
        selected = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        for arm in ARMS:
            draws[arm][draw] = roc_auc_score(labels[selected], predictions[arm][selected])
    tail = (1 - confidence) / 2

    def interval(estimate, samples):
        lower, upper = np.quantile(samples, [tail, 1 - tail])
        return {"estimate": estimate, "lower": float(lower), "upper": float(upper)}

    auc = {arm: interval(estimates[arm], draws[arm]) for arm in ARMS}
    contrasts = {
        name: interval(estimates[left] - estimates[right], draws[left] - draws[right])
        for name, (left, right) in CONTRASTS.items()
    }
    return auc, contrasts


def assess_reference_dependence(
    eeg,
    labels,
    participant_ids,
    *,
    sampling_frequency,
    fold_ids=None,
    n_splits=5,
    model_factory: Callable[[int], ParticipantClassifier] | None = None,
    random_state=17,
    n_bootstrap=2000,
    confidence=0.95,
    common_rms_tolerance=1e-6,
) -> AssessmentResult:
    """Assess reference dependence using matched participant folds.

    Parameters
    ----------
    eeg : array, shape (participants, windows, channels, samples) or (participants, channels, samples)
        Preprocessed EEG in microvolts. Use scalp EEG channels in a consistent
        order and one row per participant. Exclude EOG/ECG/reference channels.
        Filtering, eligibility and window selection are caller responsibilities.
        Windows may be repeated within a participant, never as independent rows.
    labels : array, shape (participants,)
        Binary diagnosis/task labels; 1 is the positive class.
    participant_ids : array, shape (participants,)
        Unique string or integer identifiers, used to reject repeated people.
    sampling_frequency : float
        Samples per second. The default spectral model requires >80 Hz and
        windows of at least two seconds.
    fold_ids : integer array, optional
        One outer-test fold per participant. Training uses its complement.
        Existing study folds should be supplied to preserve comparisons.
    n_splits : int
        Number of stratified folds when fold_ids is omitted.
    model_factory : callable(seed), optional
        Return a fresh ParticipantClassifier for each fit, with identical
        hyperparameters across arms. It receives training data only through
        fit(eeg, labels). Common-only inputs retain the full channel count by
        repeating the instantaneous spatial mean. The default is five-band
        logistic regression, with one copy of common-only features.
    random_state : int
        Controls generated folds, matched model initialization and bootstrap.
    n_bootstrap : int
        Number of label-stratified participant resamples, without model refits.
    confidence : float
        Coverage of pointwise percentile intervals; no multiplicity correction.
    common_rms_tolerance : float
        Reject if any participant's common RMS / full-input RMS is at or below
        this dimensionless threshold. It detects negligible spatial means,
        including already-CAR data, without identifying the physical reference.

    Returns
    -------
    AssessmentResult
        Aggregate AUCs, paired changes and conditional intervals; individual
        out-of-fold probabilities remain available separately for local use.
        This function runs three fits per fold. Inference removal reuses the
        full-input fit, including all fitted preprocessing. It does not fit
        a model to zero-valued common-only inputs.

    Notes
    -----
    Pooled out-of-fold AUC assumes scores are comparable across fitted folds.
    Intervals condition on the split and fitted models; they exclude training
    and model-selection uncertainty. A drop after removal can reflect input
    distribution shift and does not identify an artifact or physical source.
    The common component is defined over the supplied channel set.
    """
    _integer_at_least(random_state, 0, "random_state")
    _integer_at_least(n_bootstrap, 2, "n_bootstrap")
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive and finite.")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1.")
    if not np.isfinite(common_rms_tolerance) or not 0 < common_rms_tolerance < 1:
        raise ValueError("common_rms_tolerance must be strictly between 0 and 1.")
    values, target, identifiers = _validate_inputs(eeg, labels, participant_ids)
    assignments, splits = _make_folds(target, fold_ids, n_splits, random_state)

    # Normalize only for this numerical guard; do not normalize model inputs.
    peak = np.max(np.abs(values), axis=(1, 2, 3), keepdims=True)
    if (peak == 0).any():
        raise NoCommonComponentError("Assessment is not applicable to zero-signal participants.")
    scaled = values / peak
    common_ratio = np.sqrt(
        np.mean(scaled.mean(axis=2) ** 2, axis=(1, 2))
        / np.mean(scaled ** 2, axis=(1, 2, 3))
    )
    negligible = int((common_ratio <= common_rms_tolerance).sum())
    if negligible:
        raise NoCommonComponentError(
            f"Assessment is not applicable: {negligible} participant(s) have negligible "
            "spatial mean. Check for prior CAR and the selected channel set; "
            "do not automatically exclude these participants."
        )
    del scaled

    common = values.mean(axis=2, keepdims=True)
    residual = values - common
    predictions = {arm: np.full(len(target), np.nan) for arm in ARMS}
    fold_counts = []
    fitted_models = []
    for fold_number, (train, test) in enumerate(splits):
        seed = int(random_state + 100 * fold_number + 50000)

        def fresh_model(common_only=False):
            if model_factory is None:
                return BandpowerClassifier(
                    sampling_frequency, random_state=seed, common_only=common_only
                )
            model = model_factory(seed)
            if any(model is previous() for previous in fitted_models):
                raise ValueError("model_factory must return a fresh model for every fit.")
            try:
                fitted_models.append(weakref.ref(model))
            except TypeError:
                # Slotted custom estimators may not support weak references.
                # Keep those objects alive to detect accidental reuse too.
                fitted_models.append(lambda retained=model: retained)
            return model

        full_model = fresh_model()
        full_model.fit(values[train].copy(), target[train].copy())
        predictions["full_input"][test] = _predict(full_model, values[test].copy())
        predictions["inference_removal"][test] = _predict(full_model, residual[test].copy())

        common_model = fresh_model(common_only=True)
        common_model.fit(np.repeat(common[train], values.shape[2], axis=2), target[train].copy())
        predictions["common_only"][test] = _predict(
            common_model, np.repeat(common[test], values.shape[2], axis=2)
        )

        car_model = fresh_model()
        car_model.fit(residual[train].copy(), target[train].copy())
        predictions["car_retrained"][test] = _predict(car_model, residual[test].copy())
        fold_counts.append({"train": int(len(train)), "test": int(len(test))})

    auc, contrasts = _paired_intervals(target, predictions, n_bootstrap, confidence, random_state)
    summary = {
        "n_participants": len(target),
        "class_counts": np.bincount(target).tolist(),
        "input_shape": list(values.shape),
        "sampling_frequency": float(sampling_frequency),
        "model": "five_band_logistic" if model_factory is None else "custom_factory",
        "n_fits": 3 * len(splits),
        "fold_counts": fold_counts,
        "random_state": int(random_state),
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
        "interval_method": "label-stratified paired participant percentile bootstrap; fitted models fixed",
        "common_rms_tolerance": float(common_rms_tolerance),
        "minimum_common_rms_ratio": float(common_ratio.min()),
        "auc": auc,
        "contrasts": contrasts,
    }
    return AssessmentResult(summary, predictions, identifiers, target.copy(), assignments)
