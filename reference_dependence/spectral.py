"""Window-averaged log-bandpower classification with training-only scaling."""

import warnings

import numpy as np
from scipy.signal import welch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Half-open intervals, except for the inclusive 40-Hz upper limit.
FREQUENCY_BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 40))


class BandpowerClassifier:
    """Classify participant bags of EEG windows using five frequency bands.

    EEG has shape (participants, windows, channels, samples), in microvolts.
    Welch spectra use two-second Hann segments with 50% overlap and constant
    detrending. Log10 bandpowers are averaged over windows, separately for
    each channel. The 1e-12 floor is in microvolts squared.

    ``common_only=True`` retains one copy of the identical common waveform.
    This avoids changing logistic regularization by duplicating features.
    The assessment selects this option only for its default common-only arm.
    """

    def __init__(self, sampling_frequency, *, random_state=17, common_only=False):
        if not np.isfinite(sampling_frequency) or sampling_frequency <= 80:
            raise ValueError("The five-band model requires sampling_frequency > 80 Hz.")
        self.sampling_frequency = float(sampling_frequency)
        self.random_state = random_state
        self.common_only = common_only

    def features(self, eeg):
        """Return one feature row per participant; learn no cohort statistics."""
        segment_samples = int(round(2 * self.sampling_frequency))
        if eeg.shape[-1] < segment_samples:
            raise ValueError("The five-band model requires windows of at least two seconds.")
        if self.common_only:
            eeg = eeg.mean(axis=2, keepdims=True)
        frequencies, spectrum = welch(
            eeg,
            fs=self.sampling_frequency,
            window="hann",
            nperseg=segment_samples,
            noverlap=segment_samples // 2,
            detrend="constant",
            scaling="density",
            axis=-1,
        )
        frequency_step = frequencies[1] - frequencies[0]
        features = []
        for lower, upper in FREQUENCY_BANDS:
            below_upper = frequencies <= upper if upper == 40 else frequencies < upper
            selected = (frequencies >= lower) & below_upper
            power = spectrum[..., selected].sum(axis=-1) * frequency_step
            features.append(np.log10(np.maximum(power, 1e-12)).mean(axis=1))
        return np.concatenate(features, axis=1)

    def fit(self, eeg, labels):
        """Fit the scaler and logistic regression using training participants."""
        features = self.features(eeg)
        for maximum_iterations in (3000, 10000):
            self.pipeline_ = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=maximum_iterations,
                    random_state=self.random_state,
                ),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                self.pipeline_.fit(features, labels)
            converged = True
            for warning in caught:
                if issubclass(warning.category, ConvergenceWarning):
                    converged = False
                else:
                    warnings.warn(warning.message, warning.category, stacklevel=2)
            if converged:
                return self
        raise RuntimeError("Logistic regression did not converge after 10000 iterations.")

    def predict_probability(self, eeg):
        """Return the probability of label 1 for each participant."""
        return self.pipeline_.predict_proba(self.features(eeg))[:, 1]
