"""Optional adapter for the study's 19-channel EEGNet architecture."""

import numpy as np
import torch
from torch.nn import functional as functional

from analysis.core.model import EEGNet


class EEGNetClassifier:
    """Fit EEGNet on participant bags, then return mean-logit probabilities.

    This adapter uses the study architecture, training-derived scalar RMS,
    class-weighted loss, and fixed 30-epoch optimization by default. Inputs
    must contain 19 channels and 800 samples per window at 200 Hz. Sampling
    frequency is checked by the caller; this class receives waveform arrays.

    Exact study reproduction uses the commands in docs/core.md, including
    the original cohort, preprocessing, folds, runtime and initialization
    ensemble. This adapter runs one initialization per assessment call.
    Import this module only in the neural environment.
    """

    def __init__(self, *, random_state=17, device="cpu", epochs=30, batch_size=4):
        for name, value in (("epochs", epochs), ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.random_state = random_state
        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size

    @staticmethod
    def _check_shape(eeg):
        if eeg.ndim != 4 or eeg.shape[2:] != (19, 800):
            raise ValueError("This EEGNet requires (participants, windows, 19, 800) input.")

    def fit(self, eeg, labels):
        """Learn scalar normalization and network weights from training data."""
        self._check_shape(eeg)
        self.scale_ = float(np.sqrt(np.mean(np.square(eeg, dtype=np.float64))))
        if not np.isfinite(self.scale_) or self.scale_ <= 0:
            raise ValueError("Training input must have positive finite RMS.")
        training = torch.as_tensor(eeg / self.scale_, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        positive_weight = torch.tensor(
            float(np.sum(labels == 0) / np.sum(labels == 1)), device=self.device
        )
        rng = np.random.default_rng(self.random_state)
        # Preserve the caller's CPU and CUDA random-number generator states.
        with torch.random.fork_rng():
            torch.manual_seed(self.random_state)
            self.model_ = EEGNet().to(self.device)
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.001, weight_decay=0.0001)
            for _ in range(self.epochs):
                self.model_.train()
                batches = np.array_split(
                    rng.permutation(len(labels)), int(np.ceil(len(labels) / self.batch_size))
                )
                for indices in batches:
                    batch = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = functional.binary_cross_entropy_with_logits(
                        self.model_(training[batch]), target[batch], pos_weight=positive_weight
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("EEGNet training produced a nonfinite loss.")
                    loss.backward()
                    optimizer.step()
                    self.model_.constrain()
        if not all(torch.isfinite(value).all() for value in self.model_.state_dict().values()):
            raise FloatingPointError("EEGNet training produced nonfinite model parameters.")
        self.model_.eval()
        return self

    def predict_probability(self, eeg):
        """Use the unchanged training scale and weights, including after removal."""
        self._check_shape(eeg)
        probabilities = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(eeg), self.batch_size):
                batch = torch.as_tensor(
                    eeg[start : start + self.batch_size] / self.scale_,
                    dtype=torch.float32,
                    device=self.device,
                )
                probabilities.append(torch.sigmoid(self.model_(batch)).cpu().numpy())
        return np.concatenate(probabilities)
