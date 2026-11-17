"""Run the reference-dependence assessment on simulated participant windows."""

import argparse
import json
from pathlib import Path

import numpy as np

from reference_dependence import assess_reference_dependence


def simulate_eeg(signal_location, *, random_state=41):
    """Construct a balanced software fixture with common or spatial information.

    Both classes share the same paired background realizations. The intended
    class difference is therefore known, without a chance background group
    imbalance. This is a software fixture, not a model of clinical EEG or an
    estimate of the assessment's sensitivity and specificity.
    """
    if signal_location not in {"common", "spatial"}:
        raise ValueError("signal_location must be 'common' or 'spatial'.")
    rng = np.random.default_rng(random_state)
    pairs, windows, channels, samples = 24, 3, 4, 400
    sampling_frequency = 100
    time = np.arange(samples) / sampling_frequency
    labels = np.tile([0, 1], pairs)
    background = rng.normal(size=(pairs, windows, channels, samples))
    spatial_noise = background - background.mean(axis=2, keepdims=True)
    common_noise = rng.normal(size=(pairs, windows, 1, samples))
    eeg = np.repeat(spatial_noise + common_noise, 2, axis=0)
    phase = rng.uniform(0, 2 * np.pi, size=(pairs, windows, 1, 1))
    oscillation = np.repeat(np.sin(2 * np.pi * 10 * time + phase), 2, axis=0)
    amplitude = (0.5 + 3 * labels)[:, None, None, None]
    topography = np.ones((1, 1, channels, 1))
    if signal_location == "spatial":
        topography = np.array([1, -1, 1, -1])[None, None, :, None]
    eeg += amplitude * oscillation * topography
    # Keep each background pair in the same test fold. Participant IDs remain
    # distinct; paired backgrounds are a property of this simulation only.
    folds = np.repeat(np.arange(pairs) % 3, 2)
    identifiers = np.array([f"simulated-{index:03d}" for index in range(len(labels))])
    return eeg, labels, identifiers, folds, sampling_frequency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/reference_assessment.json"))
    arguments = parser.parse_args()
    summaries = {}
    for location in ("common", "spatial"):
        eeg, labels, identifiers, folds, frequency = simulate_eeg(location)
        result = assess_reference_dependence(
            eeg, labels, identifiers, sampling_frequency=frequency,
            fold_ids=folds, n_bootstrap=500,
        )
        summaries[location] = result.summary
        print(location, {arm: round(value["estimate"], 3) for arm, value in result.summary["auc"].items()})
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
