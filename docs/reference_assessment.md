# Apply the reference-dependence assessment to another dataset

`assess_reference_dependence` compares four uses of the same participant EEG. It implements the component comparisons described in the article through one entry point.

| Output arm | Training input | Held-out input | Model |
| --- | --- | --- | --- |
| `full_input` | Original EEG | Original EEG | Full-input fit |
| `common_only` | Channel mean repeated across channels | Channel mean repeated across channels | Separate fit |
| `inference_removal` | Original EEG | Original EEG minus channel mean | Unchanged full-input fit |
| `car_retrained` | Original EEG minus channel mean | Original EEG minus channel mean | Separate fit |

The common component is the instantaneous mean over the supplied scalp channels. Common-only retraining measures its predictive association. Inference-only removal measures a fitted model's response to losing it, including the effect of input distribution shift. CAR retraining measures performance after learning from the remaining signal. These comparisons do not partition AUC into fractions of information or identify a physiological source.

## Input and preprocessing

Use a NumPy array with shape `(participants, windows, channels, samples)`. An array of shape `(participants, channels, samples)` is accepted as one window per participant. All participants need the same channel order, sampling frequency and window dimensions. Use microvolts for the default spectral model.

Provide one binary label and one unique string or integer identifier per participant. Label `1` is the positive class. Put each participant's windows on the window axis, not in separate participant rows. Combine repeated sessions under the same participant before calling the function. The interface does not infer whether distinct identifiers belong to the same person.

Select scalp EEG channels only; auxiliary and reference-electrode channels change the meaning of the channel mean. Apply the intended continuous-record filtering and window selection before calling the function. Determine eligibility without inspecting these performance comparisons. The function does not filter, reject artifacts, interpolate channels or resample recordings.

The function raises `NoCommonComponentError` when any participant has common RMS / full-input RMS at or below `1e-6`, or has zero signal. This scale-independent guard catches negligible means, including previously CAR-referenced input. It does not establish the physical acquisition reference. Do not automatically exclude flagged participants: review preprocessing and the channel set.

## Use the default spectral model

Run from the repository root in the analysis environment described in the README. No additional package installation is needed.

```python
import json
from pathlib import Path

import numpy as np
from reference_dependence import assess_reference_dependence

eeg = np.load("data/eeg_windows_uv.npy")
labels = np.load("data/labels.npy")
participant_ids = np.load("data/participant_ids.npy")
fold_ids = np.load("data/fold_ids.npy")

result = assess_reference_dependence(
    eeg,
    labels,
    participant_ids,
    sampling_frequency=200,
    fold_ids=fold_ids,
    n_bootstrap=2000,
    random_state=17,
)

output = Path("outputs/reference_assessment.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
print(result.summary["contrasts"]["inference_removal_minus_full"])
```

`fold_ids` assigns each participant to exactly one outer test fold; its complement is the training set. Both classes must occur in each training and test fold. If omitted, the function creates five shuffled stratified participant folds. Use existing folds for an established comparison. Each call performs three fits per fold; the removal arm reuses the full-input fit. It does not train a model on zeros.

The default model uses Welch log10 bandpowers for 1–4, 4–8, 8–13, 13–30 and 30–40 Hz. The final 40-Hz bin is included. Two-second Hann segments overlap by 50%; constant detrending precedes spectral estimation. Bin powers are summed with the frequency-bin width, floored at `1e-12` microvolts squared, then log-transformed and averaged over windows. Full-input and CAR features retain every channel. The common-only arm retains one feature per band instead of repeating identical features. The scaler is fitted on training participants only. Logistic regression uses `C=1` and balanced class weights. The model requires sampling frequency above 80 Hz and windows of at least two seconds.

This fixed-C example is a reusable comparison, not the reproduction command for every reported spectral model. The article's analyses include nested model selection and neural initialization ensembles; their exact commands are in [core analyses](core.md).

## Use EEGNet or another model

The factory receives a seed and must return a fresh classifier on every call. A classifier implements `fit(eeg, labels)` and `predict_probability(eeg)`, returning one probability of label 1 per participant. Fitted preprocessing belongs inside `fit`. Prediction must keep all preprocessing and model parameters fixed. Hyperparameter selection, if used, must remain within training participants.

For the study's EEGNet architecture, use the neural environment. This example requires 200-Hz EEG with 19 channels and 800 samples per window. It starts 15 fits of 30 epochs by default; it is not a short software check.

```python
from reference_dependence.eegnet import EEGNetClassifier

def make_eegnet(seed):
    return EEGNetClassifier(random_state=seed, device="cuda", epochs=30)

result = assess_reference_dependence(
    eeg, labels, participant_ids,
    sampling_frequency=200,
    fold_ids=fold_ids,
    model_factory=make_eegnet,
)
```

The adapter imports the existing EEGNet architecture. It learns scalar RMS on training inputs and pools window logits within participants. The removal arm retains that scalar and those weights. Each assessment call uses one initialization per fold; it does not reconstruct the article's three-initialization ensemble. Hardware-dependent neural numerical differences remain possible. Preserve the factory settings alongside your output, since the summary identifies a custom factory but cannot infer its configuration.

## Outputs and intervals

`result.summary` contains AUC estimates and percentile intervals for each arm, and paired AUC changes relative to the full-input baseline. A negative `inference_removal_minus_full` means lower AUC after removing the mean from the fixed model's inputs. Intervals use identical label-stratified participant draws for all arms. They condition on the fitted models and split, omit training/model-selection uncertainty, and are pointwise without multiplicity correction. Pooled out-of-fold AUC assumes probabilities are comparable across fold models. A small interval or a large drop is not evidence of a particular artifact source.

`result.probabilities`, `participant_ids`, `labels` and `fold_ids` contain participant-level information and inherit the input data's access restrictions. The function writes no files. The example exports only `summary`; review even aggregate outputs against the dataset's sharing terms.

## Software checks and executable example

```sh
python -B -m unittest discover -s tests -v
python -B -m examples.reference_dependence --output outputs/reference_assessment.json
```

The example constructs two simulated datasets: one has class information in the common component, the other in a zero-mean spatial pattern. It prints four AUCs per scenario and writes their aggregate summaries. The paired background construction makes the intended information location known. These fixtures verify transformations and model behavior; they do not estimate clinical sensitivity, specificity or a general false-positive rate.

Tests cover participant and fold validation, already-CAR inputs, fixed-model removal, training-only scaling, paired bootstrap arithmetic and the two simulated information locations. In the neural environment the test suite also runs a one-epoch CPU check of EEGNet's training scale and fixed inference parameters. Without PyTorch that test is explicitly skipped. The checks and example use small simulated arrays; they do not require dataset access.
