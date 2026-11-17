# The EEG component removed by average referencing carries ADHD classification in resting EEG

Analysis code and aggregate results for the study by **Alptekin Cinbat, Ümit Şentürk and İbrahim Yücedağ**.

The study examines how EEG reference choice relates to ADHD classification, common-component power and model attribution. The main TDBRAIN cohort contains 189 adults. Linked-mastoid and common-average representations are compared using the same participants and validation folds. Component-power, auxiliary-signal, model-sensitivity and external-cohort analyses characterize the scope of that association.

## Contents

```text
analysis/
  core/           Cohort reconstruction, signal preparation and classification
  components/     Power, covariates, auxiliary signals and RMS normalization
  explanations/   Integrated Gradients, DeepSHAP and agreement analyses
  external/       External preparation, transport and power
reference_dependence/  Reusable participant-level comparison interface
examples/         Executable simulated EEG example
tests/            Component comparison and model checks
results/          Aggregate statistics and tabulated results
scripts/          Figure and table generation; aggregate consistency checks
docs/             Implementation details, numerical checks and analysis commands
```

`results/aggregate.json` contains full-precision group estimates. `results/figure_values.json` identifies the 52 statistics used by the main figures. `results/tables.json` contains the tabulated supplement results. Confidence intervals condition on the fitted models and the stated resampling unit; the paper and supplement define their interpretation.

The [analysis details](docs/analysis_details.md) and [additional analyses](docs/additional_analyses.md) accompany Online Resource 1. The [documentation index](docs/README.md) links all analysis guides. The [implementation guide](docs/methods_details.md) lists network settings and sampling seeds; [numerical validation](docs/numerical_validation.md) records tolerances and verification counts.

## Installation

Use **Python 3.11**. The analysis environment handles data preparation, classical models, statistics, figures and tables.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip==26.3
.venv/bin/python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip==26.3
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In the commands below, `python` means the interpreter in the relevant environment. Activate that environment or replace `python` with its full executable path. Run commands from the repository root. Output directories are created by the scripts; input paths are examples to replace with your local downloads.

### Neural environment

Model training and attribution use a separate environment with **PyTorch 2.0.1 and NumPy 1.26.4**. The reported neural fits used CUDA 11.8 and a GTX 1060 with 6 GB memory. A compatible NVIDIA driver is required for CUDA execution. Keep these dependencies separate from the NumPy 2 analysis environment.

```sh
python3.11 -m venv .venv-neural
.venv-neural/bin/python -m pip install --upgrade pip==26.3
.venv-neural/bin/python -m pip install -r requirements-neural.txt
.venv-neural/bin/python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

On Windows, use `py -3.11` and `.venv-neural\Scripts\python.exe` for the same commands. For CPU-only code checks, install the PyTorch wheel from `https://download.pytorch.org/whl/cpu` instead. CPU execution is not a claim of bitwise agreement with CUDA; the RMS training comparison requires the specified CUDA runtime. Installation options follow the [PyTorch version documentation](https://pytorch.org/get-started/previous-versions/#v201).

## Figures and tables from aggregate results

These commands require no participant data or model training:

```sh
python -B scripts/check_results.py
python -B scripts/figures.py --results results --output outputs/figures
python -B scripts/tables.py --results results --output outputs/tables
```

Expected outputs:

- Five main figures, each as PDF and PNG; `outputs/figures/validation.json` records numerical input checks.
- Twenty-four table CSVs (supplement and linked technical-document tables), `tables.md`, and the complete scalar-result export `aggregate_statistics.csv` in `outputs/tables`.
- A consistency check of all 52 figure values.

This route renders the recorded group results. To recalculate estimates from EEG and fitted models, follow the analysis commands below.

## Analyses from source data

Obtain the datasets through [the documented access routes](docs/data.md). The linked analysis guides contain the complete command sequences, input formats and expected output files.

| Analysis | Reproduction commands | Main outputs and expected observations |
| --- | --- | --- |
| TDBRAIN cohort and preprocessing | [Core: reconstruct the cohort](docs/core.md#reconstruct-the-cohort) | Header inventory → census → 193 eligible records → 189 retained participants; four aligned waveform representations and five participant folds |
| EEGNet, temporal CNN and spectral classification | [Core models and predictors](docs/core.md) | Held-out logits/probabilities, checkpoints and classification summaries; three-initialization EEGNet AUC approximately 0.862 linked-mastoid and 0.673 CAR |
| Matching, duration, quality, training, fold and Laplacian sensitivities | [Reported training sensitivities](docs/core.md#reported-training-sensitivities) | Explicit model specifications and comparison results; 112 matched participants and 155 in the duration stratum |
| Common-component and CAR power | [Component commands](docs/components.md#commands) | `outputs/power/results.json`; primary N=186, total common-power ADHD/control ratio approximately 0.571 |
| Age/sex, quality, HR and MASS covariates | [Component commands](docs/components.md#commands) | Locally generated covariate table, held-out probabilities and paired contrasts; unadjusted/age-sex/quality-adjusted AUC approximately 0.802/0.800/0.712 |
| Auxiliary template amplitude and archive subgroups | [Component commands](docs/components.md#commands) | `outputs/ecg/results.json` and `outputs/archive/results.json`; Erbs-template amplitude ratio approximately 1.014 |
| Participant RMS normalization | [RMS sensitivity](docs/components.md#participant-rms-sensitivity) | Ten fits and paired reference-contrast change; approximately 0.002 with interval −0.052 to 0.057 |
| IG, DeepSHAP, perturbations, seed agreement  | [Explanation commands](docs/explanations.md) | Completeness checks, locally stored attributions, faithfulness and pairwise agreement; across-reference kappa approximately 0.111 |
| Common-component removal at inference | [Held-out transformation evaluation](docs/core.md#held-out-transformation-evaluation) | `common_removal.json`; original versus channel-mean-removed AUC approximately 0.838 versus 0.526, using each participant's held-out checkpoint |
| Early/late sensitivity | [Temporal preparation](docs/core.md#separate-earlylate-intervals) and [explanations](docs/explanations.md) | Paired 13-window representations from 4–56 and 64–116 seconds; saved-model comparisons |
| External model transport | [External commands](docs/external.md) | Wroclaw probabilities and aggregate metrics, with explicit provisional-unit assumptions |
| External component-power comparisons | [External power commands](docs/external.md) | Common-power ratios approximately 1.067 Wroclaw and 1.007 Nasrabadi |

Neural commands fit multiple participant-fold models and can take substantial time. Choose the documented analysis modules explicitly. Statistical resampling uses stored held-out predictions rather than refitting models in every bootstrap replicate. Exact scalar equality is appropriate for aggregate rendering; neural numerical agreement depends on the stated runtime and hardware.

## Assess reference dependence in another dataset

[`assess_reference_dependence`](reference_dependence/assessment.py) compares full-input classification, common-only retraining, inference-only common removal and CAR retraining using the same participant folds. The default model uses bandpower and logistic regression; an optional adapter uses the study's EEGNet architecture.

```python
from reference_dependence import assess_reference_dependence

result = assess_reference_dependence(
    eeg, labels, participant_ids,
    sampling_frequency=200,
    fold_ids=fold_ids,
)
print(result.summary["auc"])
print(result.summary["contrasts"]["inference_removal_minus_full"])
```

EEG has shape `(participants, windows, channels, samples)` in microvolts; labels are binary and identifiers are unique per participant. The [usage guide](docs/reference_assessment.md) defines preprocessing requirements, model interfaces, output fields and interval interpretation. Negligible common components, including already-CAR input, make the assessment inapplicable and raise an explicit exception.

Run the software checks and simulated example without participant data:

```sh
python -B -m unittest discover -s tests -v
python -B -m examples.reference_dependence --output outputs/reference_assessment.json
```

The example prints four AUCs for each of two known information locations and saves aggregate summaries. The tests check component transformations, participant separation, training-only scaling and paired bootstrap arithmetic. They run in seconds on a CPU; the optional one-epoch EEGNet check runs only when PyTorch is installed. Exact reproduction of the article's cohort and fitted-model results uses the analysis commands above.

## Data access

TDBRAIN V3.1 requires the provider's access process. Wroclaw and Nasrabadi are obtained from their respective repositories. [Data access](docs/data.md) lists providers, versions, required files and sample definitions. Locally generated participant data and checkpoints must remain subject to the corresponding data agreements.

## Citation

Cinbat, A., Şentürk, Ü., & Yücedağ, İ. *The EEG component removed by average referencing carries ADHD classification in resting EEG*. Manuscript.

Machine-readable author metadata are in [CITATION.cff](CITATION.cff). Please also cite the original datasets and the methods used in your analysis.

## License

The authors' code and documentation are provided under the [MIT License](LICENSE). Dataset access and reuse are governed by the original providers' terms. Numerical libraries, MNE, PyTorch and SHAP retain their upstream licenses.
