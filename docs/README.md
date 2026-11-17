# Study documentation

These guides accompany Online Resource 1, Sections S1–S7, of *The EEG component removed by average referencing carries ADHD classification in resting EEG*. They provide implementation settings, numerical checks and reproduction commands. The supplement retains scientific methods, results and limitations.

| Guide | Purpose | Online Resource 1 |
| --- | --- | --- |
| [Implementation details](methods_details.md) | Network architecture, optimization, sampling seeds and software versions | S3–S6 |
| [Numerical validation](numerical_validation.md) | Completeness tolerances and numerical consistency checks | S3, S5, S6 |
| [Analysis definitions and implementation](analysis_details.md) | Detailed study methods linked from the supplement | S2–S7 |
| [Additional analyses](additional_analyses.md) | Registered ancillary explanation and spectral results | S1, S3, S6 |
| [Core analyses](core.md) | Cohort reconstruction, preprocessing, classification and sensitivity commands | S2–S4 |
| [Components and covariates](components.md) | Power, auxiliary signals, adjustment and RMS normalization commands | S5 |
| [Explanations](explanations.md) | Attribution, perturbation and agreement commands | S6 |
| [External analyses](external.md) | External preparation, transport and power commands | S7 |
| [Data access](data.md) | Provider access, input formats and cohort definitions | S2 |

The [reference-dependence interface](reference_assessment.md) applies the component comparisons to participant EEG arrays and documents the accompanying software checks.

Installation and aggregate figure/table commands are in the [repository README](../README.md). Commands run from the repository root. Participant data and model checkpoints require the documented access permissions and are not included in the aggregate-result package.
