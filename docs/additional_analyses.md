# Additional analyses

This document provides the explanation and spectral sensitivities supporting Online Resource 1, Sections S1, S3 and S6.

## Explanation sensitivities

The fixed explanation panel contains four ADHD and four control participants from each outer-test fold. Sampling seeds and draw order are listed in `docs/methods_details.md`. The selections precede the follow-up explanation outcomes and remain unchanged for the registered extension. Each fold's background contains one window from each of eight training participants, balanced by diagnosis. Background sampling uses four training participants per diagnosis without replacement and one uniformly selected window per participant. Additional-background analysis increases this to 32 distinct training participants, 16 per group. The zero-background analysis uses a single all-zero input. Background participants never overlap the corresponding test fold.

IG uses midpoint integration along each background-to-input path. Integration settings and numerical completeness limits are specified in `docs/numerical_validation.md`. Window contributions are scaled to the mean participant logit. DeepSHAP uses the same target and backgrounds and retains its additivity check. Average-reference inputs and perturbed inputs are projected onto the same zero-sum space.

The scalp regions have prespecified channel membership. Frontal contains Fp1, Fp2, F7, F3, Fz, F4 and F8; central contains C3, Cz and C4; temporal contains T7 and T8; parietal contains P7, P3, Pz, P4 and P8; occipital contains O1 and O2. Ties follow that region order. Random masks are sampled without replacement within a mask, with 100 draws per person and region size. Matching channel counts does not remove every consequence of unequal region size or spatial location.

Secondary faithfulness measures test different scales and perturbations. Normalized faithfulness divides the logit score by the larger of the absolute input-minus-background logit difference and 0.1. Energy matching uses the ten random masks closest to the attributed mask's removed energy. Probability scores apply the logistic function to predicted-class-oriented outputs. Deletion curves remove 1, 3, 5, 10 and 19 ranked channels and integrate the probability drop over removed-channel fraction. Donor replacement substitutes the corresponding channels of each training background window. The tables below summarize these comparisons.

Randomization measures dependence on fitted parameters and training labels. Earlier controls compared the trained model with a fully initialized network and with a model trained on permuted source labels. Across references and ocular conditions, mean signed regional IG rank correlation was 0.022 for the parameter control and 0.011 for the label control. Their 95% intervals were -0.080 to 0.131 and -0.081 to 0.108. These controls were part of the earlier exploratory source work.

The registered extension resets progressively larger parameter blocks. It resets the classifier, the classifier plus the separable block and its batch normalization, or the full network. Each stage uses three random initializations. Correlations are averaged across reset seeds within participant before summarization. There is no universal pass/fail rank-correlation cutoff. The numerical accuracy limits concern completeness and are separate from these dependence tests.

Temporal comparisons use two nonoverlapping 52-second intervals, from 4 to 56 seconds and from 64 to 116 seconds. Each interval contains 13 four-second windows and receives its own within-record demeaning and ocular regression. The seed-1109 model trained using the main 60-second representation is evaluated without retraining. All 189 participants meet the paired temporal eligibility rule. Median signed regional IG correlations are 1.00 for linked-mastoid reference and 0.90 for average reference, both with and without ocular regression. This is within-session stability, not test--retest reliability across visits.

Top-region agreement compares the predicted-class-oriented regional maxima of separately fitted models in different coordinates. Each model supplies its own predicted-class sign. Rank correlations instead use signed ADHD-logit regional scores without that sign adjustment. The marginal-agreement table below reports agreement expected from the two marginal region frequencies. A uniform 20% reference is not imposed because the five regions are not chosen equally often. Neither observed nor marginal agreement establishes cortical localization.

The initialization comparison calculates Cohen's kappa within each pair before averaging. Three pairs of initializations contribute within each reference, and three same-initialization pairs contribute across references. The pooled within-reference estimate averages six pairwise kappas. Bootstrap resampling draws 20 ADHD and 20 controls with replacement and carries all seed/reference labels jointly, recomputing marginal frequencies for each pair. The 20,000 replicates use the seed listed in `docs/methods_details.md`. No observed or resampled pair had undefined kappa. Online Resource 1, Section S6 gives the aggregate contrasts; `results/aggregate.json` also preserves pairwise values. These intervals are conditional on the selected panel and fitted models.

Raw top-region agreement was 67.1% within reference and 30.0% across references. The within-minus-cross-reference kappa difference was 0.348 (95% interval 0.205--0.489) for DeepSHAP and 0.347 (0.205--0.486) for fixed ADHD-logit IG. Zero backgrounds gave median signed regional rank correlations of 0.00--0.45 against the eight-person background, compared with 0.75--0.90 for a 32-person background.

## Regional ranking comparisons

| Comparison | Acq. | Avg. | Acq.+EOG | Avg.+EOG |
| --- | --- | --- | --- | --- |
| Seed 2209 | 0.850 | 0.600 | 0.900 | 0.700 |
| Seed 3309 | 0.900 | 0.600 | 0.900 | 0.700 |
| Zero background | 0.350 | 0.400 | 0.450 | 0.000 |
| 32-person background | 0.900 | 0.750 | 0.900 | 0.850 |
| Reset classifier | 0.233 | -0.067 | 0.167 | 0.167 |
| Reset separable | 0.033 | 0.017 | -0.067 | 0.100 |
| Reset all | 0.017 | -0.017 | 0.117 | -0.000 |
| Early vs late segment | 1.000 | 0.900 | 1.000 | 0.900 |

Entries are median participant Spearman correlations across five scalp regions. Comparisons use the seed-1109 model and eight-person background as the reference, except early versus late segments. Random-reset correlations are averaged over three resets within participant before taking the median. The first seven rows use the fixed 40-person panel; temporal comparisons use 189 participants. No universal correlation cutoff is assumed.

## Deletion and donor replacement

| Processing | Method | Deletion difference | Donor difference |
| --- | --- | --- | --- |
| Acq., no EOG | IG | 0.082 [0.030, 0.130] | 0.104 [0.025, 0.180] |
| Acq., no EOG | DeepSHAP | 0.084 [0.032, 0.133] | 0.104 [0.025, 0.180] |
| Avg., no EOG | IG | 0.048 [0.017, 0.079] | 0.048 [-0.010, 0.102] |
| Avg., no EOG | DeepSHAP | 0.049 [0.017, 0.081] | 0.053 [-0.001, 0.106] |
| Acq., EOG | IG | 0.133 [0.088, 0.177] | 0.102 [0.043, 0.161] |
| Acq., EOG | DeepSHAP | 0.135 [0.090, 0.179] | 0.102 [0.043, 0.161] |
| Avg., EOG | IG | 0.044 [0.015, 0.075] | 0.052 [-0.000, 0.105] |
| Avg., EOG | DeepSHAP | 0.045 [0.014, 0.077] | 0.052 [-0.000, 0.105] |

Mean differences and conditional 95% intervals in the 40-person panel, seed 1109. Both subtract channel-count-matched random perturbation effects. Deletion values are areas under probability-drop curves; donor values are predicted-class probability drops. Positive values favor attribution-guided perturbation.

## Spectral sensitivities

Training-only nested regularization, label permutations and demographic/quality residualization supplement the spectral predictor comparisons in Section S3. Full aggregate estimates are provided in `results/aggregate.json`. These models use participant-level outer folds; held-out participants do not determine preprocessing, residualization or regularization.

## Recording-duration and frequency sensitivities

To assess dependence on recording duration, we restricted analysis to the largest documented duration stratum containing both diagnostic groups. Selection used participant counts rather than discrimination and retained 155 participants (30 ADHD, 125 controls) whose documented recording length was 121 seconds. The analyzed interval remained 60 seconds. Refit three-initialization ensembles yielded AUC 0.935 (0.895--0.969) under linked-mastoid reference and 0.663 (0.560--0.758) under common average reference; the paired difference was 0.273 (0.173--0.377). Other acquisition and clinical fields were not thereby matched.

A separate exploratory spectral check did not identify concentration of the association in the upper part of the analyzed band. A low-frequency (1--20 Hz) feature vector yielded AUC 0.794, whereas a higher-frequency ($>$20--40 Hz) vector yielded 0.766. Their paired difference was -0.028 (95% interval -0.097 to 0.037). Adding the higher-frequency features to the low-frequency vector changed AUC by 0.008 (-0.017 to 0.034). These analyses cannot test muscle activity above the 40-Hz preprocessing cutoff.

The available metadata do not identify the physical source of the common signal. Total session count describes archive history and is not a verified acquisition condition. Archive membership and missing recording metadata were associated with diagnosis; residualization against those fields can remove group information without isolating a device effect. The common waveform combines the scalp-average potential and the shared reference contribution. Its classification performance does not establish a muscular, instrumental or neural origin.

## Full-cohort explanation coverage

Integrated Gradients maps were computed for the 189-person source cohort. The reported comparison with the fixed explanation panel uses only the 40-person overlap; it is not a separate 189-person faithfulness estimate. Aggregate explanation results are supplied in `results/aggregate.json`.

## Residualized spectral predictors

These comparisons use age, sex and quality variables fitted within source training folds, with all 189 participants.

| Predictor | AUC [95% CI] | Balanced accuracy | Brier |
| --- | --- | --- | --- |
| Residualized acquisition spectral | 0.702 [0.627, 0.778] | 0.592 | 0.247 |
| Residualized average spectral | 0.632 [0.549, 0.714] | 0.588 | 0.272 |

## Marginal expected top-region agreement

| Initialization | Ocular regression | Observed | Marginal expected |
| --- | --- | --- | --- |
| 1109 | No | 0.300 | 0.226 |
| 1109 | Yes | 0.450 | 0.308 |
| 2209 | No | 0.325 | 0.194 |
| 2209 | Yes | 0.375 | 0.251 |
| 3309 | No | 0.275 | 0.216 |
| 3309 | Yes | 0.300 | 0.211 |

Integrated Gradients, 40 participants. Expected agreement sums the products of reference-specific marginal region-selection frequencies; it does not assume equal selection of the five regions.
