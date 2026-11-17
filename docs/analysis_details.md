# Analysis details
This document supplies extended scientific definitions supporting Online Resource 1, Sections S2–S7.

## Representations and quality variables

These definitions support Online Resource 1, Section S2.

The four source arms use identical recording intervals and channel identities. Input tensors contain 15 windows, 19 channels and 800 time samples per participant. Filtering precedes extraction to avoid filtering each short window independently. Within-record demeaning precedes ocular regression. This transform is fitted separately to each recording without group labels and is followed by the specified reference operation.

Six quality variables describe the uncorrected linked-mastoid record: maximum channel-window peak-to-peak amplitude; fraction of channel-windows above 200 microvolts peak-to-peak; median channel root-mean-square amplitude; vertical ocular root-mean-square amplitude; horizontal ocular root-mean-square amplitude; and maximum absolute EEG--ocular correlation. They are computed over the same retained interval. The stricter-quality sensitivity requires at most 10% of channel-windows above 200 microvolts. It retains 187 participants, including all 56 ADHD participants and 131 controls.

Spectral features use Welch periodograms with two-second periodic-Hann segments, 50% overlap, constant detrending and mean averaging. Frequency bands are 1--4, 4--8, 8--13, 13--30 and 30--40 Hz, with left-closed/right-open boundaries except the inclusive final 40-Hz bin. Power is the sum over bins multiplied by the 0.5-Hz resolution. Values are floored at $10^{-12}$ in squared-microvolt units before base-10 logarithms. Log-bandpowers are averaged over the 15 windows. The five-band features of the channel mean form the common-component predictor.

Archive membership uses three released categories. The ADHD group contains 39 records labelled ADHD\_NF, one labelled ADHD\_NF/OCD and 16 without membership information. All 133 controls lack membership information. A training-fitted one-hot encoder treats missing membership as its own category and ignores unseen test categories. These strings describe archive annotation, not verified independent acquisition sites.

The surface Laplacian sensitivity uses a standard 10--20 montage rather than individual electrode digitization. Transform parameters are listed in `docs/methods_details.md`. The input is converted from microvolts to volts before the transform. The resulting units are volts per square metre. Numerical reference-invariance checks are described in `docs/numerical_validation.md`. The transform has an intentional one-dimensional nullspace and is treated as a different representation, not another physical voltage reference.

## Network definitions and training controls

These specifications support Online Resource 1, Section S3.

The two networks share participant-level aggregation and training settings. Each window receives one logit. The mean of the window logits is the participant logit, and the logistic function is applied once. External ensembles average probabilities across source-fold models; they do not concatenate target windows into the source training set.

Network layers, parameter counts and optimizer settings are listed in `docs/methods_details.md`, under ``Network architecture and optimization.''

The balanced-stopping sensitivity selects epochs using class-balanced validation loss. The inner loss averages the mean control and ADHD binary cross-entropies with weight one half each. A single stratified 20% validation subset is taken from each outer-training set, with the partition seed specified in `docs/methods_details.md`. The search has a maximum of 30 epochs and patience of eight, with stopping no earlier than epoch eight. The earliest strict minimum determines the selected epoch. Normalization is fitted on the inner training set for selection and refitted on all outer-training participants for the final model. This is nested holdout selection, not inner K-fold cross-validation.

The long-trajectory analysis reports each specified checkpoint. It trains EEGNet to 100 epochs and retains epochs 10, 20, 30, 60 and 100. Alternative five-fold partition seeds and fold-dependent initialization are listed in `docs/methods_details.md`. The corresponding tables in Online Resource 1 report all these AUCs rather than selecting the most favorable checkpoint or partition.

All 42 original reference-contrast sensitivity estimates were positive, with intervals excluding zero.

Age/sex matching pairs each ADHD participant with one control of the same recorded sex and age within three years. Maximum-cardinality, minimum-age-distance assignment yields 56 pairs. Both members of a pair remain in the same outer fold. Uncertainty is calculated by resampling pairs. Strict-quality comparisons instead preserve the original folds after removing ineligible people. Additional spectral sensitivities select regularization within training folds and residualize features against demographics and quality using training data only. The corresponding tables in Online Resource 1 report representative predictor families; `results/aggregate.json` retains their full metric matrix.

## Participant-wise RMS normalization control

The participant-wise RMS comparison used the same 189 participants, two references without ocular regression and five outer folds. For each participant and reference, one scalar was calculated across all fifteen windows, nineteen channels and 800 samples per window:
$$
 s_i=\sqrt{\frac{1}{15\cdot19\cdot800}\sum_{w,c,t} X_i(w,c,t)^2},\qquad \widetilde X_i=X_i/s_i.
$$
All scales were positive and finite. Division was performed separately within each reference, without diagnosis information, and preserved within-record channel and window amplitude ratios. The original training-derived scalar normalization was retained after this operation and was numerically close to one. Each reference was retrained for 30 epochs with seed 1109 plus the fold-specific offset, unchanged architecture, batches, optimizer and weight constraints. No channel-specific normalization, model selection or additional partition was introduced.

The ten normalized fits were compared with the ten saved seed-1109, 30-epoch baseline checkpoints, using the same neural environment specified in `docs/methods_details.md`. The primary contrasts are normalized minus original AUC within each reference, linked-mastoid minus CAR within each scaling condition, and the change in that reference contrast. Shared 20,000 diagnosis-stratified participant bootstrap draws carry all four out-of-fold predictions jointly. Their intervals condition on these fitted models and do not capture retraining variability. This control changes signal scaling before model fitting; it is not a causal decomposition of classification performance.

The linked-mastoid-minus-CAR AUC contrast was 0.177 before normalization and 0.179 afterwards. Its change was 0.002 (95% interval -0.052 to 0.057). Normalized-minus-original AUC changes were 0.022 (-0.006 to 0.051) for linked-mastoid and 0.020 (-0.034 to 0.074) for CAR. These conditional results retain the reference advantage after removing global record scale, without establishing equivalence or isolating a specific remaining signal feature.

## Power and auxiliary methods
These methods support Online Resource 1, Section S5.

The main follow-up roster contains 186 adults, including 55 with ADHD and 131 controls. Three of the original 189 recordings failed fixed auxiliary-signal eligibility. Eligibility required finite nonflat Erbs and MASS signals, at least 30 detected events in the 4--64 s interval, median-RR heart rate from 40 to 160 beats/minute, at least 90% of RR intervals from 0.375 to 1.5 seconds, detector agreement F1 of at least 0.8, at most 15% disagreement between count-based and median-RR rates, and finite MASS-band power. The two event lists used the same QRS detector with thresholds 0.6 and 0.4 and a 50-ms one-to-one matching tolerance. Agreement is a numerical consistency measure, not validation against true beats. Fifteen deterministic waveform examples and a morphology overview were inspected. Some eligible examples contained missed or spurious events. No complete event-level manual validation was available, and no recordings were selectively removed after visual inspection.

Common and CAR powers use the same Welch settings and windows. The CAR periodograms are averaged over the 19 channels before summing band bins. Each window power is logged before averaging over windows. This mean-log definition also applies to total 1--40 Hz power. It differs from logging the arithmetic mean of window powers. Group ratios are $10^{\bar z_{\mathrm{ADHD}}-\bar z_{\mathrm{control}}}$, where $z$ is the participant's mean log-power feature. Hedges' $g$ uses the pooled between-person standard deviation and a small-sample correction. All group intervals use 20,000 diagnosis-stratified participant resamples. The same draw carries a person's common and CAR measures together. Intervals are descriptive and pointwise, without multiplicity adjustment or correction for prior discovery selection. No between-component interaction test was performed.

## Archive-annotation subgroup comparison

The released dataset-membership strings were split at slashes to identify the ADHD\_NF token, including ADHD\_NF/OCD. The annotation is not interpreted as a verified acquisition site or treatment. The 186-person follow-up contains 39 annotated ADHD participants, 16 unannotated ADHD participants and 131 controls. Each ADHD subgroup was compared with controls and with the other ADHD subgroup using the existing total 1--40 Hz mean-log common and CAR features. Shared bootstrap draws resampled participants independently within these three groups, with 20,000 replicates and seed 20261116. The corresponding tables in Online Resource 1 report all six contrasts; the overlapping 189-person sensitivity retains 40, 16 and 133 participants, respectively. This analysis does not remove demographic or acquisition confounding.

## Common-bandpower residualization

The five common-bandpower predictors enter logistic regression jointly. Training folds use median imputation with a missing indicator after each numeric feature, standardization, and class-balanced logistic regression with $C=1$ and a maximum of 10,000 iterations. For adjusted arms, a Ridge model with penalty one predicts each standardized band feature from standardized covariates. Its prediction is subtracted, then residual features are standardized again before logistic regression. Age/sex-only residualization uses numeric age and a male indicator. The demographic and quality bundle adds all six measures defined in Online Resource 1 together. The same 186 people and original folds are used in every arm. All missing indicators were constant zero in this eligible sample. No hyperparameter search or repartitioning was performed.

Additional models add median-RR heart rate or MASS 60--90 Hz log power plus detector-agreement F1 to the complete demographic and quality bundle. MASS power is calculated from calibrated as-released 500-Hz data using 15 four-second windows, two-second Hann segments and 50% overlap. The inclusive 60--90 Hz bins are integrated at 0.5-Hz resolution, logged within window, then averaged. The auxiliary reference, polarity and physiological source remain unresolved, so MASS is not treated as a pure muscle measurement. The heart-rate-only classifier and the MASS-power-plus-F1 classifier retain their original roles as covariate-only comparisons.

Bootstrap estimates condition on stored out-of-fold predictions and do not refit the models. Unadjusted and demographic/quality models in the age/sex comparison reuse the original fits. Fit counts and saved-parameter reconstruction are documented in `docs/numerical_validation.md`.

Coefficient comparisons require attention to scale. Unadjusted classifier coefficients are per original feature standard deviation; adjusted coefficients are per residual standard deviation. The mean standardized gamma coefficient changed from -0.424 under demographic/quality residualization to -1.094 after adding MASS and F1. Converting within each fold to slopes per original log-power unit gave -0.974 and -3.080. These are conditional multivariate slopes with correlated features and arm-specific regularization scales. They are not marginal diagnostic effects or source maps. The positive unadjusted theta coefficient coexists with lower marginal theta power in ADHD.

Median channel RMS had arithmetic group means of 9.36 microvolts in ADHD and 11.55 microvolts in controls.

Median channel RMS is the median over 19 temporally demeaned channel standard deviations from the linked-mastoid, 1--40 Hz signal, before ocular regression or CAR. It contains intrinsic EEG amplitude. If $X_j=m+r_j$ and the time average is $E$, then
$$
 \frac{1}{C}\sum_j E[X_j^2]=E[m^2]+\frac{1}{C}\sum_j E[r_j^2].
$$
The cross terms cancel after averaging over channels because $\sum_j r_j=0$. This is an identity for mean squared channel RMS, not median RMS. It explains overlap without making RMS identical to a common-bandpower feature. Ridge coefficients conditional on correlated covariates are not independent fractions of explained variance.

## Auxiliary event-template amplitude

The saved Erbs-channel template is an arithmetic mean of baseline-corrected event epochs within 4--64 seconds. The full record was filtered from 5 to 35 Hz using a zero-double FIR filter. Each epoch spans -0.2 to +0.4 seconds around a detected event, with the -0.2 to -0.1 second mean subtracted. Peak-to-peak amplitude is measured from the averaged waveform. It is not the average of individual R-wave amplitudes and retains event-timing and morphology uncertainty.

In the 186-person subset, geometric mean template amplitudes were 209.73 microvolts for ADHD and 206.74 for controls. Their ratio was 1.014 (95% interval 0.843--1.209). The raw mean difference was 7.35 microvolts (-30.75 to 45.93), with Hedges' $g$ 0.060 (-0.252 to 0.391). The overlapping 189-person sensitivity gave a ratio of 1.055 (0.874--1.268). These 20,000-replicate diagnosis-stratified bootstrap intervals describe saved features and involve no new model fits. They neither establish amplitude equivalence nor exclude gain or acquisition differences affecting the EEG.

Explanation sensitivity methods and estimates are in [Additional analyses](additional_analyses.md), supporting Online Resource 1, Section S6.

## External power measurements
Wroclaw uses the first pre-task eyes-closed recording and the source interval and filter settings. Conventional T3, T4, T5 and T6 labels map to T7, T8, P7 and P8. Only acquisition and average reference without ocular regression are evaluated. Native values are provisionally interpreted as microvolts. Header calibration confirms the source TDBRAIN scale, but no corresponding documentation establishes the Wroclaw physical unit. For the registered seed-1109 ensemble of five source-fold models, the provisional acquisition-reference-minus-CAR AUC difference was -0.203 (95% interval -0.438 to 0.016).

## External common-power comparisons

These exploratory comparisons followed dated local specifications after TDBRAIN outcomes and earlier cohort information were known. They used no new classifier fits. Within each cohort, common power is the power of the 19-channel spatial mean and CAR power is the mean residual-channel power. The five bands and mean-log window aggregation match the source power estimands. Intervals use 20,000 diagnosis-stratified participant bootstrap replicates. The common-to-CAR ratio of ratios subtracts the participant's log CAR power from log common power before the group comparison. This relative measure cancels a uniform within-record gain but does not identify a physiological source. The corresponding tables in Online Resource 1 report all total-power comparisons; all bandwise estimates appear in `results/aggregate.json`.

Wroclaw retained all eight ADHD and eight control recordings and the existing fifteen four-second windows from 4 to 64 seconds. A global unit conversion multiplies all powers by the same squared factor and therefore cancels from the group ratio. Unequal participant gains would not cancel. The original provisional unit assumption and its quality decisions remain limitations. The power comparison is separate from, and does not complete, the registered source-model transport endpoint.

Nasrabadi includes 61 children with ADHD and 60 controls, aged 7--12 years, recorded during a visual attention/counting task at 128 Hz. The released 19-channel recordings use earlobe reference electrodes A1/A2; the exact linking procedure is insufficiently documented to equate this representation with TDBRAIN's linked-mastoid reference. Continuous signals received a fourth-order 1--40 Hz forward--backward Butterworth filter. Fourteen four-second windows from 4 to 60 seconds were analyzed at the native sampling rate. Welch segments remained two seconds, with 50% overlap and 0.5-Hz resolution. All 121 participants and all windows were retained.


## Source preparation and inclusion

A fourth-order Butterworth bandpass from 1 to 40 Hz was applied forward and backward to continuous source records. Polyphase resampling reduced 500 Hz to 200 Hz before extracting 4--64 seconds and demeaning channels. The channel order was Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1 and O2. No separate notch filter or channel interpolation was used.

One inclusion decision applied to all four representations. Exclusion followed any nonfinite uncorrected linked-mastoid channel-window, standard deviation below 0.1 microvolt, or peak-to-peak amplitude above 1000 microvolts. Retained windows were not replaced or selected separately by condition. The linked-mastoid condition retains the as-released signal; separately recorded mastoid potentials were not reconstructed. Ocular regression uses each person's own 60-second record, followed by re-referencing, with the least-squares cutoff listed in `docs/methods_details.md`.

## Component definition and prediction settings

For consistency with the main decomposition, with channel $j$, $C=19$, scalp potential $V_j$ and linked-mastoid reference $R$,
$$
\begin{aligned}
 m(t)&=\frac{1}{C}\sum_{j=1}^{C}X_j(t)
      =\frac{1}{C}\sum_{j=1}^{C}V_j(t)-R(t),\\
 X_j^{\mathrm{CAR}}(t)&=X_j(t)-m(t)
      =V_j(t)-\frac{1}{C}\sum_{k=1}^{C}V_k(t).
\end{aligned}
$$
Online Resource 1 defines common-only retraining and numerical equivalence of the residual to CAR. Each representation uses its own training-derived scalar RMS with equal participant weights. Core partition and initialization seeds are listed in `docs/methods_details.md`. Fixed-duration models train for 30 epochs. Alternative partitions, stopping, matching and spatial-transform sensitivities are specified in Online Resource 1, Sections S2–S3; architecture and optimization settings are listed in `docs/methods_details.md`. Matched-cohort EEGNet uses new five-fold assignments keeping pair members together, not predictions from full-cohort models.

The spectral logistic model jointly uses 95 window-averaged log-bandpowers, five per scalp channel. Training-only standardization precedes class-balanced logistic regression with inverse regularization strength one. Additional demographic, quality and archive predictors are described in Online Resource 1, Sections S2–S3. Their combined and training-only residualized spectral models are not adjusted EEGNet models.

## Common-power model specification

Online Resource 1 defines Welch segmentation, frequency boundaries, the power floor, common and CAR power ratios, Ridge residualization and auxiliary predictors. All transforms use training folds. The amplitude-bearing bundle enters jointly; its effect is not an RMS-specific estimate.

## Attribution estimand and numerical limits

IG and DeepSHAP target the ADHD logit averaged over a person's windows. The same background participants and windows are used across processing conditions. Neither true diagnosis nor prediction correctness selects an attributed region. Panel and background sampling are described in [Implementation details](methods_details.md); Online Resource 1, Sections S2 and S6, define regions, faithfulness and the agreement comparisons.

The faithfulness score compares removal of the attributed region with equally sized random channel sets. Let participant output be $f_i$, predicted-class sign $s_i$, selected region $R_i$ and perturbation operator $P_R$. Then
$$
\begin{aligned}
 F_i={}&s_i\{f_i(x_i)-f_i(P_{R_i}x_i)\}\\
 &-\frac{1}{100}\sum_{b=1}^{100}s_i\{f_i(x_i)-f_i(P_{R_{ib}}x_i)\}.
\end{aligned}
$$
A nonnegative logit is assigned to ADHD. Perturbation replaces the selected region with zero in normalized coordinates; CAR inputs are projected back to the zero-sum subspace. Larger $F_i$ means a larger predicted-class logit drop than random removal. The primary ocular-regression effect is corrected minus uncorrected $F_i$, averaged over references and then over additional initializations within participant. IG is primary and DeepSHAP secondary. This estimand measures model sensitivity, not a causal neural effect.

Completeness compares the attribution sum with the input-minus-background logit difference. All evaluated configurations met the numerical limits documented in `docs/numerical_validation.md`. Completeness establishes numerical consistency, not biological validity. Online Resource 1 gives signed-rank versus predicted-class top-region conventions, pairwise kappa aggregation and its sensitivities.

## Statistical measures and computing environment

Balanced accuracy, sensitivity and specificity use probability 0.5. Brier score and unweighted log loss summarize probability errors; smaller values are better. Probabilities are clipped to $10^{-7}$ through $1-10^{-7}$ for log loss only. Class-weighted training does not make these probabilities population-prevalence estimates.

Core classification and faithfulness intervals use 2000 diagnosis-stratified participant bootstrap replicates. Power, 186-person covariate, auxiliary-template and kappa analyses use 20,000 replicates with seed 20261116. Paired comparisons carry people jointly across configurations; matched comparisons resample pairs. Folds, windows and initializations do not increase sample size. Intervals condition on fitted models. Full retraining uncertainty is omitted. Secondary comparisons have no multiplicity adjustment.

