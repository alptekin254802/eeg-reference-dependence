# Explanation analyses

Dependencies: NumPy, SciPy, PyTorch, SHAP 0.46.0 (install the licensed upstream `shap==0.46.0` package), scikit-learn. Run commands from the repository root. Models use `analysis/core/models.py`.

```sh
python analysis/explanations/run.py --input person.npy --background background.npy --checkpoint model.pt --channels channels.json --index 0 --method IG --steps 32 --output outputs/ig.json
python analysis/explanations/run.py --input person.npy --background background.npy --checkpoint model.pt --channels channels.json --index 0 --method DeepSHAP --output outputs/shap.json
python analysis/explanations/agreement.py --input top_regions.npz --output outputs/agreement.json
```

`person.npy` is float32 [windows,19,800]; background is [background windows,19,800], both in the original source input units before checkpoint scaling. `channels.json` is the ordered 19-channel list. `--average` projects the model target and interventions onto the zero-sum reference subspace. Inputs and backgrounds must come from held-out and training people, respectively; input files deliberately contain no participant identities. `--index` is the fixed sorted array index used for matched random-mask seeds. Checkpoints must be trusted PyTorch files containing `state` and `scale`. Use `--device cuda` where available.

The NPZ output contains [windows,19,800] attribution; JSON contains completeness, signed and predicted-class regional scores, logit/probability/energy-matched faithfulness, deletion curves, donor replacement, and integration-step diagnostics. These files contain individual results and must be handled under the data agreement.

`selection.py` provides exact panel and training-background selection, configuration-level completeness checks, signed regional rank correlation and cosine similarity of absolute attributions. Main panel: four controls then four ADHD per fold, generator 3371+fold. Background: four training controls then four ADHD, generator 4431+fold; one random window each. Larger background: 16 ADHD then 16 controls, generator 5441+fold. Zero background: one zero window. Use identical selections across references and methods.

IG uses midpoint path integration and targets the mean window logit. Run every member of a configuration at 32 steps; accept only when median relative residual <=0.05 and p95 <=0.20. Otherwise rerun the entire configuration at 64 then 128. Zero-background starts at 64. DeepSHAP retains upstream additivity checks and scales window contributions by the window count. The CLI does not choose configuration acceptance from one person's result.

Parameter randomization uses `--reset classifier`, `--reset separable`, or `--reset all` with seeds 7109,7209,7309. Label randomization requires a separately fitted source checkpoint; the explainer never silently retrains. Compare signed regional rank correlation and cosine similarity of absolute attributions separately. Regions and tie order are frontal, central, temporal, parietal, occipital as specified in the supplement.

For temporal sensitivity, prepare paired 13-window arrays from 4-56 and 64-116 seconds, each independently demeaned and ocular-regressed, then explain using the same original 60-second-model checkpoint and training background. Compare signed regional ranks with `regional_rank_correlation`. Both intervals must meet the common quality rule; eligibility must precede explanation results. Use `analysis/core/temporal.py` for raw recording extraction; it preserves the full participant array axis and records paired eligibility.

Agreement NPZ keys: `top_regions` integer [3 initializations,2 references,N], reference order linked-mastoid/CAR, initialization order 1109/2209/3309; `labels` binary [N]. Categories 0-4 follow region order. Derive each label by argmax of signed region sums times the model's predicted-class sign; fixed ADHD-logit sensitivity omits that sign. Compute each method/ocular condition/target separately. The output recomputes marginal frequencies per pair in each joint diagnosis-stratified resample (20,000; seed 20261116), then averages pairwise kappas. It reports undefined-replicate counts rather than substituting a value.

Published numerical interpretation and sample definitions are in the article and supplement. These utilities do not assert biological localization or test-retest reliability.

## Complete configuration generation and execution

```sh
python analysis/explanations/plan.py --input outputs/cache --training outputs/training outputs/sensitivity_training --training-plan outputs/core_plan.json outputs/sensitivity_plan.json --temporal outputs/temporal --randomization-controls --output outputs/explanations
python analysis/explanations/run_plan.py --index outputs/explanations/index.json
python analysis/explanations/contrasts.py --manifest outputs/explanations/contrasts.json --method IG --output outputs/ig_contrasts.json
python analysis/explanations/contrasts.py --manifest outputs/explanations/contrasts.json --method DeepSHAP --output outputs/shap_contrasts.json
python analysis/explanations/agreement.py --manifest outputs/explanations/agreement_basic.json --method IG --output outputs/kappa.json
python analysis/explanations/agreement.py --manifest outputs/explanations/agreement_basic.json --method DeepSHAP --target adhd --output outputs/kappa_shap_fixed_target.json
python analysis/explanations/summarize.py --input outputs/explanations/initialization1109_linked_basic --method IG --output outputs/linked_ig_sensitivities.json
python analysis/explanations/summarize.py --input outputs/explanations/temporal_late_linked_basic --comparison outputs/explanations/temporal_early_linked_basic --output outputs/temporal_stability.json
```

The core training plan must include `core`, `laplacian` and (for `--randomization-controls`) `label_null` modules. Prepare temporal inputs with the core temporal command before explanation-plan generation. Omit `--temporal` only when deliberately running a subset. The generated plan covers the three EEGNet initializations, temporal CNN, zero/32-person backgrounds, full-cohort IG, classifier/separable/all-block randomization at three random seeds, paired temporal intervals and available surface-Laplacian inputs. `--randomization-controls` adds the parameter-control initialization 1109+100*fold+50000 and source-label-permutation models. No plan-generation command fits a model. `run_plan.py --configuration NAME` runs one indexed configuration. Failed completeness checks remain in `complete.json`; summaries refuse failed configurations.

All settings are generated from source preprocessing/training outputs. Individual configuration JSON contains: `input` and optional `background_input` array paths; `labels` array path; ordered `channels`; architecture; `average` boolean; and `folds`, each containing `train`, `test` and `checkpoint`. Optional fields are `methods`, `selection` (`all` or the default panel), `background` (`zero`, `large` or default), `steps`, `reset`, `random_seed`, `random_seed_by_fold`, `eligible_indices` and `device`. `input_indices` maps a compressed array's rows back to the original sorted source indices when needed. All participant-index metadata are generated locally and must remain subject to the data agreement.

The primary ocular-regression comparison averages references within each person and then initializations 2209/3309 within that person before bootstrapping. Initialization 1109 and the three-initialization mean are reported separately. `contrasts.py` preserves this aggregation for every faithfulness scale. Its randomization summaries average the signed regional rank/cosine contrasts over processing arms within each person before participant bootstrap. `summarize.py` computes confidence/correctness bins and deletion/donor controls; no bin-dependent fitting occurs. Repeat agreement commands with `agreement_eog.json` and `--target adhd` for the ocular-regressed and fixed-target sensitivities.
