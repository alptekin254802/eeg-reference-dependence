# External inference and component power

Dependencies: NumPy, SciPy, scikit-learn, PyTorch. Dataset-specific readers are needed to construct waveform inputs. Commands run from repository root.

```sh
python analysis/external/infer.py --input target.npy --checkpoints fold0.pt fold1.pt fold2.pt fold3.pt fold4.pt --output outputs/target_predictions.json
python analysis/external/report.py --predictions outputs/target_predictions.json --labels labels.npy --output outputs/target_metrics.json
python analysis/external/power.py --linked linked.npy --car car.npy --labels labels.npy --output outputs/power.json
python analysis/external/power.py --linked pediatric_windows.npy --fs 128 --labels labels.npy --output outputs/pediatric_power.json
```

Inference inputs are float32 [N,15,19,800] in source-compatible microvolts before checkpoint scaling. `--architecture temporal_cnn` selects the secondary model; `--device cuda` selects GPU. Five checkpoints produce a single-initialization fold-probability ensemble; supplying all fifteen gives the three-initialization ensemble. Outputs retain individual checkpoint probabilities as well as their mean. Checkpoint inputs must be trusted and contain `state` and `scale`.

Inference applies each source checkpoint without target fitting or threshold tuning. Bootstrap metrics use 2,000 diagnosis-stratified replicates (20261110), conditional on the fitted source models. The threshold is 0.5.

Wroclaw inference uses the retained 4-64-second eyes-closed segment, linked-mastoid and CAR versions of the shared 19 channels. Native unit uncertainty remains unresolved: a provisional conversion cannot establish verified-unit clinical transport. The roster and preparation commands below implement cohort selection and waveform preparation from locally acquired archives/records.

Power inputs contain one retained person's windows per first-axis row. With `--car`, the exact 200-Hz cached-Wroclaw route checks waveform and PSD decomposition plus gain invariance. Without `--car`, CAR is computed in float64 from linked waveforms at `--fs`; windows must be four seconds and filtered 1-40 Hz. Pediatric preparation uses fourth-order Butterworth SOS forward/backward filtering of each complete selected record before slicing windows, then per-window demeaning. `power.valid_windows` implements exclusion of all-channel zero runs of >=2 samples with a one-second guard and constant raw channels. Require at least ten retained windows. Nasrabadi uses 4-60 seconds of the task record at 128 Hz. The Wroclaw and Nasrabadi values are 1.067 [0.715,1.601] and 1.007 [0.766,1.314].

Welch settings: 2-second periodic Hann segments, 1-second overlap, constant detrending, density scaling, mean averaging, 0.5-Hz bins. Bands: delta [1,4), theta [4,8), alpha [8,13), beta [13,30), gamma [30,40], total [1,40]. Average CAR channel powers before taking logs; average window log10 power within people. No positive floor is introduced. Output contains all band ratios, total common/CAR ratio-of-ratios, 20,000 pointwise diagnosis-stratified bootstrap intervals (20261116), leave-one-out ratio range and numerical checks. Ratios are ADHD/control geometric-mean power ratios. Common power does not identify a physiological source.

Prepared waveform arrays, labels, individual predictions and checkpoints require their respective data-access permissions and are not bundled. Acquire the released datasets separately under their access conditions; the commands below construct the analyzed inputs.

## Metadata-driven waveform preparation

```sh
python analysis/external/roster.py --kind wroclaw --input Raw_data.zip --output outputs/wroclaw_tables
matlab -batch "addpath('analysis/external'); convert_tables('outputs/wroclaw_tables/conversion.json')"
python analysis/external/roster.py --kind wroclaw-converted --input outputs/wroclaw_tables --output outputs/wroclaw_manifest.json
python analysis/external/prepare.py --kind wroclaw --manifest outputs/wroclaw_manifest.json --output outputs/wroclaw
python analysis/external/roster.py --kind nasrabadi --adhd data/ADHD --control data/Control --output outputs/nasrabadi_manifest.json
python analysis/external/pediatric.py --manifest outputs/nasrabadi_manifest.json --output outputs/nasrabadi_power.json
```

Wroclaw selection sorts the sixteen archive members ending `/Session 1/2_ClosedEyes_Raw.mat`; its `ADHD` and `Control` directories define labels. MATLAB is required for native table conversion, retaining channel/time/unit metadata. Preparation refuses nonempty physical-unit metadata requiring a different documented interpretation; it does not silently change the provisional amplitude factor of one. Conventional T3/T4/T5/T6 map to T7/T8/P7/P8. Output filenames `main_linked_basic.npy` and `main_average_basic.npy` retain array ordering; labels and a QC manifest accompany them.

Nasrabadi expects the complete released group directories, with 61 ADHD and 60 controls and one samples-by-19 numeric matrix per MAT file. The generated manifest binds every input hash/shape and uses 128 Hz, [4,60) seconds.

Pediatric manifest rows contain `path`, `sha256`, binary `label`, `format` (`mat`), `fs`, `start`, `stop` and source matrix `shape`. Hash/shape/nonfinite failures stop execution. All-channel zero runs and constant-channel decisions precede filtering and require at least ten eligible four-second windows. The common-signal presence criterion, explicit FFT check, global-gain check and independent natural-log reconstruction are executed before aggregate output.

## Source-checkpoint selection

```sh
python analysis/external/infer.py --input outputs/wroclaw/main_linked_basic.npy --training-plan outputs/core_plan.json --training outputs/training --arm linked_basic --seeds 1109 --output outputs/wroclaw_linked_predictions.json
python analysis/external/infer.py --input outputs/wroclaw/main_average_basic.npy --training-plan outputs/core_plan.json --training outputs/training --arm average_basic --seeds 1109 --output outputs/wroclaw_car_predictions.json
python analysis/external/report.py --predictions outputs/wroclaw_linked_predictions.json --labels outputs/wroclaw/labels.npy --comparison outputs/wroclaw_car_predictions.json --output outputs/wroclaw_linked_metrics.json
python analysis/external/power.py --linked outputs/wroclaw/main_linked_basic.npy --car outputs/wroclaw/main_average_basic.npy --labels outputs/wroclaw/labels.npy --output outputs/wroclaw_power.json
```

Use `--seeds 1109 2209 3309` for the three-initialization ensemble and `--architecture temporal_cnn` for the secondary neural architecture. Repeat metric calculation for the CAR predictions. The label vector must align with retained waveform rows. Each participant receives the mean probability across the five source-fold checkpoints for each requested initialization. The prespecified provisional reference contrast is linked-mastoid minus CAR AUC for initialization 1109. The aggregate additionally records the three-initialization ensembles, temporal CNN, spectral and common-component comparators, and a source-prevalence constant benchmark.
