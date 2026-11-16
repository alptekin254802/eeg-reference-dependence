# Implementation details

This document provides reproduction details accompanying Online Resource 1. Scientific definitions, results and interpretation remain in the supplement. See [the documentation index](README.md) for analysis commands.

## Network architecture and optimization

| Component | EEGNet-style model | Temporal CNN |
| --- | --- | --- |
| Temporal extraction | Eight kernels of length 64; asymmetric padding 31/32 | Conv1d: 19 to 16 channels; kernel 31 |
| Spatial/second block | Depthwise convolution across 19 channels; multiplier two | Conv1d: 16 to 32 channels; kernel 15 |
| Final convolution | Depthwise temporal kernel 16; pointwise output 16 | Conv1d: 32 to 32 channels; kernel seven |
| Pooling | Average pooling by four and eight | Average pooling by four after each block; temporal mean after final block |
| Activation/dropout | ELU; dropout 0.5 after each pooling layer | ELU; dropout 0.5 after each pooling layer |
| Normalization | Three batch-normalization layers; epsilon 0.001; momentum 0.01 | Same epsilon and momentum |
| Head | 400 to one linear output | 32 to one linear output |
| Constraints | Spatial-filter max norm one; head max norm 0.25 | No max-norm constraint |
| Trainable parameters | 1809 | 24,545 |

Both models use Adam with learning rate 0.001 and weight decay 0.0001. A batch contains four participants and all their windows. The weighted binary cross-entropy positive-class weight is the training control-to-ADHD ratio.

## Ocular regression

The least-squares ocular fit uses a singular-value cutoff of $10^{-8}$.

## Surface Laplacian

MNE current-source-density parameters are automatic sphere fitting, regularization $10^{-5}$, stiffness four and 50 Legendre terms.

## Training partition seeds

The balanced-stopping validation subset uses seed 741 plus fold index.

Alternative five-fold partitions use seeds 20261110 and 20261111. Initialization seeds are offset by fold to keep training reproducible.

## RMS comparison environment

The matched RMS comparison uses PyTorch 2.0.1 and NumPy 1.26.4 for both normalized and baseline fits.

## Explanation panel and backgrounds

NumPy sampling without replacement uses seed 3371 plus the zero-based fold index, drawing controls before ADHD participants.

Seed 4431 plus fold index selects four training controls followed by four ADHD participants without replacement. The same generator then selects one of 15 windows uniformly for each person.

## Random perturbation masks

The draw seed is 91000 plus the participant's fixed sorted index.

## Bootstrap seeds

The 20,000 replicates use seed 20261116.

## Core partition and initialization

The core five-fold partition seed is 20261109; initialization seeds are 1109, 2209 and 3309.

## Computing environment

Signal preparation used MNE-Python 1.10.2, NumPy 2.3.4 and SciPy 1.16.3. Neural fitting used Python 3.11.0, PyTorch 2.0.1 with CUDA 11.8, and NumPy 1.26.4. MNE provides EEG reading and spatial transforms. Numerical checks are documented in `numerical_validation.md`; `results/aggregate.json` supplies aggregate results.
