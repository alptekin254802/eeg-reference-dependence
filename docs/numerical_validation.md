# Numerical validation

This document provides reproduction details accompanying Online Resource 1. Scientific definitions, results and interpretation remain in the supplement. See [the documentation index](README.md) for analysis commands.

## Surface-Laplacian consistency

Acquisition- and average-reference inputs produced transformed arrays with relative differences below $10^{-4}$.

## Classical fit and probability checks

The six classical comparison arms use 30 logistic and 15 Ridge fits. The age/sex-only arm uses five logistic and five Ridge fits. Bootstrap estimates condition on their stored out-of-fold predictions and do not refit them. Saved-parameter reconstruction reproduced the held-out probabilities. Unadjusted and demographic/quality models in the age/sex comparison reuse the same baseline fits and do not provide independent repetitions.

## Attribution integration and completeness

The primary implementation starts at 32 points and increases to 64 or 128 if a configuration fails the completeness limits. The zero-background sensitivity starts at 64 points.

The completeness residual denominator is the larger of the absolute input-minus-background logit difference and 0.01.

Completeness compares the attribution sum with the input-minus-background logit difference. Configuration limits are median relative residual 0.05 and 95th percentile 0.20, using the denominator defined above. All 126 evaluated configuration–method combinations met these limits, encompassing 6828 participant explanations. Completeness establishes numerical consistency, not biological validity. The explanation documentation gives signed-rank versus predicted-class top-region conventions, pairwise kappa aggregation and its sensitivities.

