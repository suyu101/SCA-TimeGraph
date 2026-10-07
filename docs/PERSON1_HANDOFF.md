# Person 1 — Final Handoff

## Responsibilities completed

Person 1 owns:

- TimeGraph dataset loading
- Ground-truth construction
- Temporal splitting
- Training-only normalization
- Lag-aware temporal windows
- Static causal evaluation
- Dynamic adaptation evaluation
- Baseline experiments
- Results aggregation

## Development dataset

A1 / Gaussian / 4 variables / Lag 2

Sample sizes verified:

- 500
- 1000
- 3000
- 5000

## Static A1 baseline

Dataset:

A1 / Gaussian / 4 variables / Lag 2 / n=5000

Rolling Regression:

- Precision: 0.7500
- Recall: 0.7500
- F1: 0.7500
- SHD: 2

PCMCI+:

- Precision: 0.8000
- Recall: 1.0000
- F1: 0.8889
- SHD: 1

## Controlled regime-change dataset

File:

Datasets/regime_change_a1.csv

Properties:

- 5000 time points
- 4 variables
- Maximum lag: 2
- Change point: 2500

Before change:

X1(t-2) -> X4(t)
X4(t)   -> X3(t)
X3(t-1) -> X2(t)
X2(t)   -> X1(t)

After change:

X1(t-2) -> X4(t)
X2(t-1) -> X3(t)
X3(t-1) -> X2(t)
X2(t)   -> X1(t)

Stable relationships:

X1(t-2) -> X4(t)
X3(t-1) -> X2(t)
X2(t)   -> X1(t)

Removed:

X4(t) -> X3(t)

Added:

X2(t-1) -> X3(t)

## Rolling Regression adaptation baseline

- Before-change F1: 1.0000
- Before-change SHD: 0
- After-change F1: 0.7500
- After-change SHD: 2
- Stable-edge preservation: 0.7603
- Change-detection accuracy: 0.8592
- Adaptation delay: 352
- Unnecessary-change rate: 0.2397

## Model interface

SCA must return:

A_pred

with shape:

(5000, 4, 4, 3)

Interpretation:

A_pred[t, source, target, lag]

where:

lag=0 -> contemporaneous
lag=1 -> source(t-1) -> target(t)
lag=2 -> source(t-2) -> target(t)

## Evaluation

Use:

evaluate_model(
    A_pred,
    A_true,
    change_point=2500
)

Metrics:

- Precision
- Recall
- F1
- SHD
- Stable-edge preservation
- Change-detection accuracy
- Adaptation delay
- Unnecessary-change rate

## Baseline availability

PCMCI+ is implemented and tested.

LPCMCI was not included because the installed Tigramite version does not expose LPCMCI.

## Person 2 / Person 3 responsibility

They provide:

A_pred

in the required shape and without future-information leakage.

Person 1 handles the final evaluation and result tables.
