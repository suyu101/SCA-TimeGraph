# Rolling Regression Baseline

## Dataset

Controlled regime-change dataset:

`regime_change_a1.csv`

- Time points: 5000
- Variables: X1, X2, X3, X4
- Change point: t = 2500
- Maximum lag: 2

## Baseline

Rolling linear regression.

- Rolling window: 500 time points
- Prediction interval: every 25 time points
- Edge threshold: 0.20

The threshold was selected using multiple validation time points.

## Validation

Validation times:

- 3500
- 3700
- 3900
- 4100
- 4249

Selected threshold:

`0.20`

Average validation performance:

- F1: 0.8143
- Precision: 0.8333
- Recall: 0.8000
- SHD: 1.4

## Test / Adaptation Results

Before-change:

- Precision: 1.0000
- Recall: 1.0000
- F1: 1.0000
- SHD: 0

After-change:

- Precision: 0.7500
- Recall: 0.7500
- F1: 0.7500
- SHD: 2

Adaptation:

- Stable-edge preservation: 0.7603
- Change-detection accuracy: 0.8592
- Adaptation delay: 352 time steps
- Unnecessary change rate: 0.2397

## Interpretation

The rolling regression baseline recovers the pre-change graph accurately. After the causal regime changes, performance decreases and the baseline requires substantial historical evidence before reaching the correct post-change edge configuration.

The baseline also makes unnecessary changes to some stable causal relationships.

These results provide the reference point for evaluating selective causal adaptation.
