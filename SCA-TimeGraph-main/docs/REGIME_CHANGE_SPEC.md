# Regime-Change Experiment Specification

## Dataset

File:
Datasets/regime_change_a1.csv

Number of time points:
5000

Number of variables:
4

Variables:
X1, X2, X3, X4

Noise:
Gaussian

Noise scale:
0.1

Random seed:
42

Change point:
t = 2500

## Regime 1: t < 2500

X4(t) = 0.25 * X1(t-2) + e4

X2(t) = 0.30 * X3(t-1) + e2

X1(t) = 0.40 * X2(t) + e1

X3(t) = 0.35 * X4(t) + e3


## Regime 2: t >= 2500

X4(t) = 0.25 * X1(t-2) + e4

X2(t) = 0.30 * X3(t-1) + e2

X1(t) = 0.40 * X2(t) + e1

X3(t) = 0.35 * X2(t-1) + e3


## Stable causal links

X1(t-2) -> X4(t)
X3(t-1) -> X2(t)
X2(t) -> X1(t)

## Removed at change point

X4(t) -> X3(t)

## Added at change point

X2(t-1) -> X3(t)
