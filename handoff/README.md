# SCA Model Handoff

## Your responsibility

Person 1 owns:

- TimeGraph data loading
- Ground truth
- Temporal splitting
- Normalization
- Evaluation
- Baseline experiments

## What the SCA team receives

For the static TimeGraph experiment:

```python
X_train
X_val
X_test
