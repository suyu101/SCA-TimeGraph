# SCA TimeGraph Experiment Plan

## 1. Development dataset

A1 / Gaussian / 4 variables / Lag 2

Purpose:
- Debug data pipeline
- Verify temporal windows
- Verify ground truth
- Verify evaluation

## 2. Static causal-discovery benchmarks

A1
B1
C1
D1

Purpose:
- Evaluate causal structure recovery
- Test robustness to different data conditions

## 3. Causal-regime-change benchmark

Required:
- Same variables over time
- Ground-truth causal graph changes at a known change point
- Stable edges remain unchanged
- At least one causal edge is added, removed, or modified

Purpose:
- Test selective causal adaptation

## 4. Metrics

Structural:
- SHD
- Precision
- Recall
- F1

Adaptation:
- Change detection accuracy
- Adaptation delay
- Stable-edge preservation
- Unnecessary adaptation

## 5. Models

- Baseline causal discovery method
- Temporal encoder baseline
- Encoder + memory
- Encoder + change detector
- Encoder + plasticity
- Full SCA