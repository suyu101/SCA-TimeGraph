# SCA-TimeGraph

## Overview

This repository contains an experimental implementation and evaluation of **Selective Causal Adaptation (SCA)** for evolving time-series causal graphs.

The project studies whether a causal discovery system can adapt to a regime change while:

* preserving stable causal relationships,
* adapting changed relationships,
* avoiding unnecessary graph changes, and
* responding to changes with limited adaptation delay.

## Manuscript validation workflow

The manuscript protocol is separate from the legacy single-regime examples
described below. It uses tuning seeds 0--19, held-out evaluation seeds 20--39,
four graph dimensions (4, 10, 20, 50), three lag orders (1, 2, 4), four
measurement-noise scales, and Gaussian/Student-$t$/uniform innovations. The
synthetic runner writes an atomic checkpoint after every condition, so it is
safe to stop and resume the same command:

```powershell
python scripts/run_validation.py --study synthetic --seeds 20
```

Do not delete `results/validation/synthetic_raw.csv` when resuming. After the
held-out run completes, generate only the validated paper artifacts with:

```powershell
python scripts/build_manuscript_tables.py
python scripts/analyze_significance.py
python scripts/build_publication_figures.py
python scripts/build_dynamic_baseline_table.py
python scripts/audit_manuscript_readiness.py
```

The detailed pre-specified protocol, external Causal Chambers evaluation, and
reporting language are in `docs/MANUSCRIPT_EXPERIMENTS.md`. The native CD-NOD
baseline comparison additionally needs the
`dynamic-baselines` optional dependency group.

The experiments use a controlled synthetic regime-change dataset based on the **TimeGraph** benchmark.

---

## Experiment Setup

Current experiment:

| Setting                     |                         Value |
| --------------------------- | ----------------------------: |
| Time points                 |                          5000 |
| Variables                   |                             4 |
| Variables                   |                X1, X2, X3, X4 |
| Maximum lag                 |                             2 |
| Regime-change point         |                    `t = 2500` |
| Ground-truth representation | `(time, source, target, lag)` |

The time-indexed ground truth is stored conceptually as:

```text
A_true[t, source, target, lag]
```

with shape:

```text
(5000, 4, 4, 3)
```

---

## Regime-Change Ground Truth

### Stable causal links (lagged, recoverable by past-only estimator)

```text
X1(t-2) -> X4(t)   lag 2, weight 0.25
X3(t-1) -> X2(t)   lag 1, weight 0.30
```

### Contemporaneous links (NOT recoverable by lag-only estimator)

> **Note**: The a1 experiment contains two contemporaneous (lag-0) links in
> the data-generating process. The SCA estimator zeroes the lag-0 plane
> and cannot recover these links. This is a known limitation of all
> past-only estimators. See `docs/METHODOLOGY.md` §3 for a full discussion.

```text
X2(t) -> X1(t)   lag 0, weight 0.40  [IN BOTH REGIMES; NOT RECOVERABLE]
X4(t) -> X3(t)   lag 0, weight 0.35  [REGIME 1 ONLY; NOT RECOVERABLE]
```

### Changed at the change point

```text
Removed:  X4(t) -> X3(t)   lag 0  [contemporaneous; not recoverable]
Added:    X2(t-1) -> X3(t) lag 1  [LAGGED: recoverable]
```

The paper's primary evidence uses `generate_synthetic_regimes()` which
produces **only lagged edges** and is therefore a fair evaluation for
all compared methods.

The time-indexed ground truth is implemented in:

```text
regime_ground_truth_tensor.py
```

The controlled dataset is:

```text
Datasets/regime_change_a1.csv
```

---

## SCA Variants

Three SCA configurations are evaluated.

### Full SCA

The proposed method combines:

* global change detection,
* edge-specific persistence,
* selective causal plasticity, and
* graph memory.

### No Persistence

This ablation removes edge-specific persistence gating while retaining adaptive graph updating.

### Global Only

This ablation removes edge-specific plasticity selectivity and uses global change intensity to control adaptation.

These variants are used to test whether persistence and selective adaptation provide meaningful benefits.

---

## Baselines

### Rolling Regression

A time-varying rolling regression baseline is evaluated using:

```text
Window size: 500
Step size: 25
Maximum lag: 2
Threshold: 0.20
```

The threshold was selected using multiple validation times:

```text
3500
3700
3900
4100
4249
```

Prediction output:

```text
results/baseline/regime_change_A_pred_baseline.npy
```

Evaluation script:

```text
evaluate_baseline.py
```

### PCMCI+

PCMCI+ is evaluated as a **static causal discovery baseline**.

Configuration:

```text
Variables: 4
Maximum lag: 2
Alpha: 0.05
```

Prediction output:

```text
results/pcmci_a1_A_pred.npy
```

Evaluation script:

```text
evaluate_pcmci.py
```

PCMCI+ produces a single static graph, so dynamic adaptation metrics such as adaptation delay and stable-edge preservation are not assigned to this baseline.

---

## Legacy single-run results (not manuscript results)

This table is retained as a development smoke test only. It comes from one
synthetic realization, has no uncertainty estimates, and does not include a
regime-aware baseline; do not use it as submission evidence.

The final model comparison is stored in:

```text
results/model_comparison.csv
```

| Model              | Before F1 | Before SHD | After F1 | After SHD | Stable Preservation | Change Detection | Adaptation Delay | Unnecessary Change |
| ------------------ | --------: | ---------: | -------: | --------: | ------------------: | ---------------: | ---------------: | -----------------: |
| Full SCA           |    0.5714 |          6 |   0.7273 |         3 |              0.9591 |           0.8712 |               80 |             0.0409 |
| No Persistence     |    0.4615 |          7 |   0.6667 |         4 |              0.9347 |           0.7780 |               41 |             0.0653 |
| Global Only        |    0.4286 |          8 |   0.5333 |         7 |              0.9187 |           0.7580 |               27 |             0.0813 |
| Rolling Regression |    1.0000 |          0 |   0.7500 |         2 |              0.7603 |           0.8592 |              352 |             0.2397 |
| PCMCI+             |    0.8889 |          1 |      N/A |       N/A |                 N/A |              N/A |              N/A |                N/A |

### Interpretation

On this regime-change experiment, Full SCA provides substantially stronger **stable-edge preservation** and faster adaptation than the rolling regression baseline.

Full SCA achieves:

```text
Stable-edge preservation: 0.9591
Adaptation delay:         80
Unnecessary change rate:  0.0409
```

compared with:

```text
Stable-edge preservation: 0.7603
Adaptation delay:         352
Unnecessary change rate:  0.2397
```

The ablations also show reduced post-change causal performance when persistence or selective plasticity is removed.

The results do **not** establish that Full SCA has the highest raw F1 on this dataset. Rolling Regression achieves a slightly higher post-change F1 (`0.7500` versus `0.7273`). The stronger result for SCA is its selective adaptation behavior and preservation of stable causal structure.

---

## Selective Plasticity Results

The detailed ablation results are stored in:

```text
results/selective_plasticity_ablation.csv
```

| Model          | Changed Plasticity | Stable Plasticity | Selectivity Ratio |
| -------------- | -----------------: | ----------------: | ----------------: |
| Full SCA       |             0.0231 |            0.0204 |             1.13x |
| No Persistence |             0.0349 |            0.0307 |             1.14x |
| Global Only    |             0.1400 |            0.1400 |             1.00x |

The Global Only variant applies approximately equal plasticity to changed and stable edges, consistent with its lack of edge-specific selectivity.

---

## Repository Structure

```text
SCA-TimeGraph/
│
├── src/sca/                        # Core Python library
│   ├── data/                       # Dataset loading and window building
│   ├── ground_truth/               # Time-indexed ground truth tensor
│   ├── models/                     # Full SCA and ablation models
│   ├── baselines/                  # Rolling regression & PCMCI+ baselines
│   ├── evaluation/                 # Metrics & evaluation protocols
│   └── visualization/              # Plotting and figure generators
│
│
├── scripts/                        # Experiment execution & reproduction scripts
│   ├── generate_regime_change.py   # Dataset generator
│   ├── run_validation.py           # Multi-seed validation (paper protocol)
│   ├── run_null_experiment.py      # Null/no-change FAR benchmark
│   ├── select_synthetic_config.py  # Freeze hyperparameters from tuning
│   ├── build_manuscript_tables.py  # Paper Tables II-IV
│   ├── analyze_significance.py     # Paired sign-flip tests
│   ├── build_publication_figures.py
│   ├── run_sca.py                  # Full SCA and ablations runner
│   ├── run_baselines.py            # Baseline models runner
│   ├── evaluate_selective_plasticity.py
│   ├── evaluate_cdnod.py           # CD-NOD baseline
│   ├── evaluate_causal_chambers.py # External benchmark
│   ├── evaluate_baseline.py
│   ├── evaluate_pcmci.py
│   ├── results_table.py            # Comparison table generator
│   └── visualize_results.py        # Figure generator
│
├── notebooks/                      # TimeGraph benchmark exploration
│   └── benchmarks/                 # a1.ipynb ... d3c.ipynb
│
├── tests/                          # Tests, verification, and diagnostics
│   ├── inspect_a1.py
│   ├── inspect_windows.py
│   ├── test_candidate_edges.py
│   ├── test_model_interface.py
│   ├── verify_regime_change.py
│   └── verify_regime_regression.py
│
├── docs/                           # Internal specs & design notes
│   ├── EXPERIMENT_PLAN.md
│   ├── MODEL_INTERFACE.md
│   ├── PERSON1_HANDOFF.md
│   └── REGIME_CHANGE_SPEC.md
│
├── results/                        # Generated tables, arrays, and figures
│   ├── baseline/
│   ├── plots/
│   ├── model_comparison.csv
│   └── selective_plasticity_ablation.csv
│
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Package installation metadata
└── README.md
```

---

## Reproducing the Main Experiment

### Environment Setup

Create and activate your Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Optionally, install the `sca` package in editable mode:

```bash
pip install -e .
```

### Step 1: Generate Dataset & Ground Truth

Generate the controlled regime-change dataset:

```bash
python scripts/generate_regime_change.py
```

Validate the time-indexed ground truth tensor:

```bash
python -m sca.ground_truth.regime_ground_truth_tensor
```

### Step 2: Run SCA Models and Baselines

Run Full SCA, No Persistence, and Global Only models:

```bash
python scripts/run_sca.py --variant all
```

Or run individual models:

```bash
python src/sca/models/sca_model.py
python src/sca/models/sca_model_no_persistence.py
python src/sca/models/sca_model_global_only.py
```

Run baselines (Rolling Regression and PCMCI+):

```bash
python scripts/run_baselines.py --model all
```

### Step 3: Evaluate and Summarize Results

Evaluate the SCA variants and ablations:

```bash
python scripts/evaluate_selective_plasticity.py
```

Evaluate baseline models:

```bash
python scripts/evaluate_baseline.py
python scripts/evaluate_pcmci.py
```

Regenerate the final comparison table:

```bash
python scripts/results_table.py
```

Generate visualization figures:

```bash
python scripts/visualize_results.py
```


---

## Validation

The final experiment was checked for:

### Prediction interface

SCA prediction arrays use:

```text
(time, source, target, lag)
```

with expected shape:

```text
(5000, 4, 4, 3)
```

### Reproducibility

Full SCA, No Persistence, and Global Only were rerun and compared against their saved prediction arrays.

All three reproduced their saved outputs exactly:

```text
Max absolute difference = 0.0
Exactly equal = True
```

### Leakage

The prediction pipeline processes the series chronologically and constructs local windows using observations available up to the current time index.

No obvious future-data leakage was identified during validation.

### Baseline validation

The rolling regression and PCMCI+ evaluations were independently rerun and reproduced their recorded results.

---

## Main Output Files

### SCA predictions

```text
results/sca_full_A_pred.npy
results/sca_no_persistence_A_pred.npy
results/sca_global_only_A_pred.npy
```

### SCA plasticity

```text
results/sca_full_plasticity_rates.npy
results/sca_no_persistence_plasticity_rates.npy
results/sca_global_only_plasticity_rates.npy
```

### Change detection

```text
results/sca_change_scores.npy
results/sca_detected_flags.npy
```

### Baseline

```text
results/baseline/regime_change_A_pred_baseline.npy
```

### PCMCI+

```text
results/pcmci_a1_A_pred.npy
```

### Final tables

```text
results/model_comparison.csv
results/selective_plasticity_ablation.csv
```

### Visualizations

```text
results/plots/
```

---

## Reproducible peer-review validation

`scripts/run_validation.py` is the canonical empirical runner. Its standard
run uses 20 independent seeds and writes both per-seed data and grouped mean
and standard-deviation tables. It covers `d={4,10,20,50}`, `L={1,2,4}`;
Gaussian, Student-t, and uniform innovations; `sigma={0.05,0.1,0.25,0.5}`;
and recurring shifts at `t={1000,2500,4000}`.

```text
python scripts/run_validation.py --study all --seeds 20
```

Use `--quick` only for a smoke test, never for manuscript tables. The runner
writes `synthetic_raw.csv`, `synthetic_mean_std.csv`, `sensitivity_raw.csv`,
and `sensitivity_mean_std.csv` under `results/validation/`. The sensitivity
study varies detection window D, regression window W, threshold theta, decay
delta, and deviation scale epsilon one factor at a time.

`scripts/evaluate_cdnod.py` provides the native CD-NOD baseline on the
pre-specified canonical dynamic subset. It records failures rather than
relabelling rolling OLS as a regime-aware method. The selected external
benchmark is versioned and documented below.

The selected external benchmark is now Causal Chambers `wt_walks_v1`, with a
locked protocol in `docs/CAUSAL_CHAMBERS_WT_WALKS.md`. Download it with
`pip install -e ".[benchmark]"` followed by
`python scripts/download_causal_chambers.py`.

---

## Related Project Documentation

Additional project documentation is available in `docs/`:

```text
docs/EXPERIMENT_PLAN.md
docs/METHODOLOGY.md        -- formal method and evaluation description
docs/MODEL_INTERFACE.md
docs/REGIME_CHANGE_SPEC.md
docs/PERSON1_HANDOFF.md
docs/CAUSAL_CHAMBERS_WT_WALKS.md
```

---

## About TimeGraph

The underlying synthetic benchmark is **TimeGraph: Synthetic Benchmark Datasets for Robust Time-Series Causal Discovery**.

This repository uses TimeGraph as the basis for controlled causal-discovery experiments and extends the workflow with a regime-change experiment and SCA evaluation.

For the original TimeGraph benchmark and its datasets/generators, see the project materials and citation below.

---

## Citation

If you use TimeGraph, please cite:

```bibtex
@inproceedings{Ferdous2025TimeGraph,
  author    = {Muhammad Hasan Ferdous and Emam Hossain and Md Osman Gani},
  title     = {{TimeGraph}: Synthetic Benchmark Datasets for Robust Time-Series Causal Discovery},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '25)},
  series    = {KDD '25},
  year      = {2025},
  isbn      = {979-8-4007-1454-2/2025/08},
  publisher = {ACM},
  address   = {Toronto, ON, Canada},
  doi       = {10.1145/3711896.3737439},
  numpages  = {11},
  location  = {Toronto, ON, Canada},
  month     = {August #3--7},
}
```

---

## License

* The **code** in this repository is licensed under the [MIT License](./LICENSE).
* The **datasets** are released under the [CC BY 4.0 License](./LICENSE-CC-BY-4.0.txt).
