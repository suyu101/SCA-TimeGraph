# Methodology: Selective Causal Adaptation (SCA)

This document provides a self-contained description of the SCA method,
its evaluation protocol, and its known limitations. It is intended for
IEEE Access reviewers and for anyone trying to reproduce or extend this work.

---

## 1. Problem Statement

We study **online causal graph estimation under regime changes**. A causal
regime change is a discrete, unknown shift in the generating mechanism of
a time series such that the causal graph — the set of directed edges
representing predictive Granger-causal relationships — changes at an unknown
time point.

The core research question is:

> Can a causal discovery system adapt to regime changes while
> (a) preserving stable relationships that have not changed,
> (b) adapting rapidly when relationships genuinely change, and
> (c) avoiding unnecessary graph changes?

This tradeoff is the **stability-plasticity tradeoff**.

---

## 2. Graph Representation

The causal graph at time $t$ is represented as a weighted adjacency tensor:

$$A[t, \text{source}, \text{target}, \text{lag}] \in \mathbb{R}$$

where:
- `source`, `target` in `{0, ..., d-1}` (variable indices)
- `lag` in `{0, 1, ..., L}` (lag order, where L is the maximum lag)
- `A[t, i, j, l]` denotes the estimated causal weight from $X_i(t-l)$ to $X_j(t)$

Self-edges are always zero: $A[t, i, i, l] = 0$ for all $t, i, l$.

### 2.1 Edge-Presence Decision Rule

An edge is considered **present** at time $t$ if:

$$|A[t, \text{source}, \text{target}, \text{lag}]| \geq \theta$$

where $\theta$ is the preregistered edge threshold **0.15** for all paper
results. This threshold was selected on tuning seeds 0-19 and is fixed for
all held-out evaluation seeds 20-39.

---

## 3. Identifiability Assumptions and Limitations

### 3.1 Lag-Only Causal Discovery

The SCA estimator uses **past-only OLS regression**: to estimate the graph at
time $t$, each target variable $X_j(t)$ is regressed on all lagged
predictors $X_i(t-l)$ for $l = 1, ..., L$. The **lag-0 plane is always set
to zero**:

$$A[\cdot, \cdot, \cdot, 0] = 0$$

**Rationale**: Conditioning on $X_i(t)$ to predict $X_j(t)$ would constitute
target-time data usage. In an online setting where the graph at time $t$
is estimated using data available up to and including time $t$, using
any observation at time $t$ (other than the target itself, which we are
predicting) risks introducing contemporaneous correlation artifacts. More
importantly, contemporaneous causal direction between $X_i(t)$ and $X_j(t)$
is **not identifiable** from observational data without additional structural
assumptions (e.g., acyclicity plus non-Gaussianity, or a known causal ordering).

**Consequence**: The SCA method discovers **Granger-causal** (predictive)
relationships at positive lags. It does not claim to identify
contemporaneous structural causal edges. This is a limitation that applies
equally to all baselines (Rolling OLS, PCMCI+, CD-NOD in lag-expansion mode).

### 3.2 The 4-Variable A1 Experiment (Legacy)

The 4-variable `regime_change_a1.csv` experiment was designed with two
contemporaneous causal links in the data-generating process:

| Link | Regime | Weight |
|------|--------|--------|
| X2(t) → X1(t) | both | 0.40 |
| X4(t) → X3(t) | Regime 1 only | 0.35 |

These links appear as lag=0 entries in the ground-truth tensor.
Since the SCA estimator zeroes the lag=0 plane, these edges are
**structurally unrecoverable** by the method. Any quantitative evaluation
of a past-only estimator against this ground truth will systematically
undercount true positives for these specific edges.

**Reporting convention**: The 4-variable a1 experiment is retained as a
development/smoke-test benchmark. Its quantitative results (F1, SHD) reflect
both the estimator's genuine capabilities and this structural limitation.
For the manuscript's primary evidence, we use `generate_synthetic_regimes()`
which produces **only lagged (lag ≥ 1) edges**, making the evaluation fair
for all compared methods.

### 3.3 Granger vs. Structural Causality

All metrics compare estimated **Granger-causal graphs** to a **Granger-causal
ground truth**. The ground truth is constructed as the set of positive-lag
coefficients in the data-generating VAR process. We do not claim that the
discovered edges are structural/interventional causes in the sense of
Pearl's do-calculus.

---

## 4. Online Graph Estimation

At each time $t$, the SCA estimator:

1. **Selects a window** $[t - W + 1, t]$ of $W$ recent observations
2. **Constructs lag features** $[X(t-1), ..., X(t-L)]$ stacked horizontally
3. **Fits OLS** per target variable, regressing the target on all lag features
4. **Extracts coefficients** as edge weights
5. **Zeroes the lag-0 plane** to prevent contemporaneous leakage

The resulting **instantaneous graph estimate** $\hat{A}_t$ is then passed
to the memory update step.

### 4.1 Window Choice

Regression window $W$ is a tunable hyperparameter. Larger $W$ gives more
stable estimates but slower adaptation. The paper-facing protocol tunes $W$
on seeds 0-19 and evaluates on seeds 20-39.

---

## 5. Selective Adaptation

The SCA model maintains a **graph memory** $M_t$ and updates it using
edge-specific plasticity rates:

$$M_t[i,j,l] = (1 - \alpha_{t}[i,j,l]) \cdot M_{t-1}[i,j,l] + \alpha_{t}[i,j,l] \cdot \hat{A}_t[i,j,l]$$

where the plasticity rate $\alpha_t[i,j,l]$ is high for edges that appear
to have changed and low for edges that appear stable.

### 5.1 Global Change Detection

Change intensity $c_t$ is a scalar signal computed as the mean absolute
difference between the current window estimate $\hat{A}_t$ and a reference
window estimate $\hat{A}_{t-D}$ computed $D$ steps earlier:

$$c_t = \frac{1}{|\mathcal{E}|} \sum_{(i,j,l) \in \mathcal{E}} |\hat{A}_t[i,j,l] - \hat{A}_{t-D}[i,j,l]|$$

where $\mathcal{E}$ is the set of all non-self, non-lag-0 edge positions.

### 5.2 Edge-Specific Persistence

For each edge $(i, j, l)$, a **persistence score** $p_{t}[i,j,l]$ tracks
how consistently the edge has been detected across recent windows:

$$p_{t}[i,j,l] = \delta \cdot p_{t-1}[i,j,l] + (1-\delta) \cdot \mathbf{1}[\hat{A}_t[i,j,l] \geq \theta]$$

where $\delta$ is the persistence decay rate. High persistence → more stable
edge → lower plasticity.

### 5.3 Combined Plasticity Rate

$$\alpha_t[i,j,l] = \text{clip}\left(\alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot c_t \cdot (1 - p_t[i,j,l]), \alpha_{\min}, \alpha_{\max}\right)$$

The Global-Only ablation uses $c_t$ alone (without edge-specific persistence).
The No-Persistence ablation uses uniform $\alpha_t[i,j,l] = c_t$ for all edges.

---

## 6. Evaluation Metrics

### 6.1 Structural Metrics (per time window)

| Metric | Formula | Description |
|--------|---------|-------------|
| Precision | TP / (TP + FP) | Fraction of predicted edges that are correct |
| Recall | TP / (TP + FN) | Fraction of true edges that are predicted |
| F1 | 2·P·R / (P + R) | Harmonic mean of precision and recall |
| SHD | FP + FN | Structural Hamming distance |

All metrics use the preregistered threshold $\theta = 0.15$ and exclude
lag-0 edges from both prediction and ground truth.

### 6.2 Adaptation Metrics

**Stable-Edge Preservation (SEP)**: Fraction of time steps (in the
post-change region) at which all lagged stable edges remain correctly
predicted. Stable edges are those with the same weight in both regimes.

**Adaptation Delay**: The first time step $t \geq t_c$ at which the
estimated graph matches the post-change ground truth on all changed
edges:

$$\text{delay} = \min\{t \geq t_c : \text{all changed edges correctly recovered}\}$$

**Generalized Adaptation Delay**: Extension for arbitrary edge change sets.
See `generalized_adaptation_delay()` in `adaptation_metrics.py`.

**Change-Detection Rate**: Fraction of post-change time steps at which
a change is detected.

**False-Alarm Rate (FAR)**: Fraction of pre-change time steps (or null
experiment time steps) at which a change is spuriously detected.

### 6.3 Null Experiment

The null experiment evaluates SCA on **stationary data** (no regime change).
A good method should have:
- FAR < 5% (strict criterion, preregistered threshold)
- FAR < 20% (lenient criterion)

Run: `python scripts/run_null_experiment.py`

---

## 7. Experimental Protocol

### 7.1 Train/Validation/Test Split

| Phase | Seeds | Purpose |
|-------|-------|---------|
| Sensitivity tuning | 0–19 | Select hyperparameters |
| Held-out evaluation | 20–39 | Final reported metrics |
| Null experiment | 100+ | False-alarm rate (separate seeds) |

**No test-set leakage**: Hyperparameters are selected on seeds 0-19.
Seeds 20-39 are used only once, for the final held-out evaluation.

### 7.2 Factorial Design

The multi-seed validation sweeps:
- Dimensions: d ∈ {4, 10, 20, 50}
- Maximum lag: L ∈ {1, 2, 4}
- Noise scale: σ ∈ {0.05, 0.1, 0.25, 0.5}
- Distribution: Gaussian, Student-t (df=5), Uniform

This gives 4 × 3 × 4 × 3 = 144 factorial conditions, each evaluated on 20
held-out seeds → 2,880 total (model, condition, seed) triples.

### 7.3 Recurring Regime Test

The benchmark uses three change points at t ∈ {1000, 2500, 4000} in a 5000-step
series. The sequence alternates A → B → A → B, so the second change (t=2500)
**reverts to a previously seen graph**. This explicitly tests whether SCA
can leverage memory to recover known regimes faster than a fresh estimator.

---

## 8. Baselines

### 8.1 Rolling OLS

A rolling-window OLS baseline that re-estimates the graph every step
with no change detection or memory. It serves as the primary comparison
for assessing whether SCA's stability-plasticity mechanism adds value.

### 8.2 CD-NOD (Canonical Dynamic Cell)

The native regime-aware CD-NOD baseline is evaluated on the pre-specified
canonical condition (d=4, L=2, Gaussian, σ=0.10). CD-NOD uses Fisher-Z
conditional independence tests and a chronological domain index. It is not
evaluated on the full factorial because a full dynamic-baseline sweep is
computationally intractable; this limitation is documented in the results.

### 8.3 PCMCI+

Evaluated as a static baseline on the external Causal Chambers benchmark.
No dynamic adaptation metrics are assigned to PCMCI+ since it estimates
a single static graph.

---

## 9. External Benchmark

The external benchmark uses the **Causal Chambers `wt_walks_v1`** dataset
(`actuators_random_walk_2` series). Processing: chronological order preserved;
administrative fields excluded; 32 graph-aligned variables retained.
SCA parameters selected on a chronological validation prefix; evaluated
on the held-out endpoint.

Citation: Gamella, Peters, Buhlmann (2025), Nature Machine Intelligence.
DOI: 10.1038/s42256-024-00964-x

---

## 10. Statistical Testing

- Two-sided paired sign-flip randomization tests (100,000 draws)
- Bootstrap 95% confidence intervals for paired mean differences
- Paired Cohen's $d_z$ as effect size
- Holm correction across all reported tests
- Primary endpoint: F1; all other metrics secondary
- SCA vs. Rolling OLS: paired across 20 held-out seeds × all factorial conditions
- SCA vs. CD-NOD: paired on the pre-specified canonical dynamic cell only

---

## 11. Reproducibility

All experiments use fixed, recorded random seeds and chronology-preserving
processing. The full command sequence is documented in `docs/MANUSCRIPT_EXPERIMENTS.md`.

Required files for reproduction:
- `scripts/run_validation.py` — multi-seed synthetic evaluation
- `scripts/run_null_experiment.py` — null (no-change) FAR check
- `scripts/select_synthetic_config.py` — freeze hyperparameters
- `scripts/build_manuscript_tables.py` — generate Tables II–IV
- `scripts/analyze_significance.py` — statistical tests
- `scripts/build_publication_figures.py` — paper figures
- `scripts/evaluate_cdnod.py` — CD-NOD baseline
- `scripts/evaluate_causal_chambers.py` — external benchmark

All scripts record the operating system, CPU, Python version, and exact
package versions in `results/reproducibility_manifest.json`.
