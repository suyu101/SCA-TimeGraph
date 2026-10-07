# Manuscript-ready experimental material

This file contains text and tables for the experimental section. Replace only
the explicitly marked placeholders after the 20-seed run completes. Do not
report the partially written `results/validation/` CSV while the process is
still running.

## Data and licence attribution

We additionally evaluate on the `wt_walks_v1` wind-tunnel dataset from Causal
Chambers. The dataset comprises controlled measurements from a real physical
system and provides an official reference causal graph for its standard
configuration. The selected `actuators_random_walk_2` experiment contains
10,000 chronological observations of 32 graph-aligned measured variables. The
median sampling interval is 0.3118 timestamp units. The data are available
under CC BY 4.0; we retain the source licence and cite Gamella, Peters and
Buhlmann (2025).

```bibtex
@article{gamella2025chamber,
  author={Gamella, Juan L. and Peters, Jonas and B{\"u}hlmann, Peter},
  title={Causal chambers as a real-world physical testbed for {AI} methodology},
  journal={Nature Machine Intelligence},
  year={2025},
  doi={10.1038/s42256-024-00964-x}
}
```

## Experimental protocol

We evaluate the proposed selective causal adaptation (SCA) method under a
factorial synthetic design. For each independent held-out random seed, we
generate a 5,000-observation process with three changes at
$t\in\{1000,2500,4000\}$. The graph alternates A$\rightarrow$B$\rightarrow$A
$\rightarrow$B, so the second change explicitly tests recovery of a previously
seen causal regime rather than an unrelated fourth graph.
We vary the number of variables $d\in\{4,10,20,50\}$, the maximum lag
$L\in\{1,2,4\}$, innovation scale $\sigma\in\{0.05,0.1,0.25,0.5\}$, and
measurement-innovation distribution (Gaussian, Student-$t$ with five degrees
of freedom, and uniform). A fixed Gaussian process innovation defines the
common signal scale; additive measurement innovations of scale $\sigma$ are
then applied to vary SNR without merely rescaling a homogeneous VAR process.
We report the arithmetic mean and sample standard deviation over held-out seeds
20--39. Hyperparameters are selected only on tuning seeds 0--19. Metrics use a
predefined absolute edge threshold of 0.15 and include precision, recall, F1,
structural Hamming distance (SHD), adaptation delay, stable-edge preservation,
change-detection rate, and false-alarm rate.

The native regime-aware CD-NOD baseline is evaluated on the canonical dynamic
subset ($d=4$, $L=2$, Gaussian $\sigma=0.10$) using the same held-out seeds
20--39. We use CD-NOD's Fisher-Z conditional-independence test at
$\alpha=0.05$, a lag-expanded observation matrix, and the chronological sample
index as its documented domain-index input. Known temporal precedence directs
adjacencies from lagged to present nodes when scoring. This subset is reported
separately because a full factorial dynamic-baseline sweep is computationally
intractable; the runner records every failure rather than substituting another
model.

For the external evaluation, we use the Causal Chambers `wt_walks_v1`
`actuators_random_walk_2` series. We preserve chronological order, exclude
administrative fields (`timestamp`, `config`, `counter`, `flag`, and
`intervention`) from causal discovery, and retain the 32 variables present in
the official standard-configuration graph. SCA parameters are selected on a
chronological validation prefix and evaluated at the held-out endpoint. The
graph-recovery comparison collapses inferred positive-lag edges across lags;
therefore it evaluates directed edge recovery but does not claim identification
of instantaneous causal directions.

## Statistical analysis

For the factorial SCA--Rolling-OLS comparison, first average each method over
all factorial conditions within each held-out seed; the resulting 20 seed
summaries are the independent paired observations. CD-NOD is tested separately
on its pre-specified canonical dynamic cell. We use two-sided paired sign-flip
randomization tests (100,000 draws), report bootstrap 95% confidence intervals
for paired mean differences and paired Cohen's $d_z$, and apply Holm correction
over the reported tests. F1 is the primary endpoint; all other metrics are
secondary.

## Table II: synthetic accuracy and adaptation

The finalized machine-readable Table II is
`results/manuscript_table_ii.csv`. It contains all 288 method-by-condition
rows ($2\times4\times3\times4\times3$), with every reported value expressed
as the mean $\pm$ sample standard deviation across the 20 held-out seeds.
This long format is intentional: it prevents the dimensionality, lag, noise,
and distribution results from being hidden by pooled averages. Rolling OLS has
no change-point detector, so its detection-rate and false-alarm entries are
reported as **N/A**, not zero.

## Hyperparameter sensitivity analysis (Table III)

| Parameter | Values | Selection rule | Reported outcomes |
|---|---|---|---|
| Detection window $D$ | 50, 100, 200 | One-factor-at-a-time | F1, SHD, detection rate (mean $\pm$ SD) |
| Regression window $W$ | 50, 100, 200 | One-factor-at-a-time | F1, SHD, detection rate (mean $\pm$ SD) |
| Threshold $\theta$ | 0.20, 0.25, 0.30 | One-factor-at-a-time | F1, SHD, detection rate, false-alarm rate (mean $\pm$ SD) |
| Decay $\delta$ | 0.70, 0.90, 0.97 | One-factor-at-a-time | F1, SHD, detection rate (mean $\pm$ SD) |
| Deviation scale $\varepsilon$ | 0.025, 0.05, 0.10 | One-factor-at-a-time | F1, SHD, detection rate (mean $\pm$ SD) |

The finalized row-level Table III is `results/manuscript_table_iii.csv`.
Across tuning seeds 0--19, F1 improves strongly with the regression window
($0.408\pm0.031$ at $W=50$ versus $0.609\pm0.052$ at $W=200$). The selected
threshold $\theta=0.30$ has the lowest false-alarm rate
($0.017\pm0.003$) among the tested thresholds and also the highest F1. The
tested detection windows show a modest F1 decrease as $D$ grows, while larger
$\varepsilon$ improves F1 and SHD in this design. These are one-factor-at-a-
time diagnostic results, not an estimate of hyperparameter interactions; the
configuration selection file records that selection separately from the
held-out factorial evaluation.

## Table IV: seed-matched regime-aware baseline

Populate this table with `scripts/build_dynamic_baseline_table.py` after all
20 held-out CD-NOD seeds have completed. The comparison is restricted to the
canonical dynamic setting ($d=4$, $L=2$, Gaussian $\sigma=0.10$), with SCA,
Rolling OLS, and native CD-NOD all evaluated on exactly seeds 20--39.

The finalized Table IV (`results/manuscript_table_iv_dynamic_baseline.csv`)
shows that CD-NOD attains higher recall ($1.000\pm0.000$) and F1
($0.775\pm0.069$) than SCA ($0.551\pm0.081$ recall and
$0.637\pm0.065$ F1) in this canonical cell. SCA has higher precision
($0.869\pm0.042$ versus $0.638\pm0.093$) and a shorter recovery measure
($456.333\pm51.480$ versus $500.000\pm0.000$); its fixed CD-NOD graph does
not adapt between regime boundaries, so the latter is a post-shift recovery
measure rather than a change-detection claim. The paired F1 difference favors
CD-NOD (Holm-adjusted $p=0.00028$), while SHD is not significantly different
(Holm-adjusted $p=0.71227$). Report these trade-offs directly; do not claim
universal superiority for SCA.

## External benchmark result

The validation-selected SCA configuration ($W=500$, $D=500$,
$\theta=0.05$, $\delta=0.90$, and $\varepsilon=0.05$) achieved precision
0.167, recall 0.548, F1 0.256, and SHD 134 on the held-out endpoint. On the
same wind-tunnel benchmark, PCMCI+ achieved precision 0.029, recall 0.024,
F1 0.026, and SHD 74. SCA consequently had higher recall and F1, whereas
PCMCI+ had lower SHD. This is a single external series, so it should be
reported as complementary evidence rather than treated as a statistical test.

## Reproducibility statement

All experiments use fixed, recorded random seeds and chronology-preserving
processing. The full commands are:

```text
python scripts/capture_environment.py
python scripts/run_validation.py --study sensitivity --seeds 20
python scripts/select_synthetic_config.py
python scripts/run_validation.py --study synthetic --seeds 20
python scripts/build_manuscript_tables.py
python scripts/analyze_significance.py
python scripts/build_publication_figures.py
python scripts/evaluate_cdnod.py
python scripts/build_dynamic_baseline_table.py
python scripts/evaluate_causal_chambers.py
python scripts/tune_causal_chambers.py
python scripts/audit_manuscript_readiness.py
```

The repository stores per-seed synthetic outcomes, grouped mean and standard
deviation tables, validation-search results, external-benchmark predictions,
and a machine-readable environment manifest. Report the operating system,
CPU, memory, Python implementation/version, and exact package versions from
`results/reproducibility_manifest.json` for the environment that ran the final
experiments.
