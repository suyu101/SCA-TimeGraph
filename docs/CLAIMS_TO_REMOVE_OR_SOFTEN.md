# Claims to remove or soften before manuscript revision

## Audit scope

This is an audit only. It does not modify the LaTeX manuscript, figures,
experiment scripts, or scientific-result files.

The manuscript locations below refer to `SCA_ACCESS_latex/access.tex` in
`C:\Users\Suyesha\Downloads\SCA_ACCESS_latex.zip`. The completed Phase 1
artifacts inspected were:

- `results/phase1/tuning/selected_by_dimension.json`
- `results/phase1/heldout_d50_l4/heldout_summary.csv`
- `results/phase1/heldout_d50_l4/paired_stats.csv`
- `results/phase1/heldout_d50_l4/null_heldout.csv`

The completed baseline *tuning* file is not a held-out comparison. No result
from KGLS, RLS, global-rate EMA, a locality sweep, or a full held-out
factorial comparison may be reported until that distinct held-out evaluation
and its seed-level analysis have completed.

## Required changes

### A1. Placeholder publication metadata

- **Location:** Preamble, lines 55--56; rendered page 1 header.
- **Current wording:** `Date of publication xxxx 00, 0000` and
  `\doi{10.1109/ACCESS.2024.0429000}`.
- **Why:** These are IEEE template placeholders, not assigned publication
  metadata. Keeping them is misleading.
- **Action:** **Remove.** Leave these fields to the publisher, or use the
  journal's current submission-template guidance.
- **Safe replacement:** No replacement text in the submitted manuscript.

### A2. Abstract: unsupported broad factorial claim

- **Location:** Abstract, line 67, sentence beginning “Under the canonical
  dynamic condition and across the factorial design”.
- **Current wording:** “SCA improves precision and F1 against the Rolling OLS
  (pooled F1 difference $+0.0498$, Holm-corrected $p<0.001$) ...”.
- **Why:** The cited full-factorial result uses the legacy fixed configuration
  (`D=50`, `W=200`, `theta=0.30`) that was subsequently found to saturate the
  detector and to violate the repository's own false-alarm selection rule.
  The new jointly tuned, null-calibrated configurations have not yet been
  evaluated on the full held-out factorial design.
- **Action:** **Replace.**
- **Safe replacement:** “On a held-out high-dimensional stress-test cell
  ($d=50$, $L=4$, Gaussian measurement noise $\sigma=0.10$), a
  dimension-aware, ridge-regularised configuration substantially outperformed
  the documented legacy OLS configuration. This targeted result does not
  establish superiority across the full factorial design.”

### A3. Abstract: obsolete detector-failure conclusion

- **Location:** Abstract, line 67, sentence beginning “Sensitivity analysis
  and a high-dimensional...”.
- **Current wording:** “... reveal a detector failure mode in which change
  detection collapses to 0\% despite unchanged false-alarm behaviour
  elsewhere.”
- **Why:** The completed d=50, L=4 repair changes this finding. The selected
  configuration has `W=1000`, `D=500`, ridge alpha `1.0`, and threshold
  `0.3314109445`; on held-out seeds 20--39 it has change-detection rate
  `0.5833 +/- 0.2836` and eligible-update null false-alarm rate
  `0.0441 +/- 0.0138`.
- **Action:** **Replace.**
- **Safe replacement:** “A pre-specified high-dimensional stress test exposed
  an ill-posed small-window OLS setting; dimension-aware windows, ridge
  regularisation, and null calibration repaired the detector in that targeted
  cell.”

### A4. Introduction has no citations in its opening paragraph

- **Location:** Introduction, line 81, first paragraph.
- **Current wording:** Domain examples and claims about changing structures
  have no citations.
- **Why:** This directly misses the faculty request that citations begin in
  the Introduction. The empirical examples are externally verifiable claims.
- **Action:** **Replace / add citations.**
- **Safe replacement:** Cite verified sources for time-series causal discovery
  and nonstationary systems in the first paragraph; avoid naming seizure,
  wind-tunnel, or macroeconomic examples unless each is supported by a source.

### A5. Introduction overstates a comparative premise

- **Location:** Introduction, line 83, final sentence.
- **Current wording:** “... it is a poor match for the common case of
  localised non-stationarity ...”.
- **Why:** “Common case” and “poor match” are broad empirical claims not
  established by this paper's one-edge synthetic generator or its single
  external series.
- **Action:** **Soften.**
- **Safe replacement:** “It may be less sample-efficient when only a subset
  of relationships changes, a setting considered in this study.”

### A6. Research questions promise unfinished baselines

- **Location:** Introduction, line 89, RQ2 and RQ3.
- **Current wording:** RQ2 asks about a global update rate; RQ3 promises a
  broad factorial comparison.
- **Why:** The global-only ablation is a small development experiment, not the
  requested matched global-rate EMA baseline. The newly implemented
  KGLS/RLS/global-EMA comparisons have no held-out factorial results.
- **Action:** **Soften.**
- **Safe replacement:** “On the available development ablation, how do the
  persistence and edgewise-gating components affect the observed behaviour?”
  For RQ3: “How does SCA compare with the reported Rolling OLS and CD-NOD
  reference evaluations, and where does the legacy configuration fail?”

### A7. Introduction summary makes unsupported performance claims

- **Location:** Introduction, line 91.
- **Current wording:** “SCA improves precision, pooled F1, and ... stable-edge
  preservation and adaptation delay relative to Rolling OLS ...”.
- **Why:** The pooled legacy-factorial conclusion is superseded by the
  configuration audit. “Stable-edge preservation” is actually recall on the
  stable true edges. Whole-graph adaptation delay is almost always censored in
  high dimensions.
- **Action:** **Replace.**
- **Safe replacement:** “The legacy evaluation showed trade-offs against its
  Rolling OLS reference, whereas the completed high-dimensional stress test
  documents a repair of a specific ill-posed configuration. Stable-edge recall
  and per-edge recovery are reported separately from censored whole-graph
  recovery.”

### A8. Contributions overstate the completed evaluation

- **Location:** Introduction, line 93, contribution (3) and (4).
- **Current wording:** “a controlled factorial evaluation ... together with a
  paired statistical-testing protocol” and “including a high-dimensional
  detector-failure regime.”
- **Why:** The legacy factorial evaluation cannot support the same claim after
  the fixed configuration is shown to be uncalibrated. The completed repair
  reverses the categorical “detector-failure regime” statement.
- **Action:** **Replace.**
- **Safe replacement:** “a documented legacy factorial evaluation and a
  separately held-out, jointly tuned d=50, L=4 stress-test repair; broad
  held-out comparison with matched streaming baselines remains future work.”

### A9. Remove internal-development narrative

- **Location:** Related Work, line 100, sentence beginning “an early internal
  iteration of the present project...”.
- **Current wording:** Describes an abandoned deep-attention and flow model.
- **Why:** This is project-process metadata, not related work or evidence.
- **Action:** **Remove.**
- **Safe replacement:** End the preceding temporal-graph-attention sentence
  after its contrast with causal discovery.

### A10. Claims of causal discovery need scope control

- **Location:** Title, keywords, Introduction lines 83--85, Related Work, and
  Conclusion line 502.
- **Current wording:** Repeated use of “causal discovery”, “causal graph”, and
  “causal relationships” for SCA outputs.
- **Why:** The implementation estimates only positive-lag, observational,
  regression-based relationships; its lag-zero plane is fixed to zero and it
  provides no structural-causal identifiability guarantee.
- **Action:** **Rename / soften.**
- **Safe replacement:** Use “past-only Granger-style predictive graph
  estimation” for SCA's output. Retain “causal discovery” only when referring
  to the broader literature, ground-truth synthetic mechanism, or explicitly
  qualified benchmark task.

### A11. Incorrect Rolling OLS description

- **Location:** Experimental Protocol, line 229.
- **Current wording:** “Rolling OLS re-estimates the graph at every step using
  the same trailing-window OLS procedure as SCA's local estimator.”
- **Why:** The completed d=50, L=4 evaluation labels the comparator “Rolling
  OLS (W=250, U=10)”; it does not update every step and it does not use SCA's
  selected local window (`W=1000`). README and older text contain other
  settings, so each reported comparison must name its actual configuration.
- **Action:** **Replace.**
- **Safe replacement:** “For the d=50, L=4 held-out stress test, Rolling OLS
  uses a 250-observation window and updates every 10 observations. It has no
  detector or graph-memory state.”

### A12. Remove meta-sentence about fabrication

- **Location:** Experimental Protocol, line 229, final clause.
- **Current wording:** “... no fabricated results are provided for methods
  that were not actually run.”
- **Why:** This is meta-commentary rather than scientific reporting.
- **Action:** **Remove.**
- **Safe replacement:** “No quantitative comparison with CASTOR or FANTOM is
  reported.”

### A13. Metric definitions are inaccurate

- **Location:** Evaluation Protocol, line 233.
- **Current wording:** “Adaptation delay is when we first see all the changes
  correctly identified ... Stable-edge preservation checks how often ... every
  edge ... is still predicted correctly.”
- **Why:** Whole-graph adaptation delay requires the entire predicted graph to
  equal the truth and is heavily censored. The implemented stable metric is
  the mean detection rate of stable true edges, not correctness of every
  stable edge or protection from false-positive changes.
- **Action:** **Rename and replace.**
- **Safe replacement:** “Whole-graph recovery delay is a censored secondary
  diagnostic. Per-edge recovery delay is reported for changed edges. Stable-
  edge recall is the fraction of stable true edges that remain above the fixed
  decision threshold over the post-change horizon.”

### A14. Hyperparameter-selection description is obsolete

- **Location:** Evaluation Protocol, line 241; Table III and its caption.
- **Current wording:** A single `D=50`, `W=200`, `theta=0.30`, `lambda=0.97`,
  `epsilon=0.10` configuration was fixed before held-out evaluation;
  one-factor sensitivity is presented as selection evidence.
- **Why:** The new selection is joint, dimension/lag-specific, and calibrated
  to a null target false-alarm rate of 0.05 using tuning seeds 0--19. The
  legacy configuration failed that selection rule.
- **Action:** **Replace.**
- **Safe replacement:** “For each (d, L), a fractional-factorial joint screen
  on tuning seeds 0--19 selected a dimension-aware window, ridge alpha,
  persistence decay, and deviation scale. The detector threshold was then
  calibrated from stationary null scores to target an eligible-update false-
  alarm rate of 0.05. One-factor sensitivity results are retained only as
  diagnostics, not as the selection procedure.”

### A15. Canonical and full-factorial legacy result tables require quarantine

- **Location:** Results, lines 260--342; Tables labelled canonical, factorial,
  pooled, and their associated figure.
- **Current wording:** Presents legacy pooled statistics as the main empirical
  result.
- **Why:** Those tables are tied to the uncalibrated fixed SCA configuration.
  They cannot be combined with the newly tuned d=50, L=4 result as though all
  cells shared the same protocol.
- **Action:** **Replace / clearly label historical.**
- **Safe replacement:** Either remove these tables from the main paper, or
  label them “legacy fixed-configuration diagnostic; not the primary
  evaluation” and place them in an appendix. Do not cite their pooled p-values
  in the abstract, discussion, or conclusion.

### A16. Development ablation is not general evidence

- **Location:** Ablation section, line 413.
- **Current wording:** “... confirming that both mechanisms contribute to the
  observed gains ...” plus stable-edge preservation and adaptation-delay
  comparisons.
- **Why:** This is one small development instance with a legacy metric and no
  20-seed canonical ablation result. It cannot answer the full RQ2.
- **Action:** **Soften and rename.**
- **Safe replacement:** “On this development instance, the ablations exhibit
  the expected qualitative ordering. This observation is not a multi-seed
  estimate of a general edge-selectivity advantage.”

### A17. External benchmark configuration inconsistency

- **Location:** External Benchmark table and text, lines approximately
  445--462.
- **Current wording:** SCA F1 `0.251`, SHD `143`, update setting `U=50` in the
  manuscript.
- **Why:** Repository validation documentation records a distinct
  validation-selected result of F1 `0.256`, SHD `134`, and `U=100`. The paper
  does not establish which configuration, edge budget, or endpoint is final;
  PCMCI+ also has a much smaller inferred edge count.
- **Action:** **Replace or remove.**
- **Safe replacement:** “A single chronological external series is reported
  descriptively. Methods use different operating points; therefore results are
  not treated as a definitive head-to-head comparison.” Report one verified
  configuration only after reconciling the saved output with the manuscript.

### A18. Failure-mode section must become a repair section

- **Location:** Failure Modes, lines 469--476; Discussion line 487;
  Limitations lines 494--497; Conclusion line 502.
- **Current wording:** The d=50, L=4 detector “fails outright”, fixed top-k is
  stated as its root cause, and failure is treated as the final conclusion.
- **Why:** The new result identifies ill-posed local OLS (`d x L = 200`
  predictors with roughly 196 local samples) and short detection windows as
  the actionable issue. The held-out repair is available for the Gaussian,
  sigma=0.10 d=50, L=4 cell only.
- **Action:** **Replace.**
- **Safe replacement:** “The legacy small-window OLS configuration failed on
  d=50, L=4. A jointly tuned dimension-aware ridge configuration (`W=1000`,
  `D=500`, ridge alpha `1.0`) achieved F1 `0.576 +/- 0.036`, AUPRC
  `0.831 +/- 0.014`, and eligible-update null false-alarm rate
  `0.044 +/- 0.014` on held-out seeds 20--39. This targeted repair does not
  validate the method over the full factorial design.”

### A19. False-alarm comparison uses a mismatched null configuration

- **Location:** Failure Modes, line 471.
- **Current wording:** Contrasts approximately 9% factorial false alarms with
  a 1.4% null experiment.
- **Why:** The null experiment used a different detection window (`D=100`)
  than the legacy selected configuration (`D=50`), so it does not validate the
  reported operating point.
- **Action:** **Remove and replace.**
- **Safe replacement:** Report only the matched frozen d=50, L=4 null
  control: eligible-update false-alarm rate `0.0441 +/- 0.0138` over held-out
  seeds 20--39. If reporting all-time rate, label it separately as
  `0.00353 +/- 0.00110` because update opportunities occur every 10 samples.

### A20. Discussion uses subjective and stale language

- **Location:** Discussion, lines 483--487.
- **Current wording:** “somewhat disappointingly”, legacy pooled F1 claims,
  and categorical high-dimensional failure statements.
- **Why:** The phrase is editorial rather than analytical, and the numerical
  claims are tied to the superseded configuration.
- **Action:** **Replace.**
- **Safe replacement:** “The legacy fixed configuration showed a trade-off in
  stable-edge recall. The targeted d=50, L=4 repair demonstrates that local
  regression conditioning and null calibration materially affect performance;
  broader comparative claims require the pending matched held-out study.”

### A21. Conclusion overclaims both breadth and ablation evidence

- **Location:** Conclusion, line 502.
- **Current wording:** “SCA consistently improves precision and pooled F1 ...”
  and “Two ablation experiments confirm ...”.
- **Why:** “Consistently” is contradicted by the legacy high-dimensional
  failure and is not established under the repaired protocol. The ablation is
  development-scale only.
- **Action:** **Replace.**
- **Safe replacement:** “SCA is a training-free, past-only graph-memory
  estimator. A held-out d=50, L=4 stress test shows that dimension-aware
  ridge estimation and null calibration can repair a documented legacy
  failure. The present evidence does not establish universal superiority over
  rolling or regime-aware alternatives.”

### A22. Bibliography and citation audit

- **Location:** Introduction lines 81--93; bibliography lines 510--538.
- **Current wording:** Fifteen bibliography entries, while the Introduction's
  first paragraph has no citations.
- **Why:** This misses both faculty requirements: citations must start in the
  Introduction, and the reference base needs expansion. Some detailed claims
  in Related Work should be rechecked against the cited papers.
- **Action:** **Add verified citations; do not guess.**
- **Safe replacement:** Expand to 35--45 entries only after independently
  verifying each title, author list, venue, year, and DOI. Prioritise
  Granger/VAR, change detection and drift, RLS/forgetting, stability--
  plasticity/continual learning, time-varying VAR, and online FCM work,
  including Liu et al. (IEEE Transactions on Fuzzy Systems, 2026,
  DOI 10.1109/TFUZZ.2026.3667554).

## Claims that must not be added

Do not add any of the following until new results exist and are analysed at
the seed level:

- SCA beats KGLS-style ridge-to-previous, RLS-forgetting, or global-rate EMA.
- SCA has a locality advantage for one-edge, 10%, 50%, or 100% graph changes.
- A broad full-factorial result for the jointly tuned dimension-aware SCA.
- A fair RPCMCI, CD-NOD, PCMCI+, CASTOR, or FANTOM comparison.
- General real-world effectiveness based on the single Causal Chambers series.
