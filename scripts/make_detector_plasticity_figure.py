"""Fig. 8: detector and plasticity traces for the development benchmark, from the arrays written by
scripts/regenerate_dev_ablation.py (final, lag-zero-free estimator; no experiment is rerun here).

Usage: python scripts/make_detector_plasticity_figure.py [results/dev_ablation_final] [figs/detector_plasticity.png]
Panel (a): change score at eligible updates (t mod U == 0, t+1 >= 2D) with the SCAConfig-default flag level (0.08)
and the calibrated (d=4, L=2) level; flags shown for the calibrated detector.  Panel (b): realised plasticity of the
added lagged edge X2(t-1)->X3(t) vs the mean over the 23 other lag>=1 slots (full model, calibrated profile),
moving mean over 5 consecutive eligible updates (50 steps).
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/dev_ablation_final")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figs/detector_plasticity.png")
cal, dfl = R / "calibrated_d4_l2", R / "defaults"
A = np.load(cal / "arrays.npz"); meta = json.loads((cal / "summary.json").read_text())
theta_cal = meta["config"]["threshold"]; theta_def = json.loads((dfl / "summary.json").read_text())["config"]["threshold"]
U, D, CP = meta["config"]["update_interval"], meta["config"]["detection_window"], meta["data"]["change_point"]
scores, P = A["scores"], A["full_plasticity"]
tu = np.array([t for t in range(2, len(scores)) if t % U == 0 and t + 1 >= 2 * D])
flag = scores[tu] >= theta_cal
CH = (1, 2, 1)
OTHER = [(i, j, l) for i in range(4) for j in range(4) if i != j for l in (1, 2) if (i, j, l) != CH]
assert len(OTHER) == 23
# plasticity is defined from the second eligible update on
tp = np.array([t for t in range(2, len(scores)) if t % U == 0 and t + 1 >= 2 * D])
sm = lambda x, k=5: np.convolve(x, np.ones(k) / k, mode="valid")
chg = P[tp, CH[0], CH[1], CH[2]]
oth = np.array([[P[t, i, j, l] for (i, j, l) in OTHER] for t in tp]).mean(axis=1)

fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True)
a = ax[0]
a.plot(tu, scores[tu], lw=0.5, color="0.35", label="change score")
a.axhline(theta_def, color="C1", ls=":", lw=0.9, label=f"default level {theta_def:g}")
a.axhline(theta_cal, color="C2", ls=":", lw=0.9, label=f"calibrated level {theta_cal:.3f}")
a.scatter(tu[flag], np.full(flag.sum(), -0.02), s=2, color="C3", label="flagged (calibrated)")
a.axvline(CP, color="k", ls="--", lw=0.8, label="true change")
a.set_ylabel("score", fontsize=7); a.set_ylim(-0.04, 0.62)
a.legend(fontsize=5, loc="upper right", ncol=2, frameon=False)
a.set_title("(a) SCA change score at eligible updates", fontsize=7)
b = ax[1]
b.plot(tp[4:], sm(chg), lw=0.9, color="C2", label=r"X2$\to$X3, lag 1 (added edge)")
b.plot(tp[4:], sm(oth), lw=0.9, color="0.3", label="mean of 23 other lag$\\geq$1 slots")
b.axvline(CP, color="k", ls="--", lw=0.8)
b.axvspan(CP + 100, CP + 300, color="C1", alpha=0.2, lw=0)
b.set_ylabel("plasticity rate", fontsize=7); b.set_xlabel("time step", fontsize=7)
b.legend(fontsize=5.5, loc="upper right", frameon=False)
b.set_title("(b) Edgewise plasticity (5-update mean)", fontsize=7)
for x in ax: x.tick_params(labelsize=6)
plt.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out, dpi=300)
print("saved", out)
