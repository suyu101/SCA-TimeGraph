import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

CHANGE_POINT = 2500

VARIABLES = ["X1", "X2", "X3", "X4"]

# Important edges for the controlled regime-change experiment
EDGES = {
    "X4 -> X3 (lag 0)": (3, 2, 0),
    "X2 -> X3 (lag 1)": (1, 2, 1),
    "X1 -> X4 (lag 2)": (0, 3, 2),
    "X2 -> X1 (lag 0)": (1, 0, 0),
    "X3 -> X2 (lag 1)": (2, 1, 1),
}


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 70)
print("SCA RESULT VISUALIZATION")
print("=" * 70)

A = np.load(
    os.path.join(RESULTS_DIR, "sca_full_A_pred.npy")
)

P = np.load(
    os.path.join(RESULTS_DIR, "sca_full_plasticity_rates.npy")
)

change_scores = np.load(
    os.path.join(RESULTS_DIR, "sca_change_scores.npy")
)

detected_flags = np.load(
    os.path.join(RESULTS_DIR, "sca_detected_flags.npy")
)

print("A_pred:", A.shape)
print("Plasticity:", P.shape)
print("Change scores:", change_scores.shape)
print("Detected flags:", detected_flags.shape)


# ============================================================
# PLOT 1 — CAUSAL EDGE TRAJECTORIES
# ============================================================

print()
print("Creating Plot 1: Causal edge trajectories...")

plt.figure(figsize=(12, 6))

time = np.arange(len(A))

for name, (s, t, lag) in EDGES.items():

    values = A[:, s, t, lag]

    plt.plot(
        time,
        values,
        label=name,
        linewidth=1.5
    )

plt.axvline(
    CHANGE_POINT,
    linestyle="--",
    linewidth=2,
    label="Regime change"
)

plt.xlabel("Time")
plt.ylabel("Causal strength")
plt.title("SCA Causal Edge Trajectories")

plt.legend(
    loc="upper right",
    fontsize=9
)

plt.grid(alpha=0.3)

plt.tight_layout()

path = os.path.join(
    PLOTS_DIR,
    "01_causal_edge_trajectories.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path)


# ============================================================
# PLOT 2 — ZOOMED EDGE TRAJECTORIES
# ============================================================

print()
print("Creating Plot 2: Regime-change zoom...")

plt.figure(figsize=(12, 6))

start = 2200
end = 2800

for name, (s, t, lag) in EDGES.items():

    values = A[
        start:end,
        s,
        t,
        lag
    ]

    plt.plot(
        np.arange(start, end),
        values,
        label=name,
        linewidth=2
    )

plt.axvline(
    CHANGE_POINT,
    linestyle="--",
    linewidth=2,
    label="Regime change"
)

plt.xlabel("Time")
plt.ylabel("Causal strength")

plt.title(
    "SCA Causal Adaptation Around Regime Change"
)

plt.legend(
    fontsize=9
)

plt.grid(alpha=0.3)

plt.tight_layout()

path = os.path.join(
    PLOTS_DIR,
    "02_regime_change_zoom.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path)


# ============================================================
# PLOT 3 — PLASTICITY TRAJECTORIES
# ============================================================

print()
print("Creating Plot 3: Plasticity trajectories...")

plt.figure(figsize=(12, 6))

for name, (s, t, lag) in EDGES.items():

    values = P[
        :,
        s,
        t,
        lag
    ]

    plt.plot(
        time,
        values,
        label=name,
        linewidth=1.5
    )

plt.axvline(
    CHANGE_POINT,
    linestyle="--",
    linewidth=2,
    label="Regime change"
)

plt.xlabel("Time")
plt.ylabel("Plasticity rate")

plt.title(
    "Selective Causal Plasticity Over Time"
)

plt.legend(
    fontsize=9
)

plt.grid(alpha=0.3)

plt.tight_layout()

path = os.path.join(
    PLOTS_DIR,
    "03_plasticity_trajectories.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path)


# ============================================================
# PLOT 4 — CHANGE DETECTION
# ============================================================

print()
print("Creating Plot 4: Change detection...")

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    change_scores,
    linewidth=1.5,
    label="Change score"
)

# Detection threshold from the current SCA configuration
CHANGE_THRESHOLD = 0.08

plt.axhline(
    CHANGE_THRESHOLD,
    linestyle=":",
    linewidth=2,
    label="Detection threshold"
)

plt.axvline(
    CHANGE_POINT,
    linestyle="--",
    linewidth=2,
    label="True regime change"
)

detected_times = time[detected_flags]

if len(detected_times) > 0:

    plt.scatter(
        detected_times,
        change_scores[detected_flags],
        s=15,
        label="Detected changes"
    )

plt.xlabel("Time")
plt.ylabel("Change score")

plt.title(
    "SCA Regime-Change Detection"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

path = os.path.join(
    PLOTS_DIR,
    "04_change_detection.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path)


# ============================================================
# PLOT 5 — ABLATION COMPARISON
# ============================================================

print()
print("Creating Plot 5: Ablation comparison...")

csv_path = os.path.join(
    RESULTS_DIR,
    "selective_plasticity_ablation.csv"
)

df = pd.read_csv(csv_path)

print()
print("Ablation data:")
print(df.to_string(index=False))


# ------------------------------------------------------------
# After F1
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["After F1"]
)

plt.ylabel("After-regime F1")
plt.title("Ablation: Causal Recovery After Regime Change")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    PLOTS_DIR,
    "05_ablation_f1.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path)


# ------------------------------------------------------------
# Selectivity
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["Selectivity Ratio"]
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
    label="No selectivity (1.0x)"
)

plt.ylabel("Plasticity selectivity ratio")
plt.title("Ablation: Selective Plasticity")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    PLOTS_DIR,
    "06_ablation_selectivity.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path)


# ------------------------------------------------------------
# Stable preservation
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["Stable Preservation"]
)

plt.ylabel("Stable-edge preservation")
plt.title("Ablation: Stable Causal Structure Preservation")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    PLOTS_DIR,
    "07_ablation_stable_preservation.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print()
print("Plots saved in:")
print(PLOTS_DIR)

print()
print("Generated files:")

for filename in sorted(os.listdir(PLOTS_DIR)):
    print("  ", filename)

print()
print("DONE")