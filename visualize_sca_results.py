import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LOAD RESULTS
# ============================================================

A = np.load("results/sca_A_pred.npy")
P = np.load("results/sca_plasticity_rates.npy")
change_scores = np.load("results/sca_change_scores.npy")
detected = np.load("results/sca_detected_flags.npy")

os.makedirs("results/plots", exist_ok=True)

print("A_pred:", A.shape)
print("Plasticity:", P.shape)


# ============================================================
# EDGE DEFINITIONS
# ============================================================

EDGES = {
    "X1 → X4 (lag 2)": (0, 3, 2),
    "X4 → X3 (lag 0)": (3, 2, 0),
    "X3 → X2 (lag 1)": (2, 1, 1),
    "X2 → X1 (lag 0)": (1, 0, 0),
    "X2 → X3 (lag 1)": (1, 2, 1),
}


# ============================================================
# 1. CAUSAL EDGE TRAJECTORIES
# ============================================================

plt.figure(figsize=(12, 6))

for name, (s, t, l) in EDGES.items():
    plt.plot(
        A[:, s, t, l],
        label=name
    )

plt.axvline(
    2500,
    linestyle="--",
    label="Regime change"
)

plt.xlabel("Time")
plt.ylabel("Causal strength")
plt.title("Causal Edge Trajectories")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/plots/causal_edge_trajectories.png",
    dpi=200
)

plt.close()


# ============================================================
# 2. PLASTICITY TRAJECTORIES
# ============================================================

plt.figure(figsize=(12, 6))

for name, (s, t, l) in EDGES.items():
    plt.plot(
        P[:, s, t, l],
        label=name
    )

plt.axvline(
    2500,
    linestyle="--",
    label="Regime change"
)

plt.xlabel("Time")
plt.ylabel("Plasticity rate")
plt.title("Selective Causal Plasticity Over Time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/plots/plasticity_trajectories.png",
    dpi=200
)

plt.close()


# ============================================================
# 3. CHANGE DETECTION
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    change_scores,
    label="Change score"
)

plt.axhline(
    0.08,
    linestyle="--",
    label="Detection threshold"
)

plt.axvline(
    2500,
    linestyle="--",
    label="True regime change"
)

plt.xlabel("Time")
plt.ylabel("Change score")
plt.title("Regime Change Detection")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/plots/change_detection.png",
    dpi=200
)

plt.close()


# ============================================================
# 4. DETECTION FLAGS
# ============================================================

plt.figure(figsize=(12, 3))

plt.plot(
    detected.astype(int)
)

plt.axvline(
    2500,
    linestyle="--",
    label="True regime change"
)

plt.xlabel("Time")
plt.ylabel("Detected")
plt.title("Detected Regime Changes")
plt.yticks([0, 1])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/plots/detection_flags.png",
    dpi=200
)

plt.close()


# ============================================================
# 5. PLASTICITY HEATMAP AROUND CHANGE
# ============================================================

# Average plasticity during the post-change period

post = np.mean(
    P[2600:2800],
    axis=0
)

# Average over lag dimension
post_2d = np.mean(
    post,
    axis=2
)

plt.figure(figsize=(7, 6))

plt.imshow(
    post_2d,
    aspect="auto"
)

plt.colorbar(
    label="Plasticity"
)

plt.xticks(
    range(4),
    ["X1", "X2", "X3", "X4"]
)

plt.yticks(
    range(4),
    ["X1", "X2", "X3", "X4"]
)

plt.xlabel("Target")
plt.ylabel("Source")
plt.title("Post-Change Average Plasticity")

plt.tight_layout()

plt.savefig(
    "results/plots/plasticity_heatmap.png",
    dpi=200
)

plt.close()


# ============================================================
# DONE
# ============================================================

print()
print("Visualization complete.")
print()
print("Saved plots:")

for filename in [
    "causal_edge_trajectories.png",
    "plasticity_trajectories.png",
    "change_detection.png",
    "detection_flags.png",
    "plasticity_heatmap.png",
]:
    print(
        "results/plots/" + filename
    )