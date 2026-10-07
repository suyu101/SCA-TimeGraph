import pandas as pd


# ============================================================
# Load validated SCA ablation results
# ============================================================

sca = pd.read_csv(
    "results/selective_plasticity_ablation.csv"
)

sca_rows = []

for _, row in sca.iterrows():
    sca_rows.append(
        {
            "Model": row["Model"],
            "Before F1": row["Before F1"],
            "Before SHD": row["Before SHD"],
            "After F1": row["After F1"],
            "After SHD": row["After SHD"],
            "Stable Preservation": row["Stable Preservation"],
            "Change Detection": row["Detection Accuracy"],
            "Adaptation Delay": row["Adaptation Delay"],
            "Unnecessary Change": row["Unnecessary Rate"],
        }
    )


# ============================================================
# Validated rolling regression baseline
# ============================================================

baseline_row = {
    "Model": "Rolling Regression",
    "Before F1": 1.0000,
    "Before SHD": 0,
    "After F1": 0.7500,
    "After SHD": 2,
    "Stable Preservation": 0.7603,
    "Change Detection": 0.8592,
    "Adaptation Delay": 352,
    "Unnecessary Change": 0.2397,
}


# ============================================================
# Validated PCMCI+ static baseline
#
# PCMCI+ produces one static graph, so dynamic adaptation
# metrics are intentionally left blank.
# ============================================================

pcmci_row = {
    "Model": "PCMCI+",
    "Before F1": 0.8889,
    "Before SHD": 1,
    "After F1": None,
    "After SHD": None,
    "Stable Preservation": None,
    "Change Detection": None,
    "Adaptation Delay": None,
    "Unnecessary Change": None,
}


# ============================================================
# Final ordering
# ============================================================

results = (
    sca_rows
    + [baseline_row]
    + [pcmci_row]
)

df = pd.DataFrame(results)


# ============================================================
# Print final table
# ============================================================

print("===== FINAL MODEL COMPARISON =====")
print()

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# Save
# ============================================================

output_path = "results/model_comparison.csv"

df.to_csv(
    output_path,
    index=False,
)

print()
print(f"Saved: {output_path}")
