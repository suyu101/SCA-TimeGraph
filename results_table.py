import pandas as pd


RESULTS = [
    {
        "Model": "Rolling Regression",
        "Before F1": 1.0000,
        "Before SHD": 0,
        "After F1": 0.7500,
        "After SHD": 2,
        "Stable Preservation": 0.7603,
        "Change Detection": 0.8592,
        "Adaptation Delay": 352,
        "Unnecessary Change": 0.2397,
    },
    {
        "Model": "PCMCI+",
        "Before F1": 0.8889,
        "Before SHD": 1,
        "After F1": None,
        "After SHD": None,
        "Stable Preservation": None,
        "Change Detection": None,
        "Adaptation Delay": None,
        "Unnecessary Change": None,
    },
]


df = pd.DataFrame(RESULTS)

print("===== MODEL COMPARISON =====")
print()

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

df.to_csv(
    "results/model_comparison.csv",
    index=False,
)

print()
print("Saved: results/model_comparison.csv")
