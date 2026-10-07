"""Create vector publication figures from completed, validated CSV artifacts.

The script emits dependency-free SVG so figures can be versioned, reviewed, and
included in a manuscript without a GUI or a plotting-backend dependency.
"""

from html import escape
from pathlib import Path
import math

import pandas as pd


COLORS = {"SCA": "#1f4e79", "Rolling OLS": "#c55a11"}


def _svg(width, height, body):
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><style>text{{font-family:Arial,sans-serif;fill:#202020}}.axis{{stroke:#555;stroke-width:1}}.grid{{stroke:#d9e1f2;stroke-width:1}}.title{{font-size:18px;font-weight:bold}}.label{{font-size:12px}}.small{{font-size:10px}}</style>{body}</svg>'


def _scaling_figure(summary, target):
    subset = summary[(summary.noise_scale == .1) & (summary.distribution == "gaussian")]
    if subset.empty:
        raise ValueError("Expected Gaussian sigma=0.1 rows for scaling figure")
    width, height, left, bottom = 900, 470, 70, 60
    values = subset.f1_mean
    ymax = max(.1, math.ceil(values.max() * 10) / 10)
    body = ['<rect width="100%" height="100%" fill="white"/>', '<text x="70" y="28" class="title">Synthetic scaling: F1 by graph size and lag</text>']
    for tick in range(6):
        y = height - bottom - tick * (height - bottom - 60) / 5
        value = ymax * tick / 5
        body += [f'<line x1="{left}" y1="{y:.1f}" x2="860" y2="{y:.1f}" class="grid"/>', f'<text x="25" y="{y + 4:.1f}" class="small">{value:.1f}</text>']
    body += [f'<line x1="{left}" y1="60" x2="{left}" y2="{height-bottom}" class="axis"/>', f'<line x1="{left}" y1="{height-bottom}" x2="860" y2="{height-bottom}" class="axis"/>']
    dimensions = sorted(subset.n_variables.unique())
    lags = sorted(subset.max_lag.unique())
    methods = ["SCA", "Rolling OLS"]
    group = (860 - left) / len(dimensions)
    bar_width = group / (len(lags) * len(methods) + 2)
    for di, dimension in enumerate(dimensions):
        x0 = left + di * group + group * .12
        body.append(f'<text x="{left + di * group + group/2:.1f}" y="{height-bottom+20}" text-anchor="middle" class="label">d={dimension}</text>')
        for li, lag in enumerate(lags):
            for mi, method in enumerate(methods):
                row = subset[(subset.n_variables == dimension) & (subset.max_lag == lag) & (subset.method == method)]
                if row.empty:
                    continue
                r = row.iloc[0]; x = x0 + (li * len(methods) + mi) * bar_width
                h = r.f1_mean / ymax * (height - bottom - 60); y = height - bottom - h
                body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width-2:.1f}" height="{h:.1f}" fill="{COLORS[method]}"/>')
                body.append(f'<line x1="{x + bar_width/2:.1f}" y1="{max(60, y-r.f1_std/ymax*(height-bottom-60)):.1f}" x2="{x + bar_width/2:.1f}" y2="{y + r.f1_std/ymax*(height-bottom-60):.1f}" stroke="black"/>')
    body += ['<rect x="650" y="38" width="12" height="12" fill="#1f4e79"/><text x="668" y="49" class="small">SCA</text>', '<rect x="735" y="38" width="12" height="12" fill="#c55a11"/><text x="753" y="49" class="small">Rolling OLS</text>', '<text x="450" y="462" text-anchor="middle" class="label">Each cluster contains L=1, 2, 4 (left to right per method)</text>']
    target.write_text(_svg(width, height, "".join(body)), encoding="utf-8")


def _sensitivity_figure(summary, target):
    width, height = 900, 650
    body = ['<rect width="100%" height="100%" fill="white"/>', '<text x="45" y="28" class="title">Hyperparameter sensitivity (mean F1 ± SD; tuning seeds 0–19)</text>']
    parameters = list(summary.parameter.unique())
    for index, parameter in enumerate(parameters):
        col, row = index % 2, index // 2
        left, top, panel_w, panel_h = 55 + col * 430, 60 + row * 185, 360, 130
        data = summary[summary.parameter == parameter].sort_values("value")
        ymin, ymax = max(0, data.f1_mean.min() - .08), min(1, data.f1_mean.max() + .08)
        body += [f'<text x="{left}" y="{top}" class="label">{escape(parameter)}</text>', f'<line x1="{left}" y1="{top+panel_h}" x2="{left+panel_w}" y2="{top+panel_h}" class="axis"/>', f'<line x1="{left}" y1="{top+15}" x2="{left}" y2="{top+panel_h}" class="axis"/>']
        for i, value in enumerate(data.itertuples()):
            x = left + (panel_w / max(1, len(data)-1)) * i
            y = top + panel_h - (value.f1_mean-ymin) / (ymax-ymin) * (panel_h-20)
            err = value.f1_std / (ymax-ymin) * (panel_h-20)
            body += [f'<line x1="{x:.1f}" y1="{y-err:.1f}" x2="{x:.1f}" y2="{y+err:.1f}" stroke="#1f4e79"/>', f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#1f4e79"/>', f'<text x="{x:.1f}" y="{top+panel_h+16}" text-anchor="middle" class="small">{value.value:g}</text>']
            if i:
                previous = data.iloc[i-1]
                px = left + (panel_w / max(1, len(data)-1)) * (i-1)
                py = top + panel_h - (previous.f1_mean-ymin)/(ymax-ymin)*(panel_h-20)
                body.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#1f4e79" stroke-width="2"/>')
    target.write_text(_svg(width, height, "".join(body)), encoding="utf-8")


def main():
    output = Path("results/publication_figures"); output.mkdir(parents=True, exist_ok=True)
    synthetic = pd.read_csv("results/validation/synthetic_mean_std.csv")
    sensitivity = pd.read_csv("results/validation/sensitivity_mean_std.csv")
    raw = pd.read_csv("results/validation/synthetic_raw.csv")
    sensitivity_raw = pd.read_csv("results/validation/sensitivity_raw.csv")
    if (
        set(raw.get("seed", [])) != set(range(20, 40))
        or raw.get("synthetic_protocol", pd.Series(dtype=str)).nunique() != 1
        or set(raw.get("synthetic_protocol", pd.Series(dtype=str))) != {"v2_recurring_graphs_measurement_snr"}
    ):
        raise SystemExit("Refusing figures: the completed held-out v2 synthetic results are required.")
    if (
        set(sensitivity_raw.get("seed", [])) != set(range(20))
        or set(sensitivity_raw.get("synthetic_protocol", pd.Series(dtype=str))) != {"v2_recurring_graphs_measurement_snr"}
    ):
        raise SystemExit("Refusing figures: the completed v2 tuning results are required.")
    _scaling_figure(synthetic, output / "figure_scaling_f1.svg")
    _sensitivity_figure(sensitivity, output / "figure_hyperparameter_sensitivity.svg")
    print(output)


if __name__ == "__main__":
    main()
