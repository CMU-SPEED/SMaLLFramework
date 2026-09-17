#!/usr/bin/env python3
"""Generate separate Plotly plots for EXP microkernel range and performance.

The data was measured with the ZEN2 implementation. Regenerate both plots:

    python3 test/plot_exp_microkernels.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go


# Relative-error limit: 0.05% (5e-4). ns/element values use representative
# in-contract inputs. Negative/positive limits are magnitudes from zero.
MEASUREMENTS = (
    ("Horner", 1.0055, 1.3483, 0.087937),
    ("Negative Horner", 1.2248, 0.0, 0.093146),
    ("Range reduced", 0.0, 53.02, 0.148634),
    ("General", 86.29, 80.0, 0.311669),
    ("General positive", 0.0, 80.0, 0.220289),
    ("General negative", 87.33, 0.0, 0.238764),
)

# Variable-Horner sweep on [-1, 1].  Error is the maximum relative error over
# 2,001 equally spaced samples; timing is the measured ZEN2 throughput.
VARIABLE_HORNER_MEASUREMENTS = (
    (5, 1.93556603e-2, 0.093045),
    (6, 3.29666608e-3, 0.138998),
    (7, 4.78775648e-4, 0.102027),
    (8, 6.05963105e-5, 0.109862),
    (9, 6.80493304e-6, 0.114808),
    (10, 8.10111089e-7, 0.127896),
    (11, 1.61214132e-7, 0.139622),
    (12, 1.56136949e-7, 0.169344),
    (13, 1.48819396e-7, 0.179121),
)


def create_range_figure() -> go.Figure:
    """Plot the negative and positive input ranges correct to 0.05%."""
    names = [row[0] for row in MEASUREMENTS]
    negative_extent = [-row[1] for row in MEASUREMENTS]
    positive_extent = [row[2] for row in MEASUREMENTS]

    figure = go.Figure()
    figure.add_bar(
        name="negative input range",
        x=names,
        y=negative_extent,
        marker_color="#636EFA",
        hovertemplate="%{x}<br>correct down to %{y:.4f}<extra></extra>",
    )
    figure.add_bar(
        name="positive input range",
        x=names,
        y=positive_extent,
        marker_color="#EF553B",
        hovertemplate="%{x}<br>correct up to %{y:.4f}<extra></extra>",
    )
    figure.update_layout(
        title="EXP microkernel correctness ranges — ZEN2 (relative error ≤ 0.05%)",
        barmode="relative",
        template="plotly_white",
        yaxis_title="Input x",
        legend_title_text="Correct input side",
        height=550,
    )
    return figure


def create_performance_figure() -> go.Figure:
    """Plot speed normalized to the fastest measured baseline, Horner."""
    names = [row[0] for row in MEASUREMENTS]
    ns_per_element = [row[3] for row in MEASUREMENTS]
    relative_speed = [ns_per_element[0] / value for value in ns_per_element]

    figure = go.Figure(
        go.Bar(
            x=names,
            y=relative_speed,
            marker_color="#00CC96",
            text=[f"{value:.2f}×" for value in relative_speed],
            textposition="auto",
            customdata=ns_per_element,
            hovertemplate="%{x}<br>%{y:.2f}× Horner speed<br>"
                          "%{customdata:.3f} ns/element<extra></extra>",
        )
    )
    figure.update_layout(
        title="EXP microkernel relative performance — ZEN2",
        template="plotly_white",
        yaxis_title="Speed relative to Horner (higher is better)",
        height=550,
    )
    return figure


def create_variable_horner_figure() -> go.Figure:
    """Plot the measured speed/accuracy trade-off for 5--13 coefficients."""
    terms = [row[0] for row in VARIABLE_HORNER_MEASUREMENTS]
    maximum_relative_error = [row[1] for row in VARIABLE_HORNER_MEASUREMENTS]
    ns_per_element = [row[2] for row in VARIABLE_HORNER_MEASUREMENTS]

    figure = go.Figure(
        go.Scatter(
            x=ns_per_element,
            y=maximum_relative_error,
            mode="markers+text",
            text=[str(value) for value in terms],
            texttemplate="%{text} terms",
            textposition="top center",
            marker={"size": 12, "color": terms, "colorscale": "Viridis",
                    "colorbar": {"title": "Terms"}},
            hovertemplate="%{text}<br>%{x:.6f} ns/element<br>"
                          "max relative error: %{y:.3e}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Variable-Horner EXP: throughput versus accuracy — ZEN2",
        template="plotly_white",
        xaxis_title="Throughput (ns / element; lower is better)",
        yaxis_title="Maximum relative error on [-1, 1] (lower is better)",
        height=550,
    )
    figure.update_yaxes(type="log")
    return figure


def create_recommendation_figure() -> go.Figure:
    """Create a range selector and the bar chart it controls."""
    names = [row[0] for row in MEASUREMENTS]
    ns_per_element = [row[3] for row in MEASUREMENTS]
    relative_speed = [ns_per_element[0] / value for value in ns_per_element]

    figure = go.Figure()
    figure.add_bar(
        x=names,
        y=relative_speed,
        marker_color=["#00CC96"] + ["#B8C2CC"] * (len(names) - 1),
        customdata=ns_per_element,
        text=[f"{value:.2f}×" for value in relative_speed],
        textposition="auto",
        hovertemplate="%{x}<br>%{y:.2f}× Horner speed<br>"
                      "%{customdata:.3f} ns/element<extra></extra>",
        yaxis="y",
    )
    figure.add_scatter(
        x=[-21.1409, 21.1400],
        y=[0.0, 0.0],
        mode="lines",
        line={"color": "#636EFA", "width": 4},
        hoverinfo="skip",
        xaxis="x",
        yaxis="y2",
    )
    figure.update_layout(
        title=("EXP microkernel recommendation for input range [-1, 1]<br>"
               "<sup>Drag the range slider; green/orange bars are the fastest "
               "negative/positive-side choices.</sup>"),
        template="plotly_white",
        height=720,
        yaxis={"domain": [0.0, 0.68],
               "title": "Speed relative to Horner (higher is better)"},
        xaxis={"domain": [0.0, 1.0], "visible": False,
               "range": [-1.0, 1.0],
               "rangeslider": {"visible": True, "thickness": 0.18}},
        yaxis2={"domain": [0.78, 1.0], "visible": False},
    )
    return figure


def write_recommendation_figure(path: Path) -> None:
    """Write a figure whose Plotly range slider updates the recommendation."""
    plot_data = [
        {"name": name, "negative_limit": negative, "positive_limit": positive,
         "ns_per_element": timing}
        for name, negative, positive, timing in MEASUREMENTS
    ]
    post_script = f"""
const kernels = {json.dumps(plot_data)};
const plot = document.getElementById('{{plot_id}}');
function fastest(limit, side) {{
  const valid = kernels
    .map((kernel, index) => ({{kernel, index}}))
    .filter(item => item.kernel[side] >= limit);
  return valid.length ? valid.reduce((best, item) =>
    item.kernel.ns_per_element < best.kernel.ns_per_element ? item : best) : null;
}}
function updateRecommendation(range) {{
  const lower = Math.min(range[0], range[1]);
  const upper = Math.max(range[0], range[1]);
  const negative = lower < 0 ? fastest(-lower, 'negative_limit') : null;
  const positive = upper > 0 ? fastest(upper, 'positive_limit') : null;
  const colors = kernels.map(() => '#B8C2CC');
  const choices = [];
  if (negative) {{ colors[negative.index] = '#636EFA'; choices.push('negative: ' + negative.kernel.name); }}
  if (positive) {{ colors[positive.index] = negative && positive.index === negative.index ? '#00CC96' : '#EF553B'; choices.push('positive: ' + positive.kernel.name); }}
  if (!negative && lower < 0) choices.push('negative: no valid kernel');
  if (!positive && upper > 0) choices.push('positive: no valid kernel');
  const title = 'EXP microkernel recommendation for input range [' + lower.toFixed(2) + ', ' + upper.toFixed(2) + ']' +
    '<br><sup>' + (choices.length ? choices.join(' | ') : 'range contains only zero') +
    '. Blue = negative side; orange = positive side; green = one kernel covers both.</sup>';
  Plotly.restyle(plot, {{'marker.color': [colors]}}, [0]);
  Plotly.relayout(plot, {{title: title}});
}}
plot.on('plotly_relayout', event => {{
  const lower = event['xaxis.range[0]'];
  const upper = event['xaxis.range[1]'];
  if (lower !== undefined && upper !== undefined) updateRecommendation([lower, upper]);
}});
updateRecommendation([-1.0, 1.0]);
"""
    create_recommendation_figure().write_html(path, include_plotlyjs=True,
                                               post_script=post_script)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test/exp_microkernel_plots"),
        help="Directory for the two HTML plots (default: %(default)s)",
    )
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    range_path = arguments.output_dir / "exp_microkernel_ranges.html"
    performance_path = arguments.output_dir / "exp_microkernel_performance.html"
    variable_horner_path = arguments.output_dir / "exp_variable_horner_tradeoff.html"
    recommendation_path = arguments.output_dir / "exp_microkernel_recommendation.html"
    create_range_figure().write_html(range_path, include_plotlyjs=True)
    create_performance_figure().write_html(performance_path, include_plotlyjs=True)
    create_variable_horner_figure().write_html(variable_horner_path,
                                                include_plotlyjs=True)
    write_recommendation_figure(recommendation_path)
    print(f"Wrote {range_path}")
    print(f"Wrote {performance_path}")
    print(f"Wrote {variable_horner_path}")
    print(f"Wrote {recommendation_path}")


if __name__ == "__main__":
    main()
