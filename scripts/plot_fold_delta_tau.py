#!/usr/bin/env python3
"""Render the reviewed fold delta-tau diagnostic as a dependency-free SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    subject = payload["summary"]["subject"]
    folds = subject["fold_uncertainty"]["fold_deltas"]
    labels = ["Fold 0", "Fold 1", "Fold 2"]
    values = [folds[f"fold_{index}"]["kendall_tau_b"] for index in range(3)]
    overall = subject["paired_comparison"]["kendall_tau_b"]["mean_delta"]
    without_fold_2 = subject["fold_uncertainty"]["leave_one_fold_out"]["fold_2"][
        "kendall_tau_b"
    ]

    width, height = 900, 470
    left, right, top, bottom = 105, 45, 75, 100
    chart_width = width - left - right
    chart_height = height - top - bottom
    maximum = 0.38

    def y(value: float) -> float:
        return top + chart_height * (1 - value / maximum)

    colors = ["#4C78A8", "#72B7B2", "#F58518"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}'
        '.tick{font-size:14px;fill:#5b6474}.label{font-size:16px;font-weight:600}'
        '.value{font-size:17px;font-weight:700}.note{font-size:14px;fill:#5b6474}</style>',
        '<text x="45" y="36" font-size="24" font-weight="700">'
        'Increment over source-performance ranking</text>',
        '<text x="45" y="60" class="note">Subject-macro Kendall τ; post-target diagnostic</text>',
    ]
    for tick in (0.0, 0.1, 0.2, 0.3):
        tick_y = y(tick)
        parts.extend(
            [
                f'<line x1="{left}" y1="{tick_y:.1f}" x2="{width-right}" '
                'y2="{:.1f}" stroke="#e5e9f0"/>'.format(tick_y),
                f'<text x="{left-16}" y="{tick_y+5:.1f}" text-anchor="end" '
                f'class="tick">{tick:.1f}</text>',
            ]
        )
    bar_width = 125
    gap = chart_width / 3
    for index, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        center = left + gap * (index + 0.5)
        bar_y = y(value)
        parts.extend(
            [
                f'<rect x="{center-bar_width/2:.1f}" y="{bar_y:.1f}" '
                f'width="{bar_width}" height="{top+chart_height-bar_y:.1f}" '
                f'rx="5" fill="{color}"/>',
                f'<text x="{center:.1f}" y="{bar_y-11:.1f}" text-anchor="middle" '
                f'class="value">+{value:.3f}</text>',
                f'<text x="{center:.1f}" y="{top+chart_height+30}" text-anchor="middle" '
                f'class="label">{label}</text>',
            ]
        )
    overall_y = y(overall)
    parts.extend(
        [
            f'<line x1="{left}" y1="{overall_y:.1f}" x2="{width-right}" '
            f'y2="{overall_y:.1f}" stroke="#7b61a8" stroke-width="2" '
            'stroke-dasharray="8 6"/>',
            f'<text x="{left+10}" y="{overall_y-8:.1f}" text-anchor="start" '
            f'fill="#7b61a8" font-size="14">All folds: +{overall:.3f}</text>',
            f'<text x="{left}" y="{height-35}" class="note">'
            f'Leave Fold 2 out: Δτ = +{without_fold_2:.3f}. '
            'Three folds are insufficient for a strong inferential claim.</text>',
            '</svg>',
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
