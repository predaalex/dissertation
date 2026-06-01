from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METRICS = [
    ("helpfulness", "Helpfulness", False),
    ("emotional_appropriateness", "Emotional appropriateness", False),
    ("empathy_supportiveness", "Empathy/supportiveness", False),
    ("overreaction_risk", "Overreaction risk", True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create statistics from scores_paired.csv.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation") / "results" / "scores_paired.csv",
        help="Path to scores_paired.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <input folder>/statistics.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score(row: dict[str, str], column: str) -> int:
    return int(round(float(row[column])))


def mean(values: list[int]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def pct(count: int, total: int) -> float:
    return round(100 * count / total, 2) if total else 0.0


def label(value: str) -> str:
    return {
        "emotion_aware": "Emotion-aware",
        "baseline": "Baseline",
        "tie": "Tie",
    }.get(value, value)


def better_variant(delta: float, lower_is_better: bool) -> str:
    if abs(delta) < 0.001:
        return "tie"
    if lower_is_better:
        return "emotion_aware" if delta < 0 else "baseline"
    return "emotion_aware" if delta > 0 else "baseline"


def build_metric_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    total = len(rows)
    for metric, pretty_name, lower_is_better in METRICS:
        baseline_mean = mean([score(row, f"baseline_{metric}") for row in rows])
        emotion_mean = mean([score(row, f"emotion_aware_{metric}") for row in rows])
        delta = round(emotion_mean - baseline_mean, 3)
        winners = Counter(row[f"{metric}_winner"] for row in rows)
        result.append(
            {
                "metric": pretty_name,
                "baseline_mean": baseline_mean,
                "emotion_aware_mean": emotion_mean,
                "difference_emotion_minus_baseline": delta,
                "better_variant": label(better_variant(delta, lower_is_better)),
                "emotion_aware_wins": winners["emotion_aware"],
                "emotion_aware_win_rate": pct(winners["emotion_aware"], total),
                "baseline_wins": winners["baseline"],
                "baseline_win_rate": pct(winners["baseline"], total),
                "ties": winners["tie"],
                "tie_rate": pct(winners["tie"], total),
            }
        )
    return result


def build_overall_preference(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    total = len(rows)
    counts = Counter(row["overall_preference"] for row in rows)
    return [
        {"variant": label(variant), "count": counts[variant], "percentage": pct(counts[variant], total)}
        for variant in ["emotion_aware", "baseline", "tie"]
    ]


def build_preference_by_emotion(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["emotion"]].append(row)

    result = []
    for emotion in sorted(grouped):
        items = grouped[emotion]
        total = len(items)
        counts = Counter(row["overall_preference"] for row in items)
        result.append(
            {
                "emotion": emotion,
                "count": total,
                "emotion_aware_preferred": counts["emotion_aware"],
                "emotion_aware_preferred_rate": pct(counts["emotion_aware"], total),
                "baseline_preferred": counts["baseline"],
                "baseline_preferred_rate": pct(counts["baseline"], total),
                "ties": counts["tie"],
                "tie_rate": pct(counts["tie"], total),
            }
        )
    return result


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_markdown_report(
    metric_summary: list[dict[str, Any]],
    overall_preference: list[dict[str, Any]],
    preference_by_emotion: list[dict[str, Any]],
    total: int,
) -> str:
    metric_table = markdown_table(
        [
            "Metric",
            "Baseline mean",
            "Emotion-aware mean",
            "Delta",
            "Better",
            "Emotion-aware wins",
            "Baseline wins",
            "Ties",
        ],
        [
            [
                row["metric"],
                row["baseline_mean"],
                row["emotion_aware_mean"],
                row["difference_emotion_minus_baseline"],
                row["better_variant"],
                f"{row['emotion_aware_wins']} ({row['emotion_aware_win_rate']}%)",
                f"{row['baseline_wins']} ({row['baseline_win_rate']}%)",
                f"{row['ties']} ({row['tie_rate']}%)",
            ]
            for row in metric_summary
        ],
    )
    preference_table = markdown_table(
        ["Variant", "Count", "Percentage"],
        [[row["variant"], row["count"], f"{row['percentage']}%"] for row in overall_preference],
    )
    emotion_table = markdown_table(
        ["Emotion", "Count", "Emotion-aware", "Baseline", "Tie"],
        [
            [
                row["emotion"],
                row["count"],
                f"{row['emotion_aware_preferred']} ({row['emotion_aware_preferred_rate']}%)",
                f"{row['baseline_preferred']} ({row['baseline_preferred_rate']}%)",
                f"{row['ties']} ({row['tie_rate']}%)",
            ]
            for row in preference_by_emotion
        ],
    )
    return "\n\n".join(
        [
            "# Emotion-Aware Ablation Statistics",
            f"Evaluated interactions: {total}",
            "## Metric Summary",
            metric_table,
            "## Overall Preference",
            preference_table,
            "## Preference by Emotion",
            emotion_table,
        ]
    )


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise ValueError("The input scores file is empty.")

    output_dir = args.output_dir or args.input.parent / "statistics"
    metric_summary = build_metric_summary(rows)
    overall_preference = build_overall_preference(rows)
    preference_by_emotion = build_preference_by_emotion(rows)

    write_csv(
        output_dir / "metric_summary.csv",
        metric_summary,
        [
            "metric",
            "baseline_mean",
            "emotion_aware_mean",
            "difference_emotion_minus_baseline",
            "better_variant",
            "emotion_aware_wins",
            "emotion_aware_win_rate",
            "baseline_wins",
            "baseline_win_rate",
            "ties",
            "tie_rate",
        ],
    )
    write_csv(
        output_dir / "overall_preference.csv",
        overall_preference,
        ["variant", "count", "percentage"],
    )
    write_csv(
        output_dir / "preference_by_emotion.csv",
        preference_by_emotion,
        [
            "emotion",
            "count",
            "emotion_aware_preferred",
            "emotion_aware_preferred_rate",
            "baseline_preferred",
            "baseline_preferred_rate",
            "ties",
            "tie_rate",
        ],
    )

    report = build_markdown_report(
        metric_summary,
        overall_preference,
        preference_by_emotion,
        total=len(rows),
    )
    (output_dir / "statistics_report.md").write_text(report + "\n", encoding="utf-8")

    print(f"Wrote statistics to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None
