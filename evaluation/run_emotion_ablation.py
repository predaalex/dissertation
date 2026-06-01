from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "gemma4:e4b"
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_ATTEMPTS = 3

METRICS = [
    ("helpfulness", False),
    ("emotional_appropriateness", False),
    ("empathy_supportiveness", False),
    ("overreaction_risk", True),
]

RESPONSE_FIELDS = [
    "id",
    "query",
    "emotion",
    "confidence",
    "baseline_response",
    "emotion_aware_response",
]

SCORE_FIELDS = [
    "id",
    "query",
    "emotion",
    "confidence",
    "baseline_helpfulness",
    "emotion_aware_helpfulness",
    "helpfulness_winner",
    "baseline_emotional_appropriateness",
    "emotion_aware_emotional_appropriateness",
    "emotional_appropriateness_winner",
    "baseline_empathy_supportiveness",
    "emotion_aware_empathy_supportiveness",
    "empathy_supportiveness_winner",
    "baseline_overreaction_risk",
    "emotion_aware_overreaction_risk",
    "overreaction_risk_winner",
    "overall_preference",
]

ANALYSIS_FIELDS = [
    "query",
    "emotion",
    "Helpfulness",
    "Emotional appropriateness",
    "Empathy/supportiveness",
    "Overreaction risk",
    "Overall preference",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline vs emotion-aware evaluation using gemma4:e4b."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation") / "interactions.csv",
        help="CSV file with id, query, emotion and confidence columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation") / "results",
        help="Directory where responses and scores will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N rows. Useful for testing with --limit 1.",
    )
    return parser.parse_args()


def read_interactions(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]

    for index, row in enumerate(rows, start=1):
        if not row.get("query"):
            raise ValueError(f"Missing query on row {index}.")
        if not row.get("emotion"):
            raise ValueError(f"Missing emotion on row {index}.")
        row["id"] = row.get("id") or str(index)
        row["confidence"] = row.get("confidence") or ""
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_result_files(
    output_dir: Path,
    responses: list[dict[str, Any]],
    paired_scores: list[dict[str, Any]],
    analysis_rows: list[dict[str, Any]],
) -> None:
    write_csv(output_dir / "responses.csv", responses, RESPONSE_FIELDS)
    write_csv(output_dir / "scores_paired.csv", paired_scores, SCORE_FIELDS)
    write_csv(output_dir / "analysis_table.csv", analysis_rows, ANALYSIS_FIELDS)


def append_debug_error(
    output_dir: Path,
    item_id: str,
    stage: str,
    attempt: int,
    error: Exception | str,
    raw_text: str = "",
) -> None:
    path = output_dir / "debug_errors.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "stage", "attempt", "error", "raw_text"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "id": item_id,
                "stage": stage,
                "attempt": attempt,
                "error": str(error),
                "raw_text": raw_text,
            }
        )


def reset_debug_file(output_dir: Path) -> None:
    path = output_dir / "debug_errors.csv"
    if path.exists():
        path.unlink()


def ollama_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. "
            "Start Ollama, then run the script again."
        ) from exc


def gemma_raw_prompt(prompt: str) -> str:
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def clean_model_output(text: str) -> str:
    text = re.sub(r"<pad>|<bos>|<eos>|<unused\d+>", "", text)
    text = re.sub(r"<start_of_turn>|<end_of_turn>", "", text)
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    text = re.sub(r"<[^>]{1,60}>", "", text)
    return text.strip()


def is_usable(text: str) -> bool:
    cleaned = re.sub(r"[\s\"'`.,;:!?{}\[\]()|/\\_-]+", "", text)
    return bool(cleaned)


def generate_text(prompt: str) -> str:
    raw_text = ollama_generate(gemma_raw_prompt(prompt))
    cleaned = clean_model_output(raw_text)
    if is_usable(cleaned):
        return cleaned
    raise RuntimeError(f"{MODEL} did not return usable text: {raw_text[:120]!r}")


def generate_text_with_retries(
    prompt: str,
    item_id: str,
    stage: str,
    output_dir: Path,
) -> str:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return generate_text(prompt)
        except Exception as exc:
            print(f"id={item_id} {stage} attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}")
            append_debug_error(output_dir, item_id, stage, attempt, exc)
    raise RuntimeError(f"id={item_id} {stage} failed after {MAX_ATTEMPTS} attempts.")


def ollama_generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "raw": True,
    }
    data = ollama_post("/api/generate", payload)
    return str(data.get("response", ""))


def build_assistant_prompt(query: str, emotion: str | None, confidence: str) -> str:
    lines = [
        "You are an emotion-aware desktop AI assistant.",
        "Be concise, practical, and helpful.",
    ]
    if emotion:
        confidence_text = f" with confidence {confidence}" if confidence else ""
        lines.append(
            f"The user's detected facial emotion is {emotion}{confidence_text}. "
            "Use it only as soft context, not certainty."
        )
    else:
        lines.append("No facial emotion context is available.")

    lines.extend(
        [
            "",
            f"User query: {query}",
            "Assistant response:",
        ]
    )
    return "\n".join(lines)


def build_evaluation_prompt(row: dict[str, Any]) -> str:
    return f"""
Evaluate two assistant responses for the same user query.

User query:
{row["query"]}

Detected facial emotion:
{row["emotion"]} with confidence {row["confidence"]}

Baseline response, generated without emotion context:
{row["baseline_response"]}

Emotion-aware response, generated with emotion as soft context:
{row["emotion_aware_response"]}

Score each response from 1 to 5:
- helpfulness: does it answer the user's request?
- emotional_appropriateness: does the tone fit the apparent user state?
- empathy_supportiveness: is it patient, reassuring, and supportive when useful?
- overreaction_risk: 1 means no overreaction, 5 means it assumes too much from the emotion.

Return only JSON. Use exactly this structure:
{{
  "baseline": {{
    "helpfulness": 1,
    "emotional_appropriateness": 1,
    "empathy_supportiveness": 1,
    "overreaction_risk": 1
  }},
  "emotion_aware": {{
    "helpfulness": 1,
    "emotional_appropriateness": 1,
    "empathy_supportiveness": 1,
    "overreaction_risk": 1
  }},
  "overall_preference": "baseline or emotion_aware or tie"
}}
""".strip()


def parse_json_object(text: str) -> dict[str, Any]:
    text = first_json_object_text(text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            fixed = re.sub(r"([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', text)
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            parsed = json.loads(fixed)

    if not isinstance(parsed, dict):
        raise ValueError("Evaluation response was not a JSON object.")
    return parsed


def first_json_object_text(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("JSON object was started but not closed.")


def evaluate_pair(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        evaluation_text = ""
        try:
            evaluation_text = generate_text(build_evaluation_prompt(row))
            return normalize_evaluation(parse_json_object(evaluation_text))
        except Exception as exc:
            print(
                f"id={row['id']} evaluation attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}"
            )
            append_debug_error(
                output_dir,
                str(row["id"]),
                "evaluation",
                attempt,
                exc,
                evaluation_text,
            )

            if not evaluation_text:
                continue

            try:
                repair_prompt = (
                    "Convert this evaluator output to valid JSON using the requested schema. "
                    "Return only JSON.\n\n"
                    f"{evaluation_text}"
                )
                repaired_text = generate_text(repair_prompt)
                return normalize_evaluation(parse_json_object(repaired_text))
            except Exception as repair_exc:
                print(
                    f"id={row['id']} repair attempt {attempt}/{MAX_ATTEMPTS} failed: "
                    f"{repair_exc}"
                )
                append_debug_error(
                    output_dir,
                    str(row["id"]),
                    "evaluation_repair",
                    attempt,
                    repair_exc,
                    evaluation_text,
                )

    raise RuntimeError(f"id={row['id']} evaluation failed after {MAX_ATTEMPTS} attempts.")


def score(value: Any) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = 3
    return max(1, min(5, number))


def normalize_scores(raw: Any) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "helpfulness": score(raw.get("helpfulness")),
        "emotional_appropriateness": score(raw.get("emotional_appropriateness")),
        "empathy_supportiveness": score(raw.get("empathy_supportiveness")),
        "overreaction_risk": score(raw.get("overreaction_risk")),
    }


def normalize_evaluation(raw: dict[str, Any]) -> dict[str, Any]:
    preference = str(raw.get("overall_preference", "tie")).strip().lower().replace("-", "_")
    if preference not in {"baseline", "emotion_aware", "tie"}:
        preference = "tie"
    return {
        "baseline": normalize_scores(raw.get("baseline")),
        "emotion_aware": normalize_scores(
            raw.get("emotion_aware")
            or raw.get("emotion-aware")
            or raw.get("emotionAware")
            or raw.get("Emotion-aware")
        ),
        "overall_preference": preference,
    }


def metric_winner(baseline_score: int, emotion_score: int, lower_is_better: bool) -> str:
    if baseline_score == emotion_score:
        return "tie"
    if lower_is_better:
        return "emotion_aware" if emotion_score < baseline_score else "baseline"
    return "emotion_aware" if emotion_score > baseline_score else "baseline"


def format_metric_cell(row: dict[str, Any], metric: str, lower_is_better: bool) -> str:
    baseline = row[f"baseline_{metric}"]
    emotion = row[f"emotion_aware_{metric}"]
    winner = row[f"{metric}_winner"]
    if winner == "tie":
        return f"tie ({baseline} vs {emotion})"
    if lower_is_better:
        return f"{winner} lower risk ({baseline} baseline vs {emotion} emotion-aware)"
    return f"{winner} better ({baseline} baseline vs {emotion} emotion-aware)"


def run_interaction(
    item: dict[str, str],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline_prompt = build_assistant_prompt(item["query"], emotion=None, confidence="")
    emotion_prompt = build_assistant_prompt(
        item["query"],
        emotion=item["emotion"],
        confidence=item["confidence"],
    )

    baseline_response = generate_text_with_retries(
        baseline_prompt,
        item["id"],
        "baseline_response",
        output_dir,
    )
    emotion_response = generate_text_with_retries(
        emotion_prompt,
        item["id"],
        "emotion_aware_response",
        output_dir,
    )

    response_row = {
        "id": item["id"],
        "query": item["query"],
        "emotion": item["emotion"],
        "confidence": item["confidence"],
        "baseline_response": baseline_response,
        "emotion_aware_response": emotion_response,
    }

    evaluation = evaluate_pair(response_row, output_dir)
    paired_row = {
        "id": item["id"],
        "query": item["query"],
        "emotion": item["emotion"],
        "confidence": item["confidence"],
        "overall_preference": evaluation["overall_preference"],
    }

    for metric, lower_is_better in METRICS:
        baseline_score = evaluation["baseline"][metric]
        emotion_score = evaluation["emotion_aware"][metric]
        paired_row[f"baseline_{metric}"] = baseline_score
        paired_row[f"emotion_aware_{metric}"] = emotion_score
        paired_row[f"{metric}_winner"] = metric_winner(
            baseline_score,
            emotion_score,
            lower_is_better,
        )

    analysis_row = {
        "query": item["query"],
        "emotion": item["emotion"],
        "Helpfulness": format_metric_cell(paired_row, "helpfulness", False),
        "Emotional appropriateness": format_metric_cell(
            paired_row,
            "emotional_appropriateness",
            False,
        ),
        "Empathy/supportiveness": format_metric_cell(
            paired_row,
            "empathy_supportiveness",
            False,
        ),
        "Overreaction risk": format_metric_cell(paired_row, "overreaction_risk", True),
        "Overall preference": paired_row["overall_preference"],
    }

    return response_row, paired_row, analysis_row


def main() -> None:
    args = parse_args()
    interactions = read_interactions(args.input, args.limit)
    reset_debug_file(args.output_dir)

    responses = []
    paired_scores = []
    analysis_rows = []

    for index, item in enumerate(interactions, start=1):
        response_row, paired_row, analysis_row = run_interaction(item, args.output_dir)
        responses.append(response_row)
        paired_scores.append(paired_row)
        analysis_rows.append(analysis_row)
        write_result_files(args.output_dir, responses, paired_scores, analysis_rows)
        print(f"Finished {index}/{len(interactions)}: id={item['id']}")

    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from None
