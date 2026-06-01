# Emotion Context Ablation Evaluation

This folder contains the evaluation flow for the dissertation experiment.

The experiment compares two responses for each query:

- `baseline`: generated without facial emotion context
- `emotion_aware`: generated with the detected emotion added as soft context

The model is fixed to:

```text
gemma4:e4b
```

## Input

The input file is:

```text
evaluation/interactions.csv
```

It contains 50 rows with this structure:

```csv
id,query,emotion,confidence
1,"I still do not understand this part. Can you explain it again?",Sad,0.82
```

The emotion labels match the FER classifier:

```text
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
```

## Run a Small Test

Use `--limit 1` or `--limit 2` while testing:

```powershell
python evaluation\run_emotion_ablation.py --limit 1
```

## Run the Full Evaluation

```powershell
python evaluation\run_emotion_ablation.py
```

The script writes:

```text
evaluation/results/responses.csv
evaluation/results/scores_paired.csv
evaluation/results/analysis_table.csv
```

## Build Statistics

After the evaluation script finishes, run:

```powershell
python evaluation\build_statistics_tables.py
```

The statistics script reads:

```text
evaluation/results/scores_paired.csv
```

and writes:

```text
evaluation/results/statistics/metric_summary.csv
evaluation/results/statistics/overall_preference.csv
evaluation/results/statistics/preference_by_emotion.csv
evaluation/results/statistics/statistics_report.md
```

The most useful file for the dissertation is:

```text
evaluation/results/statistics/statistics_report.md
```
