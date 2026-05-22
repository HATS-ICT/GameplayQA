# GameplayQA Script Arguments

All scripts are intended to run from the repository root as `uv run <script>.py`.

## Output Convention

Default outputs use stable paths so later steps can locate the latest run automatically:

- Raw benchmark results: `results/raw/<split>/<model>.jsonl`
- Judged benchmark results: `results/judged/<split>/<model>_judged.jsonl`
- Raw ablation results: `results/ablation/<split>/<ablation>_<model>.jsonl`
- Evaluation reports: `results/evaluation/<split>/evaluation_results.*`
- Plots: `results/plots/<split>/`

Use `--timestamped` on benchmark scripts only when you intentionally want an archival run file.

## `download_data.py`

Downloads the GameplayQA dataset from Hugging Face.

| Argument | Default | Description |
|---|---:|---|
| `-d`, `--destination` | `data` | Local dataset folder. |

## `process_data.py`

Crops question videos and extracts reusable frame caches in one preprocessing step.

| Argument | Default | Description |
|---|---:|---|
| `--split` | `qa` | Dataset split: `qa` or `generalization`. |
| `--csv` | split CSV | Override the question CSV path. |
| `--data-dir` | `data` | Downloaded dataset directory containing `annotation/`. |
| `--video-dir` | `data/question_videos/<split>` | Output directory for cropped question clips. |
| `--frames-dir` | `data/question_video_frames/<split>` | Output directory for extracted frame caches. |
| `--max-frames` | `32` | Maximum frames per cropped video. |
| `--workers` | `4` | Parallel workers for frame extraction. |
| `--force` | off | Rebuild both cropped videos and frame caches. |
| `--force-crop` | off | Overwrite existing cropped videos. |
| `--force-frames` | off | Re-extract existing frame folders. |
| `--skip-crop` | off | Skip the cropping phase. |
| `--skip-frames` | off | Skip frame extraction. |
| `--limit` | all | Process only the first N questions. |

## `crop_question_video.py`

Lower-level command for cropping only.

| Argument | Default | Description |
|---|---:|---|
| `--split` | `qa` | Dataset split: `qa` or `generalization`. |
| `--csv` | split CSV | Override question CSV path. |
| `--data-dir` | `data` | Downloaded dataset directory. |
| `--output-dir` | `data/question_videos/<split>` | Directory for cropped clips. |
| `--force` | off | Overwrite existing cropped clips. |
| `--limit` | all | Process only the first N questions. |

## `extract_video_frames.py`

Lower-level command for frame cache extraction only.

| Argument | Default | Description |
|---|---:|---|
| `--split` | `qa` | Dataset split: `qa` or `generalization`. |
| `--csv` | split CSV | Override question CSV path. |
| `--video-dir` | `data/question_videos/<split>` | Directory containing cropped videos. |
| `--output-dir` | `data/question_video_frames/<split>` | Directory for frame caches. |
| `--max-frames` | `32` | Maximum frames per cropped video. |
| `--workers` | `4` | Parallel extraction workers. |
| `--force` | off | Re-extract existing frame folders. |
| `--limit` | all | Process only the first N videos. |

## `run_benchmark.py`

Runs a multimodal model on the selected GameplayQA split.

| Argument | Default | Description |
|---|---:|---|
| `--list` | off | List available model keys. |
| `--model`, `-m` | `gemini-3-flash` | Model key from the registry. |
| `--split` | `qa` | Dataset split: `qa` or `generalization`. |
| `--csv`, `-c` | split CSV | Override question CSV path. |
| `--video-dir`, `-v` | `data/question_videos/<split>` | Directory with cropped videos. |
| `--frames-dir`, `-f` | `data/question_video_frames/<split>` | Directory with extracted frames for frame-based models. Ignored by video-native models. |
| `--output`, `-o` | `results/raw/<split>/<model>.jsonl` | Output JSONL path. |
| `--timestamped` | off | Append `YYYYMMDD_HHMMSS` to the default output filename. |
| `--workers`, `-w` | `5` | Parallel API workers. |
| `--limit`, `-n` | all | Benchmark only the first N questions. |
| `--resume`, `-r` | off | Resume an output JSONL. Use `--resume` alone for the default stable output, or pass a path. |
| `--seed` | `13` | Deterministic answer-option shuffle seed. |
| `--judge` | off | Run judging automatically after the benchmark finishes. |
| `--judge-model`, `-j` | `gpt-5-mini` | OpenAI model used for automatic judging. |
| `--judge-workers` | `5` | Parallel workers for automatic judging. |
| `--judge-output` | automatic | Judged JSONL output path for `--judge`. |

## `llm_as_a_judge.py`

Extracts the selected option from raw model responses and marks correctness.

| Argument | Default | Description |
|---|---:|---|
| `input` | automatic | Raw benchmark JSONL. If omitted, resolved from `--split`, `--model`, and optional `--ablation`. |
| `--model`, `-m` | `gemini-3-flash` | Model key used for automatic input/output paths. |
| `--split` | `qa` | Dataset split used for automatic paths. |
| `--ablation` | none | Judge an ablation result: `no_video`, `random_single_frame`, or `shuffled_frames`. |
| `--output`, `-o` | automatic | Judged JSONL path. |
| `--judge-model`, `-j` | `gpt-5-mini` | OpenAI model used as the judge. |
| `--workers`, `-w` | `5` | Parallel judge workers. |
| `--limit`, `-n` | all | Judge only the first N raw results. |

## `evaluate_results.py`

Aggregates judged result files into JSON, CSV, and Markdown reports.

| Argument | Default | Description |
|---|---:|---|
| `results` | automatic | Judged JSONL file(s). If omitted, discovers all `results/judged/<split>/*.jsonl`. |
| `--model`, `-m` | all discovered | Evaluate one model's default judged result. |
| `--split` | `qa` | Dataset split metadata and discovery folder. |
| `--csv` | split CSV | Question CSV metadata path. May be repeated. |
| `--output-json` | `results/evaluation/<split>/evaluation_results.json` | Evaluation JSON output. |
| `--output-csv` | `results/evaluation/<split>/evaluation_results.csv` | Evaluation CSV output. |
| `--output-md` | `results/evaluation/<split>/evaluation_results.md` | Markdown report output. |

## `plot_results.py`

Generates PNG plots from an evaluation JSON file.

| Argument | Default | Description |
|---|---:|---|
| `--split` | `qa` | Dataset split for default input/output paths. |
| `--input`, `-i` | `results/evaluation/<split>/evaluation_results.json` | Evaluation JSON path. |
| `--output-dir`, `-o` | `results/plots/<split>` | Plot output directory. |

## `ablation_benchmark.py`

Runs an ablation benchmark.

| Argument | Default | Description |
|---|---:|---|
| `--list` | off | List available ablation types. |
| `--list-models` | off | List available model keys. |
| `--ablation`, `-a` | `no_video` | One of `no_video`, `random_single_frame`, `shuffled_frames`. |
| `--model`, `-m` | `gpt-5-mini` | Model key from the registry. |
| `--split` | `qa` | Dataset split: `qa` or `generalization`. |
| `--csv`, `-c` | split CSV | Override question CSV path. |
| `--video-dir`, `-v` | `data/question_videos/<split>` | Directory with cropped videos. |
| `--frames-dir`, `-f` | `data/question_video_frames/<split>` | Directory with extracted frames. |
| `--output`, `-o` | `results/ablation/<split>/<ablation>_<model>.jsonl` | Output JSONL path. |
| `--timestamped` | off | Append `YYYYMMDD_HHMMSS` to the default output filename. |
| `--workers`, `-w` | `5` | Parallel API workers. |
| `--limit`, `-n` | all | Benchmark only the first N questions. |
| `--resume`, `-r` | off | Resume an output JSONL. Use `--resume` alone for the default stable output, or pass a path. |
| `--seed` | `13` | Random seed for deterministic option shuffle and frame ablations. |

## `evaluate_ablation.py`

Alias for `evaluate_results.py`. Pass judged ablation files explicitly, or let it discover judged JSONL files under `results/judged/<split>/`.
