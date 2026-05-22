[![Live Demo](https://img.shields.io/badge/Live%20Demo-sync--video--label.vercel.app-brightgreen?logo=vercel)](https://sync-video-label.vercel.app)
[![Annotation Software](https://img.shields.io/badge/Annotation%20Software-sync--video--label-blue?logo=github)](https://github.com/wangyz1999/sync-video-label)
[![Project Page](https://img.shields.io/badge/Project%20Page-GameplayQA-blue?logo=github)](https://hats-ict.github.io/gameplayqa/)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-ffcc33?logo=huggingface)](https://huggingface.co/datasets/wangyz1999/GameplayQA)
[![Paper](https://img.shields.io/badge/Paper-arXiv-FF0066?logo=arxiv)](https://arxiv.org/abs/2603.24329)
[![Demo Video](https://img.shields.io/badge/Demo%20Video-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=PKedELJ4XT0)
![ACL 2026](https://img.shields.io/badge/ACL-2026-purple)
![License](https://img.shields.io/badge/License-MIT-green)

# GameplayQA

GameplayQA is a benchmark for decision-dense, POV-synced video question answering in 3D gameplay environments. It provides multiple-choice questions grounded in short gameplay segments, including single-view temporal reasoning and synchronized multi-view reasoning.

The released dataset is on [Hugging Face (`wangyz1999/GameplayQA`)](https://huggingface.co/datasets/wangyz1999/GameplayQA). This repository contains:

- `data/qa.csv`: the main GameplayQA benchmark.
- `data/qa_generalization.csv`: the generalization benchmark.
- `data/annotation/`: source videos and annotation assets referenced by the CSV files.

The commands below run the standard benchmark workflow. Full command-line options are documented in [args.md](args.md).

## Quick Start

We use [`uv`](https://docs.astral.sh/uv) for package management.

This repository requires [FFmpeg](https://ffmpeg.org/) installed and available on your `PATH` for video preprocessing.

```bash
uv sync                           # intall dependencies
uv run download_data.py           # download data from huggingface
```

Rename [`.env.example`](.env.example) to `.env`, then edit `.env` and set the API keys for the model providers you plan to use (you only need the keys you will actually call).

## Standard Workflow

### Dataset split (`--split`)

- **`qa`**: main split of gameplay videos, questions in `data/qa.csv` (default if you omit `--split`).
- **`generalization`**: generalization split, questions in `data/qa_generalization.csv`.

### Parallel workers (`--workers`)

- **`run_benchmark.py`**: `-w` / `--workers` sets how many questions are benchmarked in parallel (default **5**). Increase it if your API rate limits allow; decrease it if you see throttling or want a lighter load.
- **With `--judge`**: `--judge-workers` controls parallel judging workers (default **5**).
- **`process_data.py`**: `--workers` controls parallel frame extraction (default **4**); see [args.md](args.md).

Prepare the benchmark split. This crops each question's video segment and builds the frame cache used by frame-based vision models.

```bash
uv run process_data.py --split qa
```

Run a model on the prepared split.

```bash
uv run run_benchmark.py --split qa --model gpt-5-mini --judge
```

Example with higher parallelism (tune to your API limits):

```bash
uv run run_benchmark.py --split qa --model gpt-5-mini --judge -workers 8 --judge-workers 8
```

If you run the benchmark without `--judge`, judge the raw model responses separately.

```bash
uv run llm_as_a_judge.py --split qa --model gpt-5-mini
```

Evaluate and plot the judged results.

```bash
uv run evaluate_results.py --split qa
uv run plot_results.py --split qa
```

The same workflow can be run on the generalization split by replacing `qa` with `generalization`.

## Outputs

The default output paths are stable, so each pipeline step can automatically find the previous step's result:

- Raw benchmark results: `results/raw/<split>/<model>.jsonl`
- Judged results: `results/judged/<split>/<model>_judged.jsonl`
- Evaluation tables: `results/evaluation/<split>/`
- Plots: `results/plots/<split>/`

## Ablation Studies

Run an ablation benchmark, judge it, and include it in evaluation:

```bash
uv run ablation_benchmark.py --split qa --model gpt-5-mini --ablation no_video
uv run llm_as_a_judge.py --split qa --model gpt-5-mini --ablation no_video
uv run evaluate_ablation.py --split qa
```

Available ablations are `no_video`, `random_single_frame`, and `shuffled_frames`.

## Citation

If you use GameplayQA, please cite the paper:

```bibtex
@article{wang2026gameplayqa,
  title   = {GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents},
  author  = {Wang, Yunzhe and Xu, Runhui and Zheng, Kexin and Zhang, Tianyi and Kogundi, Jayavibhav Niranjan and Hans, Soham and Ustun, Volkan},
  year    = {2026},
  eprint  = {2603.24329},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url     = {https://arxiv.org/abs/2603.24329}
}
```
