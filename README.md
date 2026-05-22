![Next.js](https://img.shields.io/badge/Next.js-16.1-black?logo=next.js)
![React](https://img.shields.io/badge/React-19-61dafb?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6?logo=typescript)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-4-06b6d4?logo=tailwindcss)
![License](https://img.shields.io/badge/License-MIT-green)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-sync--video--label.vercel.app-brightgreen?logo=vercel)](https://sync-video-label.vercel.app)
[![Project Page](https://img.shields.io/badge/Project%20Page-GameplayQA-blue?logo=github)](https://hats-ict.github.io/gameplayqa/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-FF0066?logo=arxiv)](https://arxiv.org/abs/2603.24329)
[![Demo Video](https://img.shields.io/badge/Demo%20Video-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=PKedELJ4XT0)
![ACL 2026](https://img.shields.io/badge/ACL-2026-purple)

# GameplayQA

GameplayQA is a benchmark for decision-dense, POV-synced video question answering in 3D gameplay environments. It provides multiple-choice questions grounded in short gameplay segments, including single-view temporal reasoning and synchronized multi-view reasoning.

The released dataset contains:

- `data/qa.csv`: the main GameplayQA benchmark.
- `data/qa_generalization.csv`: the generalization benchmark.
- `data/annotation/`: source videos and annotation assets referenced by the CSV files.

The commands below run the standard benchmark workflow. Full command-line options are documented in [args.md](args.md).

## Quick Start

We use [`uv`](https://docs.astral.sh/uv) for package management.

This repository requires [FFmpeg](https://ffmpeg.org/) installed and available on your `PATH` for video preprocessing.

```bash
uv sync
python download_data.py
```

Rename [`.env.example`](.env.example) to `.env`, then edit `.env` and set the API keys for the model providers you plan to use (you only need the keys you will actually call).

## Standard Workflow

Prepare the benchmark split. This crops each question's video segment and builds the frame cache used by frame-based vision models.

```bash
python process_data.py --split qa
```

Run a model on the prepared split.

```bash
python run_benchmark.py --split qa --model gpt-5-mini --judge
```

If you run the benchmark without `--judge`, judge the raw model responses separately.

```bash
python llm_as_a_judge.py --split qa --model gpt-5-mini
```

Evaluate and plot the judged results.

```bash
python evaluate_results.py --split qa
python plot_results.py --split qa
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
python ablation_benchmark.py --split qa --model gpt-5-mini --ablation no_video
python llm_as_a_judge.py --split qa --model gpt-5-mini --ablation no_video
python evaluate_ablation.py --split qa
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
