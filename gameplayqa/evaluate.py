"""Evaluation and plotting for judged GameplayQA results."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import read_csv_rows
from .judge import load_jsonl
from .progress import console, make_progress


def pct(num: int, den: int) -> float:
    return round(num / den * 100, 2) if den else 0.0


def question_kind(code: str) -> str:
    for key in ["POV-ID", "ORDER", "COUNT", "ABSENT", "INTENT", "TIME", "EXIST", "IDENT"]:
        if key in code:
            return key
    return code.split("-")[-1] if code else "unknown"


def entity_kind(code: str) -> str:
    for key in ["SA", "SS", "OA", "OS", "WO", "WE"]:
        if key in code:
            return key
    return "unknown"


def load_metadata(csv_paths: list[Path]) -> dict[str, dict]:
    metadata = {}
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        for row in read_csv_rows(csv_path):
            metadata[row["id"]] = row
    return metadata


def summarize_group(rows: list[dict], metadata: dict[str, dict], field: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        meta = metadata.get(row["question_id"], {})
        if field == "entity":
            key = entity_kind(meta.get("question_code", ""))
        elif field == "question_type":
            key = question_kind(meta.get("question_code", ""))
        else:
            key = str(meta.get(field, "unknown") or "unknown")
        buckets[key].append(row)
    return {
        key: {"correct": sum(1 for row in vals if row.get("is_correct")), "total": len(vals), "accuracy": pct(sum(1 for row in vals if row.get("is_correct")), len(vals))}
        for key, vals in sorted(buckets.items())
    }


def evaluate_files(result_files: list[Path], csv_paths: list[Path]) -> dict:
    metadata = load_metadata(csv_paths)
    models = []
    with make_progress() as progress:
        task = progress.add_task("Evaluating result files", total=len(result_files))
        for result_file in result_files:
            rows = load_jsonl(result_file)
            if not rows:
                progress.advance(task)
                continue
            model_name = rows[0].get("model_name", result_file.stem)
            model_id = rows[0].get("model_id", model_name)
            correct = sum(1 for row in rows if row.get("is_correct"))
            distractors = Counter(row.get("selected_distractor_type") or "no_answer" for row in rows if not row.get("is_correct"))
            models.append(
                {
                    "model_name": model_name,
                    "model_id": model_id,
                    "result_file": str(result_file),
                    "overall": {"correct": correct, "total": len(rows), "accuracy": pct(correct, len(rows))},
                    "by_level": summarize_group(rows, metadata, "question_level"),
                    "by_task": summarize_group(rows, metadata, "task_name"),
                    "by_game": summarize_group(rows, metadata, "game_name"),
                    "by_entity": summarize_group(rows, metadata, "entity"),
                    "by_question_type": summarize_group(rows, metadata, "question_type"),
                    "error_distractors": dict(sorted(distractors.items())),
                }
            )
            progress.advance(task)
    models.sort(key=lambda row: row["overall"]["accuracy"], reverse=True)
    return {"models": models}


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "model_id", "correct", "total", "accuracy"])
        writer.writeheader()
        for model in data["models"]:
            writer.writerow({"model_name": model["model_name"], "model_id": model["model_id"], **model["overall"]})


def write_markdown(data: dict, path: Path) -> None:
    lines = ["# GameplayQA Evaluation", "", "| Model | Correct | Total | Accuracy |", "|---|---:|---:|---:|"]
    for model in data["models"]:
        overall = model["overall"]
        lines.append(f"| {model['model_name']} | {overall['correct']} | {overall['total']} | {overall['accuracy']:.2f}% |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(result_files: list[Path], csv_paths: list[Path], output_json: Path, output_csv: Path | None, output_md: Path | None) -> None:
    data = evaluate_files(result_files, csv_paths)
    write_json(data, output_json)
    if output_csv:
        write_csv(data, output_csv)
    if output_md:
        write_markdown(data, output_md)
    for model in data["models"]:
        overall = model["overall"]
        console.print(f"{model['model_name']}: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    console.print(f"Saved evaluation JSON to {output_json}")


def plot_results(input_json: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    data = json.loads(input_json.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    models = data.get("models", [])
    if not models:
        console.print("No models to plot")
        return

    plot_groups = ["overall", "by_level", "by_question_type", "by_entity"]
    with make_progress() as progress:
        task = progress.add_task("Generating plots", total=len(plot_groups))
        names = [model["model_name"] for model in models]
        acc = [model["overall"]["accuracy"] for model in models]
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
        ax.bar(names, acc, color="#2f7f7b")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("GameplayQA Overall Accuracy")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output_dir / "overall_accuracy.png", dpi=200)
        plt.close(fig)
        progress.advance(task)

        for group_name in plot_groups[1:]:
            keys = sorted({key for model in models for key in model[group_name]})
            if keys:
                fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.8), 5))
                width = 0.8 / max(1, len(models))
                positions = list(range(len(keys)))
                for idx, model in enumerate(models):
                    vals = [model[group_name].get(key, {}).get("accuracy", 0) for key in keys]
                    offset = (idx - (len(models) - 1) / 2) * width
                    ax.bar([p + offset for p in positions], vals, width=width, label=model["model_name"])
                ax.set_xticks(positions)
                ax.set_xticklabels(keys, rotation=35)
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(group_name.replace("_", " ").title())
                ax.set_ylim(0, 100)
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(output_dir / f"{group_name}.png", dpi=200)
                plt.close(fig)
            progress.advance(task)
    console.print(f"Saved plots to {output_dir}")
