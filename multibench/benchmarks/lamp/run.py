"""LaMP-1..7 evaluation via OpenAI-compatible endpoint (parallel).

Reimplementation of upstream `LaMP/evaluate_vllm.py` on the shared
multibench.client.LLMClient. Metrics are identical to LaMP's official
harness (F1/Acc for LaMP-1,2; MAE/RMSE for LaMP-3; BLEU/ROUGE/METEOR for
LaMP-4..7), computed with the `evaluate` library.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import evaluate as hf_evaluate
from tqdm import tqdm

from ...args import add_common_llm_args
from ...client import client_from_args
from ...utils import atomic_write_json, benchmark_data_dir, ensure_dir


VALID_TASKS = [f"LaMP-{i}" for i in range(1, 8)]


# ---------- metrics (identical to upstream) ----------

def compute_classification_metrics(preds, labels, all_labels):
    f1_metric = hf_evaluate.load("f1")
    accuracy_metric = hf_evaluate.load("accuracy")

    def to_idx(x):
        try:
            return all_labels.index(x.strip())
        except ValueError:
            return -1

    pred_ids = [to_idx(p) for p in preds]
    label_ids = [to_idx(l) for l in labels]
    result_acc = accuracy_metric.compute(predictions=pred_ids, references=label_ids)
    result_f1 = f1_metric.compute(
        predictions=pred_ids, references=label_ids,
        labels=list(range(len(all_labels))), average="macro"
    )
    return {"accuracy": result_acc["accuracy"], "f1": result_f1["f1"]}


def compute_regression_metrics(preds, labels):
    mse_metric = hf_evaluate.load("mse")
    mae_metric = hf_evaluate.load("mae")

    def to_float(x, fallback):
        try:
            return float(x.strip())
        except (ValueError, AttributeError):
            y = float(fallback)
            return 1.0 if abs(1 - y) > abs(5 - y) else 5.0

    pred_vals = [to_float(p, l) for p, l in zip(preds, labels)]
    label_vals = [to_float(l, l) for l in labels]
    result_mae = mae_metric.compute(predictions=pred_vals, references=label_vals)
    result_rmse = mse_metric.compute(predictions=pred_vals, references=label_vals, squared=False)
    return {"MAE": result_mae["mae"], "RMSE": result_rmse["mse"]}


def compute_generation_metrics(preds, labels):
    bleu_metric = hf_evaluate.load("sacrebleu")
    rouge_metric = hf_evaluate.load("rouge")
    meteor_metric = hf_evaluate.load("meteor")

    preds_clean = [p.strip() for p in preds]
    labels_clean = [[l.strip()] for l in labels]
    result_bleu = bleu_metric.compute(predictions=preds_clean, references=labels_clean)
    result_rouge = rouge_metric.compute(predictions=preds_clean, references=labels_clean)
    result_meteor = meteor_metric.compute(predictions=preds_clean, references=labels_clean)
    return {
        "bleu": result_bleu["score"],
        "rouge-1": result_rouge["rouge1"],
        "rouge-2": result_rouge["rouge2"],
        "rouge-L": result_rouge["rougeL"],
        "meteor": result_meteor["meteor"],
    }


# ---------- prompt assembly ----------

def _build_prompts(questions, task, use_profile, num_retrieved, retriever, is_ranked):
    if not use_profile:
        return [q["input"] for q in questions]

    from transformers import AutoTokenizer
    from .prompts.prompts import create_prompt_generator
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_generator, _ = create_prompt_generator(
        num_retrieved, retriever, is_ranked, max_length=2048, tokenizer=tokenizer
    )
    prompts = []
    for q in tqdm(questions, desc="Preparing profile-augmented prompts"):
        prompts.append(prompt_generator(q["input"], q["profile"], task))
    return prompts


# ---------- CLI ----------

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LaMP (1..7) evaluation, parallel.")
    add_common_llm_args(ap)
    ap.add_argument("--task", required=True, choices=VALID_TASKS)
    ap.add_argument("--validation-data", "--validation_data", dest="validation_data",
                    default=None,
                    help="Path to dev_questions.json (default: data/lamp/<Task>/dev_questions.json).")
    ap.add_argument("--golds-json", "--golds_json", dest="golds_json", default=None,
                    help="Path to dev_outputs.json (default: data/lamp/<Task>/dev_outputs.json).")
    ap.add_argument("--use-profile", "--use_profile", dest="use_profile", action="store_true")
    ap.add_argument("--num-retrieved", "--num_retrieved", dest="num_retrieved",
                    type=int, default=3)
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "contriever", "recency", "random"])
    ap.add_argument("--is-ranked", "--is_ranked", dest="is_ranked", action="store_true")
    return ap


def _default_paths(task: str) -> tuple[Path, Path]:
    root = benchmark_data_dir("lamp")
    # Upstream directories are LaMP_1, LaMP_2, ...  (not LaMP-1)
    sub = task.replace("-", "_")
    return root / sub / "dev_questions.json", root / sub / "dev_outputs.json"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = ensure_dir(args.output_dir)

    val_path = Path(args.validation_data) if args.validation_data else _default_paths(args.task)[0]
    gold_path = Path(args.golds_json) if args.golds_json else _default_paths(args.task)[1]

    with val_path.open() as f:
        questions = json.load(f)
    with gold_path.open() as f:
        golds_data = json.load(f)
    gold_map = {g["id"]: g["output"] for g in golds_data["golds"]}

    if args.max_items:
        questions = questions[: args.max_items]

    prompts = _build_prompts(
        questions, args.task, args.use_profile, args.num_retrieved,
        args.retriever, args.is_ranked,
    )
    items = [
        {"id": q["id"], "prompt": p, "gold": gold_map.get(q["id"], "")}
        for q, p in zip(questions, prompts)
    ]

    client = client_from_args(args)
    preds = client.chat_batch(
        items,
        build_messages=lambda it: it["prompt"],
        workers=args.workers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        desc=f"LaMP {args.task}",
    )
    # normalise exceptions to empty strings
    preds = ["" if isinstance(p, Exception) else p for p in preds]

    from .data_utils.labels import get_all_labels
    all_labels = get_all_labels(args.task)

    if args.task in {"LaMP-1", "LaMP-2"}:
        results = compute_classification_metrics(preds, [it["gold"] for it in items], all_labels)
    elif args.task == "LaMP-3":
        results = compute_regression_metrics(preds, [it["gold"] for it in items])
    else:
        results = compute_generation_metrics(preds, [it["gold"] for it in items])

    preds_json = {"task": args.task.replace("-", "_"),
                  "golds": [{"id": it["id"], "output": p} for it, p in zip(items, preds)]}
    atomic_write_json(out_dir / "predictions.json", preds_json)

    results["model"] = args.model
    results["task"] = args.task
    results["num_items"] = len(items)
    atomic_write_json(out_dir / "results.json", results)

    print(f"\n=== {args.task} — {args.model} ===")
    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
