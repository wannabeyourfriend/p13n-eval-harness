"""LaMP-QA — personalized long-form QA with rubric-based LLM-judge scoring.

Upstream: https://github.com/LaMP-Benchmark/LaMP-QA  (arXiv 2506.00137)

Protocol
--------
For each item in a category (3 categories × ~800–900 items):
  1. Retrieve top-k user-history snippets from `profile` via BM25 on the question.
  2. Generate an answer conditioned on retrieved context.
  3. For each rubric_aspect, ask the judge to score the answer 0/1/2.
  4. Per-item score = mean(per_aspect) / 2.  Category score = mean(per-item).

Uses multibench.client.LLMClient.chat_batch for both generation and judging so
concurrency is controlled by a single --workers flag.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ...args import add_common_llm_args
from ...client import LLMClient, client_from_args
from ...utils import atomic_write_json, benchmark_data_dir, ensure_dir


CATEGORIES = ["Art_and_Entertainment", "Lifestyle_and_Personal_Development", "Society_and_Culture"]
SPLIT_DEFAULT = "test"

JUDGE_SYS = (
    "You are a strict evaluator. Given a user's question, a generated answer, "
    "and one evaluation aspect (with rationale), rate how well the answer "
    "addresses that aspect on a 0-2 scale:\n"
    "  0 = not addressed / wrong\n"
    "  1 = partially addressed\n"
    "  2 = fully and clearly addressed\n"
    "Respond with ONLY a single digit: 0, 1, or 2."
)


def _tokenize(t: str) -> list[str]:
    return re.findall(r"\w+", t.lower())


def _retrieve(query: str, profile: list[dict], k: int) -> list[dict]:
    if not profile:
        return []
    if len(profile) <= k:
        return profile
    corpus = [_tokenize(p["text"]) for p in profile]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    idx = sorted(range(len(profile)), key=lambda i: -scores[i])[:k]
    return [profile[i] for i in idx]


def _gen_prompt(item: dict, topk: int) -> list[dict]:
    retrieved = _retrieve(item["question"], item["profile"], topk)
    ctx = "\n\n".join(f"[User history {i+1}]: {p['text'][:800]}" for i, p in enumerate(retrieved))
    user = (
        "You are a helpful assistant answering a user's question. "
        "Use the user's past content below as context to personalize your answer.\n\n"
        f"{ctx}\n\n"
        f"Question: {item['question']}\n\n"
        "Provide a thorough, personalized answer:"
    )
    return [{"role": "user", "content": user}]


def _judge_messages(question: str, answer: str, aspect: dict) -> list[dict]:
    body = (
        f"Question: {question}\n\n"
        f"Generated answer:\n{answer}\n\n"
        f"Evaluation aspect: {aspect['aspect']}\n"
        f"Why this aspect matters: {aspect['reason']}\n\n"
        "Score (0, 1, or 2):"
    )
    return [{"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": body}]


def _parse_score(s: str) -> int:
    m = re.search(r"[012]", s or "")
    return int(m.group()) if m else 0


def _data_path(category: str, split: str) -> Path:
    return benchmark_data_dir("lamp") / "LaMP-QA" / "data" / category / split / f"{split}.json"


def _run_category(
    *,
    category: str,
    split: str,
    sim_client: LLMClient,
    judge_client: LLMClient,
    topk: int,
    workers: int,
    out_dir: Path,
    limit: int | None,
    max_tokens: int,
) -> dict:
    data_path = _data_path(category, split)
    with data_path.open() as f:
        data = json.load(f)
    if limit:
        data = data[:limit]

    ensure_dir(out_dir)
    preds_path = out_dir / "predictions.json"
    scores_path = out_dir / "scores.json"

    # ---- Generate ----
    if preds_path.exists():
        with preds_path.open() as f:
            cached = {p["id"]: p["output"] for p in json.load(f)}
    else:
        cached = {}
    todo = [x for x in data if x["id"] not in cached]
    print(f"[{category}] generating {len(todo)}/{len(data)} answers")
    if todo:
        answers = sim_client.chat_batch(
            todo,
            build_messages=lambda item: _gen_prompt(item, topk),
            workers=workers,
            max_tokens=max_tokens,
            temperature=0.0,
            desc=f"gen {category}",
        )
        for item, ans in zip(todo, answers):
            # chat_batch stores exceptions at failed indices — coerce so the
            # downstream json dump works and the judge pass gets a string.
            cached[item["id"]] = "" if isinstance(ans, BaseException) else str(ans or "")
    preds_list = [{"id": x["id"], "output": cached.get(x["id"], "")} for x in data]
    atomic_write_json(preds_path, preds_list)

    # ---- Judge per aspect ----
    if scores_path.exists():
        with scores_path.open() as f:
            scores: dict = json.load(f)
    else:
        scores = {}
    jobs: list[tuple] = []
    for x in data:
        if x["id"] in scores:
            continue
        ans = cached.get(x["id"], "")
        for idx, asp in enumerate(x["rubric_aspects"]):
            jobs.append((x["id"], idx, x["question"], ans, asp))
    print(f"[{category}] judging {len(jobs)} aspects")
    if jobs:
        grades = judge_client.chat_batch(
            jobs,
            build_messages=lambda job: _judge_messages(job[2], job[3], job[4]),
            workers=workers,
            max_tokens=4,
            temperature=0.0,
            desc=f"judge {category}",
        )
        per_item: dict[str, dict[int, int]] = {}
        for (iid, idx, *_), g in zip(jobs, grades):
            txt = "" if isinstance(g, BaseException) else str(g or "")
            per_item.setdefault(iid, {})[idx] = _parse_score(txt)
        for x in data:
            if x["id"] in scores:
                continue
            asp_scores = per_item.get(x["id"], {})
            n = len(x["rubric_aspects"])
            vals = [asp_scores.get(i, 0) for i in range(n)]
            scores[x["id"]] = {"per_aspect": vals, "mean": (sum(vals) / n / 2.0) if n else 0.0}
    atomic_write_json(scores_path, scores)

    means = [s["mean"] for s in scores.values()]
    avg = sum(means) / len(means) if means else 0.0
    print(f"[{category}] mean normalized score = {avg:.4f} over {len(means)} items")
    return {"category": category, "n_items": len(means), "mean_score": avg}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LaMP-QA rubric-based evaluation.")
    add_common_llm_args(ap)
    ap.add_argument("--judge-api-base", "--judge_api_base", default=None)
    ap.add_argument("--judge-model", "--judge_model", default=None)
    ap.add_argument("--topk", type=int, default=5, help="BM25 top-k profile snippets")
    ap.add_argument("--split", default=SPLIT_DEFAULT, choices=["train", "validation", "test"])
    ap.add_argument("--categories", nargs="+", default=CATEGORIES, choices=CATEGORIES)
    ap.add_argument("--gen-max-tokens", type=int, default=800,
                    help="Max tokens for answer generation (default 800).")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_root = ensure_dir(args.output_dir)

    sim_client = client_from_args(args)
    if args.judge_api_base or args.judge_model:
        judge_client = LLMClient(
            model_name=args.judge_model or args.model,
            api_base=args.judge_api_base or args.api_base,
            api_key=args.api_key,
            default_temperature=0.0,
            default_max_tokens=8,
            strip_think=True,
        )
    else:
        judge_client = sim_client

    summary = []
    for cat in args.categories:
        cat_dir = out_root / cat
        s = _run_category(
            category=cat, split=args.split,
            sim_client=sim_client, judge_client=judge_client,
            topk=args.topk, workers=args.workers,
            out_dir=cat_dir, limit=args.max_items,
            max_tokens=args.gen_max_tokens,
        )
        summary.append(s)

    overall = sum(s["mean_score"] for s in summary) / len(summary) if summary else 0.0
    atomic_write_json(out_root / "summary.json", {
        "model": args.model,
        "judge": args.judge_model or args.model,
        "split": args.split,
        "topk": args.topk,
        "per_category": summary,
        "overall_mean_score": overall,
    })
    print(f"\n[lampqa] overall mean score = {overall:.4f}  (avg of {len(summary)} categories)")
    for s in summary:
        print(f"  {s['category']:<40s} n={s['n_items']:<5d} mean={s['mean_score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
