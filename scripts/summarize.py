#!/usr/bin/env python3
"""Summarize PrefEval + BigTom + Sotopia results for one or more models.

Walks the per-model results directory written by eval_one_model.sh and prints:

    === <model> ===
    PrefEval
      travel        <avg>%  (N topics)
      shop          <avg>%
      lifestyle     <avg>%
      entertain     <avg>%
      pet           <avg>%
      education     <avg>%
      professional  <avg>%
      Overall       <avg>%  (all 20 topics, unweighted topic average)

    BigTom
      true_belief   <acc>%
      false_belief  <acc>%
      Overall       <acc>%

    Sotopia
      <per-dimension avg>
      Overall (avg across 7 dims) <score>

The per-topic prefeval accuracy is read from the `*_summary.json` files written
alongside each mcq_results topic dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


PREFEVAL_GROUP_PREFIX = {
    "travel":       "travel",
    "shop":         "shop",
    "lifestyle":    "lifestyle",
    "entertain":    "entertain",
    "pet":          "pet",
    "education":    "education",
    "professional": "professional",
}


def _group_for(topic: str) -> str:
    for prefix, group in PREFEVAL_GROUP_PREFIX.items():
        if topic.startswith(prefix + "_") or topic == prefix:
            return group
    return "other"


def _prefeval_topic_summaries(model_root: Path) -> list[dict]:
    mcq_root = model_root / "PrefEval" / "mcq_results"
    if not mcq_root.exists():
        return []
    out = []
    for summary_path in mcq_root.rglob("*_summary.json"):
        try:
            with summary_path.open() as f:
                out.append(json.load(f))
        except Exception as e:
            print(f"  (warn) failed to parse {summary_path}: {e}", file=sys.stderr)
    return out


def _bigtom_summary(model_root: Path, model: str) -> dict | None:
    safe = model.replace("/", "_")
    path = model_root / "BigTom" / f"summary_{safe}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _sotopia_summary(model_root: Path) -> dict | None:
    path = model_root / "Sotopia" / "sotopia_summary.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def summarize_model(model: str, results_root: Path) -> None:
    model_root = results_root / model
    print(f"\n=== {model} ===")

    # ---- PrefEval ----
    topic_summaries = _prefeval_topic_summaries(model_root)
    if topic_summaries:
        per_group_accs: dict[str, list[float]] = defaultdict(list)
        for s in topic_summaries:
            per_group_accs[_group_for(s["topic"])].append(s["accuracy"])
        print("PrefEval")
        all_accs: list[float] = []
        for group in ["travel", "shop", "lifestyle", "entertain", "pet",
                      "education", "professional"]:
            if group in per_group_accs:
                accs = per_group_accs[group]
                avg = sum(accs) / len(accs)
                print(f"  {group:<13} {_fmt_pct(avg)}  ({len(accs)} topics)")
                all_accs.extend(accs)
        if all_accs:
            overall = sum(all_accs) / len(all_accs)
            print(f"  {'Overall':<13} {_fmt_pct(overall)}  ({len(all_accs)} topics)")
    else:
        print("PrefEval: (no results found)")

    # ---- BigTom ----
    bt = _bigtom_summary(model_root, model)
    if bt:
        print("\nBigTom")
        accs = bt.get("accuracy", {})
        both = []
        for cond in ["true_belief", "false_belief"]:
            if cond in accs:
                v = accs[cond]
                print(f"  {cond:<13} {_fmt_pct(v)}")
                both.append(v)
        if both:
            print(f"  {'Overall':<13} {_fmt_pct(sum(both)/len(both))}")
    else:
        print("\nBigTom: (no results found)")

    # ---- Sotopia ----
    so = _sotopia_summary(model_root)
    if so:
        print("\nSotopia")
        dim_avgs = []
        for dim, d in so.get("dimensions", {}).items():
            avg = d.get("avg")
            rng = d.get("range")
            if avg is not None:
                dim_avgs.append(avg)
                print(f"  {dim:<34} {str(rng):<10} avg={avg:.2f}")
        if "overall_score" in so:
            print(f"  Overall score: {so['overall_score']:.2f}")
        elif dim_avgs:
            print(f"  Overall (mean of {len(dim_avgs)} dims): {sum(dim_avgs)/len(dim_avgs):.2f}")
    else:
        print("\nSotopia: (no results found)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    for model in args.models:
        summarize_model(model, args.results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
