"""Sotopia — unified runner over the paper-aligned simplified harness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...args import add_common_llm_args
from ...client import LLMClient, client_from_args
from ...utils import atomic_write_json, ensure_dir

from .core import run_sotopia


SCENARIOS_FILE = Path(__file__).with_name("sotopia_scenarios.json")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sotopia unified runner.")
    add_common_llm_args(ap)
    ap.add_argument("--scenarios", default=None,
                    help="Path to scenarios JSON (default: sotopia_scenarios.json).")
    ap.add_argument("--max-scenarios", "--max_scenarios", dest="max_scenarios",
                    type=int, default=None,
                    help="Subset of scenarios to run (random, seeded). Default: all.")
    ap.add_argument("--judge-api-base", "--judge_api_base", dest="judge_api_base",
                    default=None)
    ap.add_argument("--judge-model", "--judge_model", dest="judge_model",
                    default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = ensure_dir(args.output_dir)

    sim_client = client_from_args(args)
    if args.judge_api_base or args.judge_model:
        judge_client = LLMClient(
            model_name=args.judge_model or args.model,
            api_base=args.judge_api_base or args.api_base,
            api_key=args.api_key,
            default_temperature=0.0, default_max_tokens=2048,
            strip_think=True,
        )
    else:
        judge_client = sim_client

    scen_path = Path(args.scenarios) if args.scenarios else SCENARIOS_FILE
    with scen_path.open(encoding="utf-8") as f:
        scenarios = json.load(f)

    # --max-items takes precedence over --max-scenarios if both set
    max_scen = args.max_items if args.max_items else args.max_scenarios
    results, summary = run_sotopia(
        scenarios, sim_client=sim_client, judge_client=judge_client,
        workers=args.workers, seed=args.seed, max_scenarios=max_scen,
    )
    atomic_write_json(out_dir / "sotopia_results.json", results)
    atomic_write_json(out_dir / "sotopia_summary.json", summary)

    print(f"[sotopia] overall={summary['overall_score']:.2f}  "
          f"per-agent={summary['per_agent_overall_avg']:.2f}  "
          f"scenarios={summary['total_scenarios']}  "
          f"agents={summary['total_agent_evaluations']}")
    for dim, d in summary["dimensions"].items():
        print(f"  {dim:<34} {str(d['range']):<10} avg={d['avg']:.2f}  std={d['std']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
