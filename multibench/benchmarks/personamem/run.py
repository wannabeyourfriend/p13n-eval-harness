"""PersonaMem-v2 — unified runner.

Upstream: https://github.com/bowen-upenn/PersonaMem-v2

The upstream `inference.py` ships its own argparse CLI; we wrap it so the
unified multibench flags translate to its argument names. `QueryLLM` inside
upstream reads the model name from the config YAML plus the env vars
`OPENAI_API_KEY` / `OPENAI_BASE_URL` — we set those at launch so a local
vLLM endpoint is used transparently.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from ...args import add_common_llm_args
from ...utils import benchmark_data_dir, ensure_dir


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PersonaMem-v2 unified runner.")
    add_common_llm_args(ap)
    ap.add_argument("--eval-mode", "--eval_mode", dest="eval_mode",
                    default="mcq", choices=["mcq", "generative", "both"])
    ap.add_argument("--size", default="32k", choices=["32k", "128k"])
    ap.add_argument("--benchmark-file", "--benchmark_file", dest="benchmark_file",
                    default=None,
                    help="Path to benchmark CSV (default: data/personamem/benchmark.csv).")
    ap.add_argument("--config", default=None,
                    help="Path to config.yaml (default: <module>/data_generation/config.yaml).")
    ap.add_argument("--run-judges", "--run_judges", dest="run_judges",
                    action="store_true",
                    help="Generative-mode LLM judge pass.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return ap


def _setup_env(args) -> str:
    # upstream QueryLLM reads OPENAI_API_KEY / OPENAI_BASE_URL to reach the backend
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    elif "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "not-needed"
    os.environ["OPENAI_BASE_URL"] = args.api_base
    # config path
    if args.config:
        return args.config
    return str(Path(__file__).parent / "data_generation" / "config.yaml")


def _resolve_bench_file(args) -> str:
    if args.benchmark_file:
        return args.benchmark_file
    data_dir = benchmark_data_dir("personamem")
    cand = data_dir / f"benchmark_{args.size}.csv"
    if cand.exists():
        return str(cand)
    cand = data_dir / "benchmark.csv"
    if cand.exists():
        return str(cand)
    raise FileNotFoundError(f"No benchmark CSV under {data_dir}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = ensure_dir(args.output_dir)
    config_path = _setup_env(args)
    bench_file = _resolve_bench_file(args)

    forwarded = [
        "--model_name", args.model,
        "--result_path", str(out_dir),
        "--eval_mode", args.eval_mode,
        "--size", args.size,
        "--benchmark_file", bench_file,
        "--config", config_path,
        "--parallel", str(args.workers),
    ]
    if args.max_items:
        forwarded += ["--max_items", str(args.max_items)]
    if args.run_judges:
        forwarded += ["--run_judges"]
    if args.verbose:
        forwarded += ["--verbose"]
    forwarded += list(args.extra)

    orig_argv = sys.argv[:]
    sys.argv = ["inference"] + forwarded
    try:
        runpy.run_module("multibench.benchmarks.personamem.inference",
                         run_name="__main__")
    finally:
        sys.argv = orig_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
