"""PersonaLens — unified runner over the upstream-derived harness.

Upstream: https://github.com/amazon-science/PersonaLens

This is a thin adapter: the upstream `generate_dialogue.py` and
`evaluate_dialogue.py` modules already ship a rich argparse CLI (including
`--vllm`, `--vllm_api_base`, `--vllm_model_name`, `--parallel`). Rather than
rewriting them, we translate the unified multibench flags into their
argument names and hand off to the module's `__main__` block via `runpy`.

Stages
------
- gen  : generate_dialogue.py — simulate user↔assistant dialogues
- eval : evaluate_dialogue.py — LLM-as-judge over generated dialogues
         (--eval-dim / -d selects the dimension: personalization,
         task_completion, naturalness, coherence, ...).

Env-vars DATA_DIR and OUTPUT_DIR point the upstream code at data and
output paths (it reads them at import time).
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from ...args import add_common_llm_args
from ...utils import benchmark_data_dir, ensure_dir


def _forward(module_dotted: str, argv_for_module: list[str]) -> int:
    """Run an upstream-like __main__ module under a spoofed sys.argv."""
    orig_argv = sys.argv[:]
    sys.argv = [module_dotted.split(".")[-1]] + argv_for_module
    try:
        runpy.run_module(module_dotted, run_name="__main__")
    finally:
        sys.argv = orig_argv
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PersonaLens unified runner.")
    add_common_llm_args(ap)
    ap.add_argument("--stage", required=True, choices=["gen", "eval"],
                    help="gen = generate_dialogue; eval = evaluate_dialogue (LLM judge).")
    ap.add_argument("--model-tag", "--model_tag", dest="model_tag", default=None,
                    help="PersonaLens-style model tag, e.g. 'us-profile-mar31_d_p_s'. "
                         "Defaults to --model when omitted.")
    ap.add_argument("--eval-dim", "--eval_dim", "-d", dest="eval_dim", default="personalization",
                    help="Judge dimension (eval stage only).")
    ap.add_argument("--sample", default="s5", choices=["s3", "s5", "s10", "all"],
                    help="User sample size: s3=30 users, s5=50, s10=100, all=range.")
    ap.add_argument("--user-model", "--user_model", dest="user_model",
                    default=None, help="User-simulator model id (gen stage).")
    ap.add_argument("--assistant-model", "--assistant_model", dest="assistant_model",
                    default=None, help="Assistant model id (gen stage; default=--model).")
    ap.add_argument("--judge-model", "--judge_model", dest="judge_model",
                    default=None, help="Judge model (eval stage; default=--model).")
    ap.add_argument("--judge-api-base", "--judge_api_base", dest="judge_api_base",
                    default=None, help="Judge API base (eval stage; default=--api-base).")
    ap.add_argument("--data-dir", "--data_dir", dest="data_dir", default=None,
                    help="Override PersonaLens data dir (default: data/personalens).")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Any additional flags passed through to the upstream script verbatim.")
    return ap


def _setup_env(args) -> None:
    data_dir = Path(args.data_dir) if args.data_dir else benchmark_data_dir("personalens")
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["OUTPUT_DIR"] = str(ensure_dir(args.output_dir))


def _sample_flag(sample: str) -> list[str]:
    return {"s3": ["-s3"], "s5": ["-s5"], "s10": ["-s10"], "all": []}[sample]


def _common_vllm_flags(args, model_name: str, api_base: str) -> list[str]:
    flags = [
        "--vllm",
        "--vllm_model_name", model_name,
        "--vllm_api_base", api_base,
        "--parallel", str(args.workers),
    ]
    return flags


def _run_gen(args) -> int:
    model_tag = args.model_tag or args.model
    flags = _common_vllm_flags(args, args.model, args.api_base)
    flags += _sample_flag(args.sample)
    flags += ["-m", args.assistant_model or model_tag]
    if args.user_model:
        flags += ["-u", args.user_model]
    flags += list(args.extra)
    return _forward("multibench.benchmarks.personalens.src.generate_dialogue", flags)


def _run_eval(args) -> int:
    model_tag = args.model_tag or args.model
    judge_model = args.judge_model or args.model
    judge_api = args.judge_api_base or args.api_base
    flags = _common_vllm_flags(args, judge_model, judge_api)
    flags += _sample_flag(args.sample)
    flags += ["-m", model_tag, "-d", args.eval_dim]
    flags += list(args.extra)
    return _forward("multibench.benchmarks.personalens.src.evaluate_dialogue", flags)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _setup_env(args)
    if args.stage == "gen":
        return _run_gen(args)
    if args.stage == "eval":
        return _run_eval(args)
    raise ValueError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
