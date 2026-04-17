"""Unified `multibench` CLI entry point.

Usage:
    multibench run <benchmark> [--api-base URL --model NAME --workers N ...] [benchmark-specific args]

Each benchmark ships its own argparse-based `run` function under
`multibench.benchmarks.<name>.run:main`. The dispatcher here just discovers
the module and forwards remaining argv to it.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from types import ModuleType

BENCHMARKS = [
    "bigtom",
    "lamp",
    "personalens",
    "personamem",
    "prefeval",
    "sotopia",
]


def _load(bench: str) -> ModuleType:
    try:
        return importlib.import_module(f"multibench.benchmarks.{bench}.run")
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Unknown or unimportable benchmark '{bench}'. Known: {BENCHMARKS}\n"
            f"Import error: {e}"
        )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="multibench",
        description="Run personalization benchmarks against an OpenAI-compatible endpoint.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a single benchmark.")
    p_run.add_argument("benchmark", choices=BENCHMARKS)
    p_run.add_argument("rest", nargs=argparse.REMAINDER,
                       help="Arguments forwarded to the benchmark runner (use -- to separate).")

    sub.add_parser("list", help="List available benchmarks.")

    args = parser.parse_args(argv)

    if args.cmd == "list":
        print("\n".join(BENCHMARKS))
        return 0

    if args.cmd == "run":
        mod = _load(args.benchmark)
        forwarded = args.rest
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        if not hasattr(mod, "main"):
            raise SystemExit(f"Benchmark '{args.benchmark}' has no main() entry point.")
        rc = mod.main(forwarded)
        return rc or 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
