"""Shared argparse helpers: common flags every benchmark accepts."""
from __future__ import annotations

import argparse


def add_common_llm_args(parser: argparse.ArgumentParser) -> None:
    """Add flags for the OpenAI-compatible endpoint shared across benchmarks.

    These are the unified CLI contract. Every benchmark runner should use them
    (and then add its own benchmark-specific flags on top).
    """
    g = parser.add_argument_group("LLM endpoint")
    g.add_argument("--api-base", "--api_base", dest="api_base",
                   default="http://localhost:8000/v1",
                   help="OpenAI-compatible base URL.")
    g.add_argument("--model", "--model-name", "--model_name", dest="model",
                   required=True,
                   help="Model name as served by the vLLM/OpenAI endpoint.")
    g.add_argument("--api-key", "--api_key", dest="api_key", default=None,
                   help="Optional API key (default: $OPENAI_API_KEY or 'not-needed').")
    g.add_argument("--max-tokens", "--max_tokens", dest="max_tokens",
                   type=int, default=512, help="Default max completion tokens.")
    g.add_argument("--temperature", type=float, default=0.0,
                   help="Default sampling temperature.")
    g.add_argument("--no-strip-think", dest="strip_think", action="store_false",
                   default=True, help="Keep <think>...</think> blocks in output.")

    g2 = parser.add_argument_group("Runtime")
    g2.add_argument("--workers", "--parallel", dest="workers", type=int, default=32,
                    help="Parallel request workers (ThreadPoolExecutor).")
    g2.add_argument("--max-items", "--max_items", dest="max_items",
                    type=int, default=None,
                    help="Optional cap on items for quick testing.")
    g2.add_argument("--output-dir", "--output_dir", dest="output_dir",
                    default=None, required=True,
                    help="Directory to write results to.")
    g2.add_argument("--seed", type=int, default=0, help="Random seed where used.")
