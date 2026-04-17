"""Unified OpenAI-compatible LLM client shared by all benchmarks.

Wraps a single HTTP endpoint (vLLM / OpenAI / any compatible server) and
exposes synchronous `.chat()` + parallel `.chat_batch()` helpers.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from openai import OpenAI
from tqdm import tqdm

from .utils import strip_think_tags

logger = logging.getLogger(__name__)


@dataclass
class LLMClient:
    """Thin, reusable wrapper around an OpenAI-compatible endpoint.

    Attributes
    ----------
    model_name : str
        The served model id on the server.
    api_base : str
        Full base URL, e.g. "http://localhost:8000/v1".
    api_key : str
        Any non-empty string works for vLLM. Falls back to $OPENAI_API_KEY.
    max_retries : int
        Exponential backoff retries on transient failures.
    default_max_tokens : int
        Default `max_tokens` when not overridden at call site.
    default_temperature : float
        Default sampling temperature.
    strip_think : bool
        If true, strip <think>...</think> reasoning blocks from completions.
    """

    model_name: str = "default"
    api_base: str = "http://localhost:8000/v1"
    api_key: str | None = None
    max_retries: int = 6
    default_max_tokens: int = 512
    default_temperature: float = 0.0
    strip_think: bool = True

    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self._client = OpenAI(base_url=self.api_base, api_key=key)

    # -------- single call --------
    def chat(
        self,
        messages: Sequence[dict] | str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
        extra_body: dict | None = None,
    ) -> str:
        """Synchronous chat completion. Returns stripped text content."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        kwargs: dict[str, Any] = {
            "model": model or self.model_name,
            "messages": msgs,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        backoff = 1.0
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                return strip_think_tags(text) if self.strip_think else text.strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("chat attempt %d/%d failed: %s", attempt + 1, self.max_retries, e)
                if attempt == self.max_retries - 1:
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        raise RuntimeError(f"chat failed after {self.max_retries} retries: {last_err}")

    # -------- parallel batch --------
    def chat_batch(
        self,
        items: Sequence[Any],
        build_messages: Callable[[Any], Sequence[dict] | str],
        *,
        workers: int = 20,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        desc: str = "chat",
        on_error: str = "store",  # or "raise"
    ) -> list[Any]:
        """Call `.chat()` in parallel for each item.

        Each element of `items` is passed to `build_messages(item)` which must
        return a messages sequence or a user-text string.

        Returns a list aligned with `items`. On error, stores the exception
        object at that index (if `on_error='store'`).
        """
        results: list[Any] = [None] * len(items)

        def _one(idx: int) -> tuple[int, Any]:
            try:
                msgs = build_messages(items[idx])
                out = self.chat(
                    msgs,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return idx, out
            except Exception as e:  # noqa: BLE001
                if on_error == "raise":
                    raise
                return idx, e

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(_one, i) for i in range(len(items))]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                idx, val = fut.result()
                results[idx] = val
        return results


def client_from_args(args) -> LLMClient:
    """Construct an LLMClient from argparse-style args with attributes
    `api_base`, `model` (or `model_name`), optional `api_key`, `max_tokens`,
    `temperature`, `strip_think`.
    """
    model_name = getattr(args, "model", None) or getattr(args, "model_name", None) or "default"
    return LLMClient(
        model_name=model_name,
        api_base=getattr(args, "api_base", "http://localhost:8000/v1"),
        api_key=getattr(args, "api_key", None),
        default_max_tokens=getattr(args, "max_tokens", 512),
        default_temperature=getattr(args, "temperature", 0.0),
        strip_think=getattr(args, "strip_think", True),
    )
