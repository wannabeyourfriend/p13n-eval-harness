# multi-bench

Self-contained evaluation harness for personalization benchmarks, unified over
OpenAI-compatible endpoints (vLLM, OpenAI, any chat-completions server).

## Benchmarks

| Name          | Upstream                                     | Harness fidelity vs. paper |
|---------------|-----------------------------------------------|----------------------------|
| `bigtom`      | cicl-stanford/procedural-evals-tom            | ✅ identical MCQ protocol  |
| `lamp`        | LaMP-Benchmark/LaMP                           | ✅ identical metrics       |
| `personalens` | amazon-science/PersonaLens                    | ✅ judge swapped to gpt-4.1-mini (configurable) |
| `personamem`  | bowen-upenn/PersonaMem-v2                     | ⚠️ upstream-modified MCQ prompt (kept from orig fork) |
| `prefeval`    | amazon-science/PrefEval                       | ✅ generation + cls + judge + accuracy all wired |
| `sotopia`     | sotopia-lab/sotopia (paper-aligned reimpl.)  | ⚠️ simplified — clamp instead of raise on OOB scores |

CURATe (`lize-alberts/llm_prag_benchmark`) was dropped.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Unified CLI contract

Every benchmark accepts the same common flags:

| flag                        | purpose                                  |
|-----------------------------|------------------------------------------|
| `--api-base URL`            | OpenAI-compatible base URL               |
| `--model NAME` (required)   | Model name on the endpoint               |
| `--api-key KEY`             | Optional (default `$OPENAI_API_KEY`)     |
| `--output-dir DIR` (req)    | Where to write results                   |
| `--workers N`               | Parallel request workers (default 32)    |
| `--max-items N`             | Cap items for quick testing              |
| `--max-tokens`, `--temperature`, `--no-strip-think`, `--seed` | defaults |

Then each benchmark adds its own flags on top (topic, task, dim, etc.).

## Usage

```bash
# list available benchmarks
multibench list

# BigTom — both conditions in parallel
multibench run bigtom -- --api-base http://localhost:8002/v1 \
    --model us-profile-mar31 --condition both --workers 64 \
    --output-dir results/us-profile-mar31/BigTom

# LaMP — any of LaMP-1..7
multibench run lamp -- --api-base http://localhost:8002/v1 \
    --model us-profile-mar31 --task LaMP-1 --use-profile \
    --num-retrieved 3 --retriever bm25 --workers 64 \
    --output-dir results/us-profile-mar31/LaMP_1

# PrefEval — full pipeline (gen + cls + judge + accuracy)
multibench run prefeval -- --api-base http://localhost:8002/v1 \
    --model us-profile-mar31 --topic travel_restaurant \
    --inter-turns 2 --task zero-shot --stage all --workers 64 \
    --judge-api-base https://api.openai.com/v1 --judge-model gpt-4.1-mini \
    --output-dir results/us-profile-mar31/PrefEval

# PersonaLens — generate then judge
multibench run personalens -- --stage gen --api-base http://localhost:8002/v1 \
    --model us-profile-mar31 --sample s5 --workers 64 \
    --output-dir results/us-profile-mar31/PersonaLens
multibench run personalens -- --stage eval --api-base https://api.openai.com/v1 \
    --model gpt-4.1-mini --model-tag us-profile-mar31_d_p_s \
    --eval-dim personalization --sample s5 --workers 200 \
    --output-dir results/us-profile-mar31/PersonaLens

# PersonaMem-v2 — MCQ eval
multibench run personamem -- --api-base http://localhost:8002/v1 \
    --model us-profile-mar31 --eval-mode mcq --size 32k --workers 64 \
    --output-dir results/us-profile-mar31/PersonaMem

# Sotopia — 90 scenarios, parallelised
multibench run sotopia -- --api-base http://localhost:8002/v1 \
    --model us-profile-mar31 --workers 32 \
    --judge-api-base https://api.openai.com/v1 --judge-model gpt-4.1-mini \
    --output-dir results/us-profile-mar31/Sotopia
```

## Data

Each benchmark expects its dataset under `data/<name>/` as a symlink to the
upstream download. Symlinks are gitignored. Create them once:

```bash
# examples — adjust paths to your checkouts
ln -sfn /path/to/BigTom/data          data/bigtom
ln -sfn /path/to/LaMP/data            data/lamp
ln -sfn /path/to/PersonaLens/data     data/personalens
ln -sfn /path/to/PersonaMem-v2/data   data/personamem
ln -sfn /path/to/PrefEval/benchmark_dataset  data/prefeval
# sotopia scenarios ship with the package (sotopia_scenarios.json)
```

## Package layout

```
multi-bench/
├── multibench/
│   ├── cli.py              # entry point: `multibench run <bench>`
│   ├── client.py           # LLMClient — single shared OpenAI-compatible client
│   ├── args.py             # shared argparse group (add_common_llm_args)
│   ├── utils.py            # strip_think_tags, atomic_write_json, data paths
│   └── benchmarks/
│       ├── bigtom/     run.py  prompt_evaluate.txt
│       ├── lamp/       run.py  prompts/  metrics/  data_utils/
│       ├── personalens/ run.py  src/  util/
│       ├── personamem/  run.py  inference.py  query_llm.py  ...
│       ├── prefeval/    run.py  utils/  generation_task/  classification_task/
│       │                error_type/  config.yaml
│       └── sotopia/    run.py  core.py  sotopia_scenarios.json
├── data/                  # symlinked datasets (gitignored contents)
├── results/               # default output root (gitignored)
├── pyproject.toml
└── requirements.txt
```

## What's different from upstream

See `docs/MIGRATION_NOTES.md` for the line-by-line diff of what was kept,
ported, added, or dropped per benchmark. Headline changes:

- **Everything talks to an OpenAI-compatible endpoint** (`LLMClient`), no
  AWS Bedrock / boto3 / Replicate dependency.
- **Everything parallelises at the item level** via `ThreadPoolExecutor` in
  `LLMClient.chat_batch`. Previously-serial harnesses (PrefEval gen/cls/judge,
  BigTom, LaMP) now respect `--workers`.
- **PrefEval judge ported** from Bedrock Claude-3-Sonnet to the shared OpenAI
  client; `--judge-api-base` / `--judge-model` let you pick a separate grader
  (e.g. keep generation local, run judge on gpt-4.1-mini).
- **`<think>...</think>` stripping** is always on (configurable via
  `--no-strip-think`) so Qwen3 reasoning blocks don't pollute answer text.
