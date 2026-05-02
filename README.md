# evaluations

Unified evaluation harness for **six personalization benchmarks**, exposed
behind a single CLI (`multibench run <bench> --`) that targets any
OpenAI-compatible endpoint (vLLM, OpenAI, Anthropic via gateway, …).

| Benchmark     | Capability                          | Upstream                               | Protocol            |
|---------------|-------------------------------------|----------------------------------------|---------------------|
| `personamem`  | long-term memory                    | bowen-upenn/PersonaMem-v2              | implicit-persona MCQ|
| `prefeval`    | preference adherence                | amazon-science/PrefEval                | gen + cls + judge   |
| `bigtom`      | theory of mind                      | cicl-stanford/procedural-evals-tom     | ToM MCQ (2 × 200)   |
| `lamp`        | per-task personalization            | LaMP-Benchmark/LaMP                    | 7 tasks (F1/MAE/BLEU)|
| `personalens` | long-context personalized dialogue  | amazon-science/PersonaLens             | gen + 4-dim judge   |
| `sotopia`     | social interaction                  | sotopia-lab/sotopia                    | 7-dim social judge  |

Each benchmark parallelises at the item level via `--workers`. Evaluation
protocols, prompts, and metric functions match upstream byte-for-byte; the
only first-class divergence is that all backends are routed through a
single `LLMClient` so generation and judge endpoints can be split via
`--api-base` / `--judge-api-base`.

## Quickstart

```bash
pip install -e .

# 1. point at any OpenAI-compatible endpoint
export OPENAI_API_KEY=sk-...
M=Qwen/Qwen3-8B
BASE=http://localhost:8002/v1

# 2. run any benchmark — same flag shape
multibench run personamem -- --api-base $BASE --model $M --workers 64 \
                             --output-dir results/$M/PersonaMem
```

A full six-bench sweep for one model: `bash scripts/run_all.sh`.

## CLI

Common flags every benchmark accepts:

```
--api-base    URL          OpenAI-compatible endpoint
--model       NAME         model name on that endpoint                (required)
--output-dir  DIR          where per-item JSONL + summary lands       (required)
--workers     N            parallel requests                          (default 32)
--max-items   N            cap items for a smoke run
--api-key     KEY          (default: $OPENAI_API_KEY)
[--max-tokens N · --temperature F · --seed N · --no-strip-think]
```

Per-benchmark extras (topic, task, eval-mode, judge model, …) are listed
under `multibench run <bench> -- --help`. The most-used invocations:

```bash
# BigTom — both forward + backward conditions
multibench run bigtom      -- --api-base $BASE --model $M --condition both \
                              --output-dir results/$M/BigTom

# LaMP — one task at a time, choose retriever + profile usage
multibench run lamp        -- --api-base $BASE --model $M --task LaMP-1 \
                              --use-profile --num-retrieved 3 --retriever bm25 \
                              --output-dir results/$M/LaMP_1

# PrefEval — gen + cls + judge in one shot
multibench run prefeval    -- --api-base $BASE --model $M --topic travel_restaurant \
                              --inter-turns 2 --task zero-shot --stage all \
                              --judge-api-base https://api.openai.com/v1 \
                              --judge-model    gpt-4.1-mini \
                              --output-dir results/$M/PrefEval

# PersonaLens — separate gen and judge stages (judge can be a stronger model)
multibench run personalens -- --stage gen  --api-base $BASE --model $M \
                              --sample s5 --output-dir results/$M/PersonaLens
multibench run personalens -- --stage eval --api-base https://api.openai.com/v1 \
                              --model gpt-4.1-mini --workers 200 \
                              --eval-dim personalization --sample s5 \
                              --model-tag "${M}_d_p_s" \
                              --output-dir results/$M/PersonaLens

# PersonaMem-v2 — 32k context window required
multibench run personamem  -- --api-base $BASE --model $M --eval-mode mcq \
                              --size 32k --output-dir results/$M/PersonaMem

# Sotopia — paper-aligned 7-dim judge
multibench run sotopia     -- --api-base $BASE --model $M --workers 32 \
                              --judge-api-base https://api.openai.com/v1 \
                              --judge-model    gpt-4.1-mini \
                              --output-dir results/$M/Sotopia
```

## Data

Each benchmark reads from `data/<name>/`. Symlink to your local upstream
checkouts (gitignored) once per machine:

```bash
ln -sfn /path/to/BigTom/data                data/bigtom
ln -sfn /path/to/LaMP/data                  data/lamp
ln -sfn /path/to/PersonaLens/data           data/personalens
ln -sfn /path/to/PersonaMem-v2/data         data/personamem
ln -sfn /path/to/PrefEval/benchmark_dataset data/prefeval
# sotopia scenarios ship in-package as sotopia_scenarios.json
```

## License

Each benchmark retains its upstream license. See per-benchmark source
under `multibench/benchmarks/<name>/`.
