# p13n-eval-harness

Unified evaluation harness for 6 personalization benchmarks over any
OpenAI-compatible endpoint (vLLM, OpenAI, etc.). Every benchmark takes
the same base flags; every benchmark parallelises via `--workers`.

## Benchmarks

| name          | upstream                               | protocol            |
|---------------|----------------------------------------|---------------------|
| `bigtom`      | cicl-stanford/procedural-evals-tom     | ToM MCQ (2 × 200)   |
| `lamp`        | LaMP-Benchmark/LaMP                    | 7 tasks, F1/MAE/BLEU|
| `personalens` | amazon-science/PersonaLens             | gen + 4-dim judge   |
| `personamem`  | bowen-upenn/PersonaMem-v2              | implicit-persona MCQ|
| `prefeval`    | amazon-science/PrefEval                | gen + cls + judge   |
| `sotopia`     | sotopia-lab/sotopia (paper-aligned)    | 7-dim social judge  |

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## CLI — unified flags

```
multibench run <bench> -- \
    --api-base   URL           (OpenAI-compatible base)
    --model      NAME          (required; model name on the endpoint)
    --output-dir DIR           (required)
    --workers    N             (default 32; parallel requests)
    --max-items  N             (cap for quick testing)
    --api-key    KEY           (default $OPENAI_API_KEY)
    [--max-tokens N  --temperature F  --seed N  --no-strip-think]
```

Each benchmark adds its own flags on top (topic, task, dim, …). See
`multibench run <bench> -- --help`.

## Usage

```bash
# BigTom
multibench run bigtom -- \
    --api-base http://localhost:8002/v1 --model $M --workers 64 \
    --condition both --output-dir results/$M/BigTom

# LaMP (run per task)
multibench run lamp -- \
    --api-base http://localhost:8002/v1 --model $M --workers 64 \
    --task LaMP-1 --use-profile --num-retrieved 3 --retriever bm25 \
    --output-dir results/$M/LaMP_1

# PrefEval — full pipeline (gen + cls + judge + accuracy)
multibench run prefeval -- \
    --api-base http://localhost:8002/v1 --model $M --workers 64 \
    --topic travel_restaurant --inter-turns 2 --task zero-shot --stage all \
    --judge-api-base https://api.openai.com/v1 --judge-model gpt-4.1-mini \
    --output-dir results/$M/PrefEval

# PersonaLens — generate then judge
multibench run personalens -- --stage gen \
    --api-base http://localhost:8002/v1 --model $M --workers 64 --sample s5 \
    --output-dir results/$M/PersonaLens
multibench run personalens -- --stage eval \
    --api-base https://api.openai.com/v1 --model gpt-4.1-mini --workers 200 \
    --model-tag "${M}_d_p_s" --eval-dim personalization --sample s5 \
    --output-dir results/$M/PersonaLens

# PersonaMem-v2
multibench run personamem -- \
    --api-base http://localhost:8002/v1 --model $M --workers 64 \
    --eval-mode mcq --size 32k --output-dir results/$M/PersonaMem

# Sotopia
multibench run sotopia -- \
    --api-base http://localhost:8002/v1 --model $M --workers 32 \
    --judge-api-base https://api.openai.com/v1 --judge-model gpt-4.1-mini \
    --output-dir results/$M/Sotopia
```

Full sweep for one model: `scripts/run_all.sh`.

## Data

Each benchmark reads from `data/<name>/`. These are symlinks to upstream
downloads and are gitignored. Create once:

```bash
ln -sfn /path/to/BigTom/data          data/bigtom
ln -sfn /path/to/LaMP/data            data/lamp
ln -sfn /path/to/PersonaLens/data     data/personalens
ln -sfn /path/to/PersonaMem-v2/data   data/personamem
ln -sfn /path/to/PrefEval/benchmark_dataset  data/prefeval
# sotopia scenarios ship with the package (sotopia_scenarios.json)
```

## Layout

```
multibench/
├── cli.py           # `multibench run <bench>` dispatcher
├── client.py        # shared LLMClient — chat() + parallel chat_batch()
├── args.py          # common CLI flags
├── utils.py         # strip_think, atomic_write_json, data-dir resolver
└── benchmarks/
    ├── bigtom/      run.py
    ├── lamp/        run.py  prompts/  data_utils/
    ├── personalens/ run.py  src/      util/
    ├── personamem/  run.py  inference.py  query_llm.py  …
    ├── prefeval/    run.py  utils/    generation_task/  classification_task/
    └── sotopia/     run.py  core.py   sotopia_scenarios.json
```

## Divergences from upstream

- All backends routed through a single OpenAI-compatible `LLMClient`
  (no boto3/Bedrock, no Replicate).
- Previously-serial harnesses (PrefEval gen/cls/judge, BigTom, LaMP) now
  parallelise at the item level via `ThreadPoolExecutor`.
- PrefEval LLM judge ported from Bedrock Claude-3-Sonnet to the shared
  client; `--judge-api-base` / `--judge-model` allow splitting generation
  and judge endpoints.
- `<think>...</think>` stripping on by default for Qwen3-style models.
- Evaluation protocols match upstream (MCQ prompts, metric functions,
  score ranges). Sotopia uses a paper-aligned 7-dim judge rather than
  the full upstream Redis/FastAPI package; OOB judge scores trigger one
  retry before falling back to clamping.

## License

Each benchmark retains its upstream license. See individual
`multibench/benchmarks/<name>/` for the ported source files.
