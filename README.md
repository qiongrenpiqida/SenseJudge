# SenseJudge Pipeline

This directory contains a compact pipeline for generating preference descriptions, judging paired responses, and reporting SenseJudge accuracy.

## Files

- `run_right.sh`: End-to-end pipeline runner.
- `0_get_dev_for_preference.py`: Splits human labels into dev and test sets.
- `1_get_preference.py`: Generates preference descriptions from the dev set.
- `2_judge.py`: Runs the judge model on dev/test data.
- `3_judge_count.py`: Selects dev preferences and computes test accuracy.
- `analysis.py`: Shared scoring and preference-selection helpers.
- `util.py`: Shared OpenAI-compatible model API client.
- `data/input/human_label/*.json`: Input human-label files.
- `data/benchmark/new_sense_bmk.json`: Benchmark prompts with English topic tags.

Generated runtime directories:

- `data/data_process/`: Dev/test split output.
- `data/preference/`: Generated preference descriptions.
- `data/result/`: Judge outputs.

These generated directories can be deleted and recreated by rerunning the pipeline.

## Model API

The model client uses the OpenAI-compatible Chat Completions endpoint:

```text
{MODEL_API_BASE}/chat/completions
```

If `MODEL_API_BASE` already ends with `/chat/completions`, it is used directly.

### DeepSeek API

```bash
export DEEPSEEK_API_KEY="sk-..."

MODEL_NAME=deepseek-chat \
PREFERENCE_MODEL_NAME=deepseek-chat \
TOPICS=Math \
MAX_SAMPLES=0 \
NUM_WORKERS=2 \
RUN_PREFERENCE=1 \
FORCE_RERUN=1 \
./run_right.sh
```

Optional:

```bash
export DEEPSEEK_API_BASE="https://api.deepseek.com/v1"
```

### Self-Hosted Model

Use any OpenAI-compatible service, such as vLLM or another local deployment:

```bash
export MODEL_API_BASE="http://localhost:8000/v1"
export MODEL_API_KEY=""

MODEL_NAME=qwen2.5-72b-instruct \
PREFERENCE_MODEL_NAME=qwen2.5-72b-instruct \
TOPICS=Math \
MAX_SAMPLES=0 \
NUM_WORKERS=2 \
RUN_PREFERENCE=1 \
FORCE_RERUN=1 \
./run_right.sh
```

If your service requires authentication:

```bash
export MODEL_API_KEY="your-api-key"
```

## Environment Variables

Pipeline variables:

- `NAME`: Human-label file name without `.json`. Default: `sense-label3`.
- `INPUT_JSON_FILE`: Input label path. Default: `data/input/human_label/${NAME}.json`.
- `MODEL_NAME`: Judge model name. Default: `deepseek-v4-pro`.
- `PREFERENCE_MODEL_NAME`: Preference-generation model name. Default: same as `MODEL_NAME`.
- `TOPICS`: Space-separated topic names. Example: `TOPICS="Math Code"`.
- Supported topic names: `Code`, `Translation`, `Role`, `NLU`, `Math`, `Logic`, `Writing`, `QA`.
- `MAX_SAMPLES`: Limit samples per topic. `0` means full topic.
- `NUM_WORKERS`: Parallel judge workers.
- `RUN_PREFERENCE`: `1` to regenerate preferences, `0` to reuse existing preferences.
- `FORCE_RERUN`: `1` to delete existing judge outputs before rerun.

Model API variables:

- `MODEL_API_BASE`: Generic OpenAI-compatible API base URL.
- `MODEL_API_KEY`: Generic API key.
- `OPENAI_API_BASE`: Alternative generic API base URL.
- `OPENAI_API_KEY`: Alternative generic API key.
- `SELF_HOSTED_API_BASE`: Alternative self-hosted API base URL.
- `LOCAL_MODEL_API_BASE`: Local fallback API base. Default: `http://localhost:8000/v1`.
- `LOCAL_MODEL_API_KEY`: Local model API key, if needed.
- `MODEL_API_MAX_TOKENS`: Maximum tokens for model output.
- `MODEL_API_TEMPERATURE`: Sampling temperature. Default: `0`.
- `MODEL_API_TOP_P`: Optional top-p value.
- `MODEL_API_TIMEOUT`: Request timeout in seconds.
- `MODEL_API_EXTRA_HEADERS`: Optional JSON object of extra HTTP headers.

DeepSeek-compatible variables:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_API_BASE`
- `DEEPSEEK_MAX_TOKENS`
- `DEEPSEEK_THINKING`
- `DEEPSEEK_REASONING_EFFORT`

## Pipeline Steps

`run_right.sh` executes four steps:

1. Split input human labels into dev/test sets.
2. Generate preference descriptions on the dev set.
3. Run dev and test judging.
4. Select dev preferences and compute SenseJudge accuracy.

## Scoring Notes

- `reverse=False`: Response A is treated as the correct answer.
- `reverse=True`: Response B is treated as the correct answer.
- Unparseable judge outputs are counted as incorrect.
- Ties default to Response A.
- The denominator is computed from the actual available data, not from the deprecated `--denominator` argument.

## Quick Smoke Test

Use mock models to test the local pipeline shape without calling an external API:

```bash
MODEL_NAME=mock-response-a \
PREFERENCE_MODEL_NAME=mock-response-a \
TOPICS=Math \
MAX_SAMPLES=1 \
NUM_WORKERS=1 \
RUN_PREFERENCE=1 \
FORCE_RERUN=1 \
./run_right.sh
```
