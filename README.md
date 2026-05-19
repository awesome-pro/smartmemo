# SmartMemo

SmartMemo is a semantic memory and caching layer for LLM agent calls. Its core thesis is
simple: cosine similarity is a useful candidate selector, but it is not semantic
equivalence. SmartMemo uses embedding search to find likely cache candidates, then can use
a learned equivalence classifier to decide whether a cached response is safe to reuse.

The current implementation ships the baseline and the first classifier-gated cache path:

- async `SmartMemo.get_or_call(...)`
- SQLite persistence
- embedding provider protocol
- FAISS-backed vector search when `smartmemo[ml]` is installed
- dependency-light in-memory search for tests and smoke demos
- measured cosine-baseline benchmark fixtures for customer-support prompts
- classifier training, evaluation, checkpoint inference, and optional classifier-gated hits
- durable feedback export for classifier retraining data
- manual feedback-driven retraining with validation gates and checkpoint promotion

By default, SmartMemo keeps the lightweight cosine baseline. When you provide a classifier
checkpoint, cosine search becomes the candidate selector and the learned classifier makes
the final cache-hit decision. SmartMemo does not ship a production pretrained classifier yet.

## Install

```bash
pip install smartmemo
pip install "smartmemo[ml]"
```

For local development:

```bash
uv sync --all-extras
uv run pytest
uv run ruff check
uv run pyright
```

## Minimal Example

```python
from smartmemo import SmartMemo

cache = SmartMemo(domain="customer-support")

async def call_llm(prompt: str) -> str:
    return "fresh LLM response"

result = await cache.get_or_call(
    prompt="Summarize this customer's latest billing ticket",
    llm_function=call_llm,
)

print(result.response)
print(result.was_cache_hit)
```

## Baseline Benchmark

The customer-support benchmark is intentionally designed to show the baseline failure
mode: prompts about the same object can require opposite actions.

```bash
uv run python benchmarks/cosine_baseline_customer_support.py
```

The numbers from that benchmark are the only performance claims this implementation makes.

## Classifier Pipeline

SmartMemo includes a trainable pair classifier over prompt embeddings:

```bash
uv run smartmemo train-classifier \
  --data data/fixtures/customer_support_pairs.jsonl \
  --out models/classifier-smoke.pt \
  --embedding-provider hash \
  --embedding-dim 64 \
  --epochs 2
```

Use the hash provider only for smoke checks. Real experiments should install
`smartmemo[ml]` and use the SentenceTransformers embedding provider.

Use a trained checkpoint for classifier-gated cache decisions:

```python
from smartmemo import ClassifierConfig, SmartMemo

cache = SmartMemo(
    domain="customer-support",
    classifier=ClassifierConfig(model_path="models/classifier-smoke.pt"),
)
```

When the classifier is active, `CacheResult.classifier_score` is populated for classifier
hits and classifier-gated misses that had candidates.

## Feedback Export

SmartMemo records cache-hit lookups so explicit feedback can become training data:

```python
result = await cache.get_or_call(
    prompt="Approve the customer's refund request",
    llm_function=call_llm,
)

if result.was_cache_hit and user_rejected_answer:
    await cache.report_bad_hit(result.query_id, reason="wrong refund decision")

written = cache.export_feedback_pairs("data/feedback_pairs.jsonl")
print(written)
```

The exported JSONL uses the same prompt-pair shape accepted by `smartmemo train-classifier`.

## Manual Retraining

Use `smartmemo retrain` to turn durable feedback into a candidate classifier checkpoint:

```bash
uv run smartmemo --db-path .smartmemo/cache.db retrain \
  --out models/classifier-candidate.pt \
  --validation-data data/validation_pairs.jsonl \
  --seed-data data/fixtures/customer_support_pairs.jsonl \
  --domain customer-support \
  --min-precision 0.95 \
  --promote-to models/classifier-active.pt
```

The command always trains a candidate and writes an auditable
`<checkpoint>.report.json`. Promotion only copies the candidate to `--promote-to` when the
validation gates pass. SmartMemo does not run background retraining or automatically reload
classifiers at runtime.

## Release

Version `0.0.4` is configured for PyPI as `smartmemo`. The repository publishes through
GitHub Actions trusted publishing from `.github/workflows/publish-pypi.yml` with the
`pypi` environment.

```bash
git tag v0.0.4
git push origin v0.0.4
```

That tag builds the source distribution and wheel, then uploads them to PyPI.
