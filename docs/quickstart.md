# Quickstart

SmartMemo is an async-first semantic cache for LLM agent calls. Embedding search finds
candidate cache entries; a learned equivalence classifier decides whether a candidate is
genuinely safe to reuse. That classifier is what separates SmartMemo from a fixed cosine
threshold: "approve this refund" and "deny this refund" are highly cosine-similar but must
never share a cache entry.

## Install

SmartMemo's embedding and classifier stack depends on PyTorch, FAISS, and
SentenceTransformers, so install the `ml` extra:

```bash
pip install "smartmemo[ml]"
```

## Minimal use

```python
from smartmemo import SmartMemo

cache = SmartMemo(domain="customer-support")

async def call_llm(prompt: str) -> str:
    return "fresh response from your provider"

result = await cache.get_or_call(prompt="Summarize this ticket", llm_function=call_llm)
print(result.response)
print(result.was_cache_hit)
```

Without a classifier, SmartMemo decides cache hits with a cosine-similarity threshold —
the measured baseline the classifier is built to beat.

## Classifier-gated caching (recommended)

SmartMemo ships a pretrained generic equivalence classifier. Turn it on with one line:

```python
from smartmemo import ClassifierConfig, SmartMemo

cache = SmartMemo(
    domain="customer-support",
    classifier=ClassifierConfig.bundled(),
)
```

Cosine search now only selects candidates; the learned classifier makes the final
cache-hit decision, and `CacheResult.classifier_score` is populated. The bundled model
is a generic cold-start classifier — accuracy on your own traffic improves with the
feedback-driven retraining loop below. See `docs/ml/how-the-classifier-works.md`.

## Feedback export

```python
result = await cache.get_or_call(prompt="Summarize this ticket", llm_function=call_llm)

if result.was_cache_hit:
    await cache.report_bad_hit(result.query_id, reason="user rejected cached answer")

cache.export_feedback_pairs("data/feedback_pairs.jsonl")
```

## Manual retraining

```bash
uv run smartmemo --db-path .smartmemo/cache.db retrain \
  --out models/classifier-candidate.pt \
  --validation-data data/validation_pairs.jsonl \
  --seed-data data/fixtures/customer_support_pairs.jsonl \
  --domain customer-support \
  --min-precision 0.95 \
  --promote-to models/classifier-active.pt
```

The retrain command trains a candidate checkpoint and writes a report next to it. It only
promotes the checkpoint when validation gates pass; runtime classifier loading remains an
explicit `ClassifierConfig(model_path=...)` choice.
