# SmartMemo

SmartMemo is a semantic memory and caching layer for LLM agent calls. Its core thesis is
simple: cosine similarity is a useful candidate selector, but it is not semantic
equivalence. SmartMemo uses embedding search to find likely cache candidates, then uses a
learned equivalence classifier to decide whether a cached response is safe to reuse.

As of `0.1.0`, SmartMemo **ships a pretrained classifier**, so that decision works out of
the box — no training required.

- async `SmartMemo.get_or_call(...)`
- a bundled pretrained equivalence classifier (`classifier-v1`), opt-in with one line
- SQLite persistence
- embedding provider protocol with SentenceTransformers embeddings and FAISS vector search
- a reproducible local-LLM training-data pipeline and a hand-curated gold test set
- classifier training, evaluation, checkpoint inference, and classifier-gated cache hits
- durable feedback export and manual feedback-driven retraining with validation gates

Without a classifier, SmartMemo decides cache hits with a cosine threshold — the measured
baseline. With the bundled classifier, cosine search becomes the candidate selector and
the learned classifier makes the final cache-hit decision.

## Install

SmartMemo's embedding and classifier stack depends on PyTorch, FAISS, and
SentenceTransformers, so install the `ml` extra:

```bash
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
from smartmemo import ClassifierConfig, SmartMemo

cache = SmartMemo(
    domain="customer-support",
    classifier=ClassifierConfig.bundled(),
)

async def call_llm(prompt: str) -> str:
    return "fresh LLM response"

result = await cache.get_or_call(
    prompt="Summarize this customer's latest billing ticket",
    llm_function=call_llm,
)

print(result.response)
print(result.was_cache_hit)
print(result.classifier_score)
```

## The Bundled Classifier

`classifier-v1` is a generic, cross-domain equivalence classifier shipped inside the
package at `smartmemo/_models/classifier-v1.pt`. It is a small MLP over
`all-MiniLM-L6-v2` embeddings, trained on ~8,800 labeled prompt pairs built by a local
LLM paraphraser (positives) and templated same-object/opposite-action swaps (hard
negatives). The whole pipeline is `scripts/generate_training_data.py`.

Measured on a hand-curated gold set of 84 prompt pairs (31 equivalent, 53 not):

| Decision method                   | Precision | Recall | F1   |
|------------------------------------|-----------|--------|------|
| Cosine baseline (at equal recall)  | 0.53      | 0.90   | 0.67 |
| `classifier-v1` (threshold 0.95)   | 0.85      | 0.90   | 0.88 |

That is **+32 precision points at equal recall**: on this gold set the cosine baseline
makes 25 false-positive cache hits where `classifier-v1` makes 5. The full, auditable
model card is `smartmemo/_models/classifier-v1.report.json`.

`classifier-v1` is a cold-start model. It is bound to the `all-MiniLM-L6-v2` embedding
space (384 dimensions), and per-domain accuracy improves with the feedback-driven
retraining loop below.

## Benchmarks

```bash
uv run python benchmarks/cosine_baseline_customer_support.py
uv run python benchmarks/classifier_vs_cosine.py
```

The first benchmark shows the cosine baseline's false-positive failure mode on
customer-support prompts. The second scores the bundled classifier against the cosine
baseline on the gold set and writes `benchmarks/results/classifier_vs_cosine.json`.

## Training Your Own Classifier

SmartMemo includes a trainable pair classifier over prompt embeddings. To reproduce the
shipped model from the committed dataset:

```bash
uv run python scripts/train_classifier_v1.py
```

To train on your own JSONL prompt pairs:

```bash
uv run smartmemo train-classifier \
  --data data/fixtures/customer_support_pairs.jsonl \
  --out models/classifier-custom.pt \
  --domain customer-support \
  --epochs 5
```

Then point SmartMemo at the checkpoint:

```python
from smartmemo import ClassifierConfig, SmartMemo

cache = SmartMemo(
    domain="customer-support",
    classifier=ClassifierConfig(model_path="models/classifier-custom.pt"),
)
```

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

Version `0.1.0` is configured for PyPI as `smartmemo`. The repository publishes through
GitHub Actions trusted publishing from `.github/workflows/publish-pypi.yml` with the
`pypi` environment.

```bash
git tag v0.1.0
git push origin v0.1.0
```

That tag builds the source distribution and wheel, then uploads them to PyPI.
