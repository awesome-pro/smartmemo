# Quickstart

SmartMemo is an async-first semantic cache for LLM agent calls. The core package ships
the baseline cache layer: embeddings, top-k retrieval, SQLite persistence, and a cosine
threshold decision. If you provide a trained classifier checkpoint, SmartMemo keeps
embedding search as the fast candidate selector and uses the classifier as the actual
cache-hit decision.

Install the core package:

```bash
pip install smartmemo
```

Install real embedding and vector-search dependencies:

```bash
pip install "smartmemo[ml]"
```

Minimal use:

```python
from smartmemo import SmartMemo

cache = SmartMemo(domain="customer-support")

async def call_llm(prompt: str) -> str:
    return "fresh response from your provider"

result = await cache.get_or_call(prompt="Summarize this ticket", llm_function=call_llm)
print(result.response)
print(result.was_cache_hit)
```

Classifier-gated use:

```python
from smartmemo import ClassifierConfig, SmartMemo

cache = SmartMemo(
    domain="customer-support",
    classifier=ClassifierConfig(model_path="models/classifier-v1.pt"),
)
```

No production pretrained checkpoint is bundled yet. Train your own checkpoint with
`smartmemo train-classifier` before enabling classifier-gated decisions.

Feedback export:

```python
result = await cache.get_or_call(prompt="Summarize this ticket", llm_function=call_llm)

if result.was_cache_hit:
    await cache.report_bad_hit(result.query_id, reason="user rejected cached answer")

cache.export_feedback_pairs("data/feedback_pairs.jsonl")
```

The exported file can be passed to `smartmemo train-classifier`. This is a manual feedback
loop: SmartMemo stores and exports the training signal, but it does not retrain or deploy
new classifiers automatically yet.
