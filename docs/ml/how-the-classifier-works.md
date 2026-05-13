# How The Classifier Works

SmartMemo uses embedding search to find candidate cache entries, but an optional
classifier can make the final decision about whether a candidate is actually equivalent.
This matters because cosine similarity is only a neighborhood signal. Prompt pairs can
have very similar embeddings while asking for opposite actions.

The Phase 2 classifier is intentionally small. It receives two embedding vectors and
builds the standard matching representation:

```text
[embedding_a, embedding_b, abs(embedding_a - embedding_b), embedding_a * embedding_b]
```

That vector goes through a compact MLP and returns a probability in `[0, 1]`. A higher
probability means the model believes both prompts should produce the same useful response.

The model implementation lives in `src/smartmemo/classifier/model.py`. At runtime,
`SmartMemo(..., classifier=ClassifierConfig(model_path=...))` loads a checkpoint and
uses classifier scores instead of cosine thresholding for cache-hit decisions.
