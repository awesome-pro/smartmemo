# Evaluation

SmartMemo optimizes for precision before recall. A false negative costs one extra LLM
call. A false positive returns the wrong cached answer, which is the production failure
mode this project is built to avoid.

Evaluate a checkpoint with:

```bash
uv run smartmemo eval-classifier \
  --data data/fixtures/customer_support_pairs.jsonl \
  --model models/classifier-v1.pt \
  --domain customer-support \
  --split test
```

The evaluator reports precision, recall, F1, accuracy, ROC AUC when both classes are
present, expected calibration error, and the confusion-matrix counts.

## The gold set and the cosine baseline

The trustworthy benchmark for the shipped classifier is `data/gold/equivalence_gold.jsonl`
— 84 hand-curated prompt pairs that are never used for training. Compare the classifier
against the cosine baseline on it with:

```bash
uv run python benchmarks/classifier_vs_cosine.py
```

Both methods are scored on the same embeddings, and precision is compared *at equal
recall* — a fair comparison, since either method can trade recall for precision by moving
its threshold. The acceptance gate for `classifier-v1` is precision at least 10 points
above the cosine baseline at equal recall; it currently clears that by +32 points. Results
are written to `benchmarks/results/classifier_vs_cosine.json`, and the shipped model card
is `smartmemo/_models/classifier-v1.report.json`.
