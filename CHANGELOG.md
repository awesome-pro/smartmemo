# Changelog

## 0.1.0

- Ship `classifier-v1`, a pretrained generic equivalence classifier, inside the package.
- Add `ClassifierConfig.bundled()` for opt-in, zero-training classifier-gated caching.
- Add `scripts/generate_training_data.py`: a local-LLM (Ollama) paraphrase and templated
  hard-negative training-data pipeline, with a committed 10,800-pair dataset.
- Add a hand-curated 84-pair equivalence gold test set under `data/gold/`.
- Add `scripts/train_classifier_v1.py` and `benchmarks/classifier_vs_cosine.py`:
  `classifier-v1` beats the cosine baseline by +32 precision points at equal recall on
  the gold set.
- Document that the optional `[ml]` dependencies (PyTorch, FAISS, SentenceTransformers)
  are required to import smartmemo.

## 0.0.4

- Add `smartmemo retrain` for manual feedback-to-checkpoint retraining.
- Add `smartmemo.feedback.retrain_from_feedback(...)` with auditable retrain reports.
- Support optional seed data, validation gates, and checkpoint promotion when gates pass.
- Document the manual feedback loop from collection through deliberate classifier promotion.

## 0.0.3

- Add durable lookup records for cache hits so feedback can be exported later.
- Add durable good/bad feedback events tied to cache-hit query IDs.
- Add `SmartMemo.export_feedback_pairs(...)` and `smartmemo export-feedback`.
- Export feedback-derived JSONL compatible with the classifier training pipeline.

## 0.0.2

- Add optional classifier-gated cache decisions through `SmartMemo(..., classifier=...)`.
- Populate `CacheResult.classifier_score` when the classifier evaluates cache candidates.
- Preserve cosine-threshold behavior when no classifier checkpoint is configured.
- Document classifier-enabled usage and the current cold-start boundary.

## 0.0.1

Initial SmartMemo release.

- Async semantic-cache facade with SQLite persistence.
- Dependency-light hash embeddings for tests and smoke demos.
- Optional SentenceTransformers and FAISS support through the `ml` extra.
- Customer-support cosine-baseline benchmark fixture.
- Pair-classifier model, dataset, training, evaluation, and checkpoint inference utilities.
- CLI commands for cache stats, classifier training, and classifier evaluation.
- GitHub Actions CI and PyPI trusted publishing workflow.
