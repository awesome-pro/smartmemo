"""Video-ready SmartMemo product demo.

This script shows the core production use case:

1. A normal cosine-only semantic cache finds a near match and reuses the wrong
   answer.
2. SmartMemo uses the same candidate search, but a learned classifier blocks
   unsafe reuse.
3. SmartMemo still preserves safe reuse for true paraphrases.
4. A bad cache hit can be exported as JSONL feedback for retraining.

Run from the repository root:

    uv run --all-extras python demo.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from decimal import Decimal
from pathlib import Path

from smartmemo import CacheConfig, ClassifierConfig, SmartMemo
from smartmemo.exceptions import MissingDependencyError
from smartmemo.models import CacheResult

ARTIFACT_DIR = Path("data/generated/demo")
BASELINE_DB = ARTIFACT_DIR / "cosine-only.db"
SMARTMEMO_DB = ARTIFACT_DIR / "smartmemo.db"
FEEDBACK_PATH = ARTIFACT_DIR / "feedback_pairs.jsonl"

SEED_PROMPT = "Change the staging config to enable debug logging."
PARAPHRASE_PROMPT = "Update the staging configuration to allow debug logging."
OPPOSITE_PROMPT = "Revise the staging configuration so debug logging is turned off."


class ConfigAgent:
    """Small deterministic stand-in for an expensive code-agent LLM call."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, prompt: str) -> str:
        self.calls += 1
        lowered = prompt.lower()
        if "off" in lowered or "disable" in lowered or "disabled" in lowered:
            return (
                "Set DEBUG_LOGGING=false in staging.yaml and restart the worker "
                "deployment so verbose request logs are disabled."
            )
        return (
            "Set DEBUG_LOGGING=true in staging.yaml and restart the worker "
            "deployment so verbose request logs are enabled."
        )


async def main() -> None:
    configure_demo_environment()
    reset_artifacts()
    print_header()

    baseline_agent = ConfigAgent()
    smartmemo_agent = ConfigAgent()

    config = CacheConfig(
        db_path=BASELINE_DB,
        cosine_threshold=0.84,
        classifier_threshold=0.95,
        estimated_llm_cost_usd=Decimal("0.004"),
    )
    smart_config = config.model_copy(update={"db_path": SMARTMEMO_DB})

    async with (
        SmartMemo(
            domain="software-engineering",
            config=config,
        ) as cosine_only,
        SmartMemo(
            domain="software-engineering",
            config=smart_config,
            classifier=ClassifierConfig.bundled(threshold=0.95),
        ) as smartmemo,
    ):
        await prime_cache(cosine_only, baseline_agent)
        await prime_cache(smartmemo, smartmemo_agent)

        print_section("1. Unsafe near match")
        print("Cached request:")
        print(f"  {SEED_PROMPT}")
        print("New request:")
        print(f"  {OPPOSITE_PROMPT}")
        print()

        baseline_bad = await cosine_only.get_or_call(
            prompt=OPPOSITE_PROMPT,
            llm_function=baseline_agent,
            model="config-agent-v1",
        )
        smartmemo_blocked = await smartmemo.get_or_call(
            prompt=OPPOSITE_PROMPT,
            llm_function=smartmemo_agent,
            model="config-agent-v1",
        )

        print_result(
            "Cosine-only cache",
            baseline_bad,
            expected="fresh disable-debug response",
            llm_calls=baseline_agent.calls,
        )
        print_result(
            "SmartMemo classifier gate",
            smartmemo_blocked,
            expected="fresh disable-debug response",
            llm_calls=smartmemo_agent.calls,
        )

        if baseline_bad.was_cache_hit:
            print("Takeaway:")
            print(
                "  The threshold-only cache reused the enable-debug response for a "
                "disable-debug request. That is the false-positive failure "
                "SmartMemo is designed to prevent."
            )
        print()

        print_section("2. Safe paraphrase")
        print("New request:")
        print(f"  {PARAPHRASE_PROMPT}")
        print()

        smartmemo_hit = await smartmemo.get_or_call(
            prompt=PARAPHRASE_PROMPT,
            llm_function=smartmemo_agent,
            model="config-agent-v1",
        )
        print_result(
            "SmartMemo classifier gate",
            smartmemo_hit,
            expected="cached enable-debug response",
            llm_calls=smartmemo_agent.calls,
        )

        if smartmemo_hit.was_cache_hit:
            await smartmemo.report_good_hit(smartmemo_hit.query_id)
            print("Takeaway:")
            print(
                "  SmartMemo is not just blocking cache hits. It preserves useful "
                "reuse when the prompts are truly equivalent."
            )
        print()

        print_section("3. Feedback export")
        if baseline_bad.was_cache_hit:
            await cosine_only.report_bad_hit(
                baseline_bad.query_id,
                reason=(
                    "cosine-only cache reused an enable-debug answer for a "
                    "disable-debug request"
                ),
            )
        exported = cosine_only.export_feedback_pairs(FEEDBACK_PATH, split="train")
        print(f"Exported feedback pairs: {exported}")
        print(f"Feedback JSONL path: {FEEDBACK_PATH}")
        print_feedback_preview(FEEDBACK_PATH)

        print_section("4. Runtime stats")
        print("Cosine-only cache:")
        print_stats(cosine_only)
        print("SmartMemo:")
        print_stats(smartmemo)

    print_section("Demo complete")
    print("What this proves:")
    print("  - Embedding similarity is useful for candidate search.")
    print("  - It is not safe enough to decide semantic equivalence alone.")
    print("  - SmartMemo adds a learned classifier gate before cache reuse.")
    print("  - Feedback can be exported for classifier retraining.")


async def prime_cache(cache: SmartMemo, agent: ConfigAgent) -> None:
    await cache.get_or_call(
        prompt=SEED_PROMPT,
        llm_function=agent,
        model="config-agent-v1",
    )


def reset_artifacts() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (BASELINE_DB, SMARTMEMO_DB, FEEDBACK_PATH):
        path.unlink(missing_ok=True)


def configure_demo_environment() -> None:
    """Keep terminal output focused during screen recording."""

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


def print_header() -> None:
    print()
    print("=" * 78)
    print("SmartMemo demo: semantic caching with a learned safety gate")
    print("=" * 78)
    print("Use case: a code assistant wants cheaper repeated LLM calls,")
    print("but cannot risk reusing a config change when the user's intent flips.")
    print()


def print_section(title: str) -> None:
    print("-" * 78)
    print(title)
    print("-" * 78)


def print_result(
    label: str,
    result: CacheResult,
    *,
    expected: str,
    llm_calls: int,
) -> None:
    source = "CACHE HIT" if result.was_cache_hit else "CACHE MISS -> LLM CALLED"
    print(label)
    print(f"  decision:          {source}")
    print(f"  expected:          {expected}")
    print(f"  cosine score:      {format_score(result.similarity_score)}")
    print(f"  classifier score:  {format_score(result.classifier_score)}")
    print(f"  cost saved:        ${result.cost_saved_usd}")
    print(f"  config LLM calls:  {llm_calls}")
    print("  returned:")
    print(f"    {result.response}")
    print()


def print_feedback_preview(path: Path) -> None:
    if not path.exists() or not path.read_text().strip():
        print("No feedback rows were written.")
        print()
        return

    record = json.loads(path.read_text().splitlines()[0])
    metadata = record["metadata"]
    print("First feedback row:")
    print(f"  label:             {record['label']} (0 means bad cache hit)")
    print(f"  query prompt:      {record['prompt_a']}")
    print(f"  cached prompt:     {record['prompt_b']}")
    print(f"  similarity score:  {format_score(metadata.get('similarity_score'))}")
    print(f"  reason:            {metadata.get('reason')}")
    print()


def print_stats(cache: SmartMemo) -> None:
    stats = cache.stats()
    print(f"  entries:      {stats.total_entries}")
    print(f"  lookups:      {stats.total_lookups}")
    print(f"  hits:         {stats.cache_hits}")
    print(f"  misses:       {stats.cache_misses}")
    print(f"  hit rate:     {stats.hit_rate:.0%}")
    print(f"  saved:        ${stats.total_cost_saved_usd}")
    print()


def format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except MissingDependencyError as exc:
        print("SmartMemo ML dependencies are required for this demo.")
        print("Run: uv run --all-extras python demo.py")
        print(f"Details: {exc}")
