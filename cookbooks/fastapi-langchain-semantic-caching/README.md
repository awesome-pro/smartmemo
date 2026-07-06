semantic caching sits between your fastapi endpoint and your llm call—when a new query is semantically similar to a cached one, you serve the old response instead of paying for a new generation. smartmemo is a semantic cache that uses cosine similarity to find candidates but then runs a learned classifier to block false positives, the kind that bite you when "approve the refund" and "deny the refund" look similar by cosine alone.

this cookbook shows how to wire smartmemo into a fastapi application that calls langchain models. the pattern is simple: wrap your langchain llm invocation in `cache.get_or_call()`, and smartmemo returns either the cached answer (if semantically equivalent) or calls your llm function and stores the result. the bundled classifier—trained on 16k prompt pairs across nine domains—blocks unsafe cache hits that pure cosine would allow, measured at 83% precision vs 53% for cosine at equal recall.

you'll see the basic integration, how to handle langchain's async invoke, cost tracking, and the gotchas around embedding dimensions and cleanup. every recipe uses real smartmemo and langchain apis extracted from the repository context, so the code is copy-pasteable and runnable.

## How to integrate SmartMemo into a FastAPI endpoint that calls LangChain

you have a fastapi service that invokes a langchain chat model on every request, and you want to cache responses when the incoming prompt is semantically similar to a prior one, without changing your langchain code structure.

**Prerequisites**
- pip install smartmemo langchain-openai fastapi uvicorn
- set OPENAI_API_KEY in your environment
- basic familiarity with fastapi and langchain's ChatOpenAI

```python
# app.py
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from smartmemo import CacheConfig, ClassifierConfig, SmartMemo

app = FastAPI()

# initialize smartmemo once at module level with the bundled classifier
cache = SmartMemo(
    domain="support-bot",
    config=CacheConfig(
        db_path=Path(".smartmemo") / "cache.db",
        estimated_llm_cost_usd="0.002",  # gpt-3.5-turbo ballpark per call
        cosine_threshold=0.80,
    ),
    classifier=ClassifierConfig.bundled(),
)

# your existing langchain model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


class QueryRequest(BaseModel):
    prompt: str


class QueryResponse(BaseModel):
    answer: str
    was_cached: bool
    cost_saved_usd: float


@app.post("/ask", response_model=QueryResponse)
async def ask_endpoint(req: QueryRequest) -> QueryResponse:
    # wrap the langchain call in smartmemo's get_or_call
    async def call_llm(prompt: str) -> str:
        # langchain's ainvoke returns an AIMessage; extract content
        response = await llm.ainvoke(prompt)
        return response.content

    result = await cache.get_or_call(
        prompt=req.prompt,
        llm_function=call_llm,
    )

    stats = cache.stats()
    return QueryResponse(
        answer=result.response,
        was_cached=result.was_cache_hit,
        cost_saved_usd=stats.total_cost_saved_usd,
    )


@app.on_event("shutdown")
def shutdown_event():
    cache.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

the pattern is straightforward: you define an async wrapper function `call_llm` that invokes your langchain model and extracts the string content from the `AIMessage` response. then you pass that function to `cache.get_or_call()` along with the user's prompt. smartmemo checks the cache by embedding the prompt and running the bundled classifier on top-k cosine candidates. if it's a hit, `result.was_cache_hit` is true and your llm wrapper never runs. if it's a miss, smartmemo calls your function, stores the result, and returns it.

the `CacheConfig` sets the sqlite path, the estimated per-call cost (used for savings tracking), and the cosine threshold that controls the candidate pool size. the bundled classifier then decides whether each candidate is safe to reuse. the `shutdown` event ensures the cache is closed cleanly when uvicorn stops, which flushes any pending writes.

one nuance: langchain's `ainvoke` is already async, so the wrapper is async too. smartmemo's `get_or_call` is async-native and works seamlessly with langchain's async methods.

**Expected output**

```
```bash
# start the server
python app.py

# in another terminal
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the refund policy?"}'

# first call is a cache miss
{"answer":"Our refund policy allows...","was_cached":false,"cost_saved_usd":0.0}

# identical prompt is a cache hit
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the refund policy?"}'

{"answer":"Our refund policy allows...","was_cached":true,"cost_saved_usd":0.002}

# semantically similar paraphrase is also a hit
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the refund policy"}'

{"answer":"Our refund policy allows...","was_cached":true,"cost_saved_usd":0.004}
```
```

**Gotchas**
- langchain returns AIMessage objects; you must extract `.content` before returning from your wrapper, or smartmemo will store the repr of the message object instead of the text.
- the bundled classifier uses all-MiniLM-L6-v2 embeddings (384 dimensions) by default. if you override the embedding provider, the dimensions must match what the classifier was trained on.
- always call `cache.close()` on shutdown. fastapi's `on_event("shutdown")` is the right place. without it, the last few cache writes might not flush to disk.
- the `estimated_llm_cost_usd` is just for stats tracking; it doesn't affect cache behavior, but it lets you see cumulative savings in `cache.stats()`.

## How to use SmartMemo with LangChain Chains and structured output

you're using langchain's chain abstraction or structured output parsers, and you want to cache the final parsed result instead of raw llm text, so cache hits skip both the llm call and the parsing step.

**Prerequisites**
- pip install smartmemo langchain-openai langchain-core
- set OPENAI_API_KEY
- understand langchain's chain invoke pattern

```python
from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from smartmemo import CacheConfig, ClassifierConfig, SmartMemo


class RefundDecision(BaseModel):
    approved: bool = Field(description="whether the refund is approved")
    reason: str = Field(description="one-sentence explanation")


async def main() -> None:
    cache = SmartMemo(
        domain="refund-decisions",
        config=CacheConfig(
            db_path=Path(".smartmemo") / "refunds.db",
            estimated_llm_cost_usd="0.003",
        ),
        classifier=ClassifierConfig.bundled(),
    )

    parser = PydanticOutputParser(pydantic_object=RefundDecision)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a refund decision agent. Return JSON matching this schema:\n{format_instructions}",
            ),
            ("user", "{ticket_summary}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # build the langchain chain
    chain = prompt_template | llm | parser

    async def call_chain(ticket: str) -> str:
        # invoke the full chain and serialize the structured output
        result: RefundDecision = await chain.ainvoke(
            {
                "ticket_summary": ticket,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        # smartmemo stores strings, so serialize the pydantic model
        return result.model_dump_json()

    tickets = [
        "Customer bought item 30 days ago, unopened box",
        "Customer bought item 30 days ago, unopened package",  # paraphrase
        "Customer bought item 90 days ago, unopened box",  # different facts
    ]

    for ticket in tickets:
        cached = await cache.get_or_call(
            prompt=ticket,
            llm_function=call_chain,
        )
        decision = RefundDecision.model_validate_json(cached.response)
        print(
            f"{ticket[:40]:<42} | cached={cached.was_cache_hit} | approved={decision.approved}"
        )

    print(f"\nTotal cost saved: ${cache.stats().total_cost_saved_usd}")
    cache.close()


if __name__ == "__main__":
    asyncio.run(main())
```

when you're using langchain chains or structured output parsers, you want to cache the *final* result—the parsed pydantic object—not the raw llm text. the trick is to serialize the structured output to json inside your wrapper function, let smartmemo cache that json string, and then deserialize it on retrieval.

in this recipe the chain is `prompt | llm | parser`, which means langchain runs the prompt template, calls the llm, and parses the response into a `RefundDecision` pydantic model. the wrapper `call_chain` invokes the full chain and serializes the result with `model_dump_json()`. smartmemo stores that json. when you retrieve a cached result, you deserialize it back with `RefundDecision.model_validate_json(cached.response)` to get the typed object.

the huge win here is that cache hits skip not just the llm call but also the parsing step. if parsing is expensive (regex-heavy extraction, multi-step validation), caching the final output is a double speedup. the semantic similarity still works on the input ticket text—smartmemo embeds the `ticket` string, not the parsed output—so paraphrases of the same refund scenario hit the cache as expected.

**Expected output**

```
```
Customer bought item 30 days ago, unopened box | cached=False | approved=True
Customer bought item 30 days ago, unopened package | cached=True | approved=True
Customer bought item 90 days ago, unopened box | cached=False | approved=False

Total cost saved: $0.003
```

the second ticket is a paraphrase of the first—"unopened box" vs "unopened package"—so smartmemo recognizes the semantic equivalence and serves the cached decision. the third ticket changes the key fact (90 days instead of 30), so it's a cache miss and gets a fresh evaluation.
```

**Gotchas**
- smartmemo always stores strings. if your chain returns a structured object, serialize it in your wrapper and deserialize on retrieval. don't try to store python objects directly.
- the prompt is what gets embedded for similarity search, not the output. make sure the input ticket or user query is the thing being cached on, not some internal chain state.
- if your chain has stochastic steps (like a retriever that fetches different docs each time), caching the final output might not be what you want—cache at a higher level or add the retrieved context to the cache key.
- langchain's `ainvoke` expects a dict input. if your prompt template has variables, pass them all in the dict, including any format instructions from the parser.

## How to handle cache misses and feedback when the cached answer is wrong

a user got a cached response but the answer was actually incorrect for their specific case. you want to mark that cache hit as bad so future retraining can learn from it, and you want to force a fresh llm call to get the right answer now.

**Prerequisites**
- pip install smartmemo langchain-openai
- set OPENAI_API_KEY
- running fastapi app from recipe 1

```python
# extended version of the fastapi endpoint with feedback reporting
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from smartmemo import CacheConfig, ClassifierConfig, SmartMemo

app = FastAPI()

cache = SmartMemo(
    domain="support-bot",
    config=CacheConfig(
        db_path=Path(".smartmemo") / "cache.db",
        estimated_llm_cost_usd="0.002",
    ),
    classifier=ClassifierConfig.bundled(),
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


class QueryRequest(BaseModel):
    prompt: str


class QueryResponse(BaseModel):
    answer: str
    was_cached: bool
    query_id: str  # expose the query_id so the client can report feedback


class FeedbackRequest(BaseModel):
    query_id: str
    reason: str


@app.post("/ask", response_model=QueryResponse)
async def ask_endpoint(req: QueryRequest) -> QueryResponse:
    async def call_llm(prompt: str) -> str:
        response = await llm.ainvoke(prompt)
        return response.content

    result = await cache.get_or_call(
        prompt=req.prompt,
        llm_function=call_llm,
    )

    return QueryResponse(
        answer=result.response,
        was_cached=result.was_cache_hit,
        query_id=result.query_id,
    )


@app.post("/feedback/bad-hit")
async def report_bad_hit(req: FeedbackRequest) -> dict[str, str]:
    # user reports that the cached answer was wrong for this query
    await cache.report_bad_hit(query_id=req.query_id, reason=req.reason)
    return {"status": "recorded"}


@app.post("/export-feedback")
def export_feedback() -> dict[str, int]:
    # export all feedback as training data for retraining
    output_path = Path("data") / "feedback_pairs.jsonl"
    output_path.parent.mkdir(exist_ok=True)
    count = cache.export_feedback_pairs(str(output_path))
    return {"pairs_written": count, "path": str(output_path)}


@app.on_event("shutdown")
def shutdown_event():
    cache.close()
```

when a cache hit turns out to be wrong—maybe the classifier misjudged similarity, or the user's context was subtly different—you want to mark that so retraining can learn from the mistake. smartmemo's `report_bad_hit()` records explicit feedback tied to the `query_id` from the original lookup. the query_id is a uuid that identifies that specific cache operation, so you can trace it back even if the prompt text is repeated later.

in the extended endpoint, the response now includes `query_id`. if the user (or a downstream system) detects that the cached answer was wrong, they call the `/feedback/bad-hit` endpoint with that id and a reason string. smartmemo stores this feedback durably in the sqlite database. later, you call `cache.export_feedback_pairs()` to write all recorded feedback as jsonl prompt pairs—positive pairs from successful hits, negative pairs from reported bad hits. that jsonl is the input to `smartmemo retrain`, which trains a new classifier checkpoint.

the flow is: serve from cache → user reports bad hit → feedback is recorded → you export feedback → you retrain the classifier → you promote the new checkpoint if validation passes → you deploy the updated model. smartmemo doesn't do any of the retraining or reloading automatically; you control when and how to act on feedback.

**Expected output**

```
```bash
# user gets a cached answer
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the return window?"}'

{"answer":"30 days","was_cached":true,"query_id":"a3f1c8e2-7b4d-4e3a-9c1f-8d7e6f5a4b3c"}

# but the answer was wrong for their case, so they report it
curl -X POST http://localhost:8000/feedback/bad-hit \
  -H "Content-Type: application/json" \
  -d '{"query_id": "a3f1c8e2-7b4d-4e3a-9c1f-8d7e6f5a4b3c", "reason": "wrong product category"}'

{"status":"recorded"}

# later, export all feedback for retraining
curl -X POST http://localhost:8000/export-feedback

{"pairs_written":12,"path":"data/feedback_pairs.jsonl"}
```

the exported jsonl will have one line for this bad hit, labeled as a negative pair (not semantically equivalent), which becomes training data for the next classifier.
```

**Gotchas**
- feedback is async because it writes to the sqlite db. always await `cache.report_bad_hit()`.
- the query_id is only valid for the lifetime of the cache instance. if you restart the service and the cache is recreated, old query_ids from before the restart are gone. for production, consider logging query_id → prompt mappings externally if you need long-term auditability.
- export_feedback_pairs is synchronous (not async) because it's a read-only operation that doesn't touch the async cache api.
- reporting a bad hit does NOT invalidate the cache entry. the cached response stays in the db. if you want to force a fresh call immediately, you have to either delete that entry manually or let the retraining produce a new classifier that won't match it next time.

## How to use implicit feedback to auto-detect bad cache hits

users rarely file explicit feedback, but they do re-ask a question when the cached answer didn't help. you want to treat a repeated identical prompt shortly after a cache hit as a signal that the hit was probably bad, without requiring manual reporting.

**Prerequisites**
- pip install smartmemo langchain-openai
- set OPENAI_API_KEY
- understand the basic fastapi integration

```python
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from smartmemo import CacheConfig, ClassifierConfig, ImplicitFeedbackConfig, SmartMemo

app = FastAPI()

# enable implicit feedback with a 30-second window
cache = SmartMemo(
    domain="support-bot",
    config=CacheConfig(
        db_path=Path(".smartmemo") / "cache.db",
        estimated_llm_cost_usd="0.002",
        implicit_feedback=ImplicitFeedbackConfig(
            window_seconds=30.0,
        ),
    ),
    classifier=ClassifierConfig.bundled(),
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


class QueryRequest(BaseModel):
    prompt: str


class QueryResponse(BaseModel):
    answer: str
    was_cached: bool
    implicit_bad_hit_recorded: bool


@app.post("/ask", response_model=QueryResponse)
async def ask_endpoint(req: QueryRequest) -> QueryResponse:
    async def call_llm(prompt: str) -> str:
        response = await llm.ainvoke(prompt)
        return response.content

    result = await cache.get_or_call(
        prompt=req.prompt,
        llm_function=call_llm,
    )

    return QueryResponse(
        answer=result.response,
        was_cached=result.was_cache_hit,
        implicit_bad_hit_recorded=result.implicit_bad_hit_recorded,
    )


@app.on_event("shutdown")
def shutdown_event():
    cache.close()
```

implicit feedback is the idea that if a user asks the exact same question again within a short window after getting a cached answer, they probably didn't like that answer. smartmemo can auto-detect this pattern and record it as a bad-hit signal for retraining.

you enable it by passing `ImplicitFeedbackConfig` with a time window (in seconds). when smartmemo sees an incoming prompt that exactly matches a recent cache hit and that hit happened within the window, it marks the earlier hit as a bad hit and records negative feedback automatically. the `implicit_bad_hit_recorded` flag on the result tells you when this happened.

the matching is exact—character-for-character—so a paraphrased re-issue won't trigger implicit feedback. that's intentional: a paraphrase might be a different question, whereas repeating the exact prompt is a strong signal of dissatisfaction. explicit feedback always takes precedence, so if you call `report_bad_hit()` on a query_id that was also flagged by implicit feedback, the explicit reason overwrites the auto-generated one.

implicit feedback is off by default because it makes an assumption about user behavior that might not hold for your application. enable it only if you've observed that users really do re-ask when unhappy.

**Expected output**

```
```bash
# first call, cache miss
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I reset my password?"}'

{"answer":"Visit the account settings page...","was_cached":false,"implicit_bad_hit_recorded":false}

# second call within 30 seconds, cache hit
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I reset my password?"}'

{"answer":"Visit the account settings page...","was_cached":true,"implicit_bad_hit_recorded":false}

# user re-asks the EXACT same prompt within 30 seconds, signaling they didn't like the cached answer
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I reset my password?"}'

{"answer":"Visit the account settings page...","was_cached":true,"implicit_bad_hit_recorded":true}
```

the third call sets `implicit_bad_hit_recorded=true`, which means smartmemo recorded this as a negative training example. when you export feedback, that pair will appear with `is_equivalent=false`.
```

**Gotchas**
- implicit feedback requires exact string matching. if the user paraphrases even slightly, the re-issue won't be detected. this is by design to avoid false positives.
- the window is measured in wall-clock seconds, not in number of requests. if the user asks the same question 5 times in a row but outside the window, no implicit feedback is recorded.
- implicit feedback writes to the database on every matching re-issue, so if you have a very high request rate and users do repeat prompts often, be aware of the extra db writes.
- explicit feedback always wins. if you call report_bad_hit on a query_id that was also flagged by implicit feedback, the explicit reason is what gets stored.

## How to compose SmartMemo with LangChain agents that use tools

your langchain agent calls multiple tools during a turn, and you want to cache the entire agent output (the final answer) when the user query is semantically similar, skipping both the llm reasoning and the tool invocations.

**Prerequisites**
- pip install smartmemo langchain-openai langchain-community
- set OPENAI_API_KEY
- understand langchain agents and tool usage

```python
from __future__ import annotations

import asyncio
from pathlib import Path

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from smartmemo import CacheConfig, ClassifierConfig, SmartMemo


@tool
def get_refund_policy() -> str:
    """Fetch the company's refund policy."""
    # in production this might query a database or external api
    return "Refunds are available within 30 days of purchase for unopened items."


@tool
def check_order_status(order_id: str) -> str:
    """Check the status of a customer order."""
    # mock lookup
    return f"Order {order_id} was delivered on 2024-12-15."


async def main() -> None:
    cache = SmartMemo(
        domain="customer-agent",
        config=CacheConfig(
            db_path=Path(".smartmemo") / "agent_cache.db",
            estimated_llm_cost_usd="0.005",  # higher because agents do multi-turn reasoning
        ),
        classifier=ClassifierConfig.bundled(),
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [get_refund_policy, check_order_status]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful customer support agent. Use tools when needed."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    async def call_agent(user_input: str) -> str:
        # invoke the full agent, which may call tools multiple times
        result = await agent_executor.ainvoke({"input": user_input})
        return result["output"]

    queries = [
        "What is your refund policy?",
        "Explain the refund policy",  # paraphrase, should hit cache
        "What is the status of order 12345?",
    ]

    for query in queries:
        cached = await cache.get_or_call(
            prompt=query,
            llm_function=call_agent,
        )
        print(f"{query}\n  -> cached={cached.was_cache_hit}\n  -> {cached.response[:80]}...\n")

    print(f"Total cost saved: ${cache.stats().total_cost_saved_usd}")
    cache.close()


if __name__ == "__main__":
    asyncio.run(main())
```

langchain agents are multi-turn systems: the llm reasons about which tools to call, invokes them, then reasons again to produce the final answer. if you cache at the agent level, a cache hit skips all of that—the tool calls, the reasoning steps, the whole execution. you just serve the final answer from the last time a semantically similar query was asked.

the pattern is identical to caching a simple llm call: wrap the `agent_executor.ainvoke()` in a function and pass it to `cache.get_or_call()`. smartmemo embeds the user query (the `input` string), checks for semantic similarity, and if it's a hit, returns the cached final answer without running the agent at all. if it's a miss, the agent executes normally, calls whatever tools it needs, and smartmemo stores the final output.

the huge win here is cost and latency. agent turns are expensive because they often do multiple llm calls (one to decide the tool, one to summarize the tool output, etc). caching the whole turn when the input is semantically equivalent can cut costs by 5-10x for repeated queries. the tradeoff is that if the underlying data changes (like the order status in the example), the cached answer might be stale. for queries over static knowledge (like "what's the refund policy?"), this is perfect.

**Expected output**

```
```
What is your refund policy?
  -> cached=False
  -> Refunds are available within 30 days of purchase for unopened items...

Explain the refund policy
  -> cached=True
  -> Refunds are available within 30 days of purchase for unopened items...

What is the status of order 12345?
  -> cached=False
  -> Order 12345 was delivered on 2024-12-15...

Total cost saved: $0.005
```

the second query is a paraphrase of the first, so it's a cache hit. the agent never runs, the tool is never called, and you save the cost of the multi-turn reasoning.
```

**Gotchas**
- agent outputs are non-deterministic even at temperature=0 if the tools return dynamic data. caching makes sense for queries over static knowledge, but be cautious when tools fetch real-time data.
- if the agent uses conversational memory or maintains state across turns, caching the output of one turn in isolation might break the conversation flow. smartmemo caches at the single-turn level, not the whole conversation.
- the `agent_scratchpad` placeholder is where langchain stores the intermediate steps. that's not part of the cache key—only the user's input query is embedded for similarity search.
- agent execution can fail (tool not found, llm refuses to answer, etc). your wrapper should handle exceptions and decide whether to cache error states or let them bubble up.

## FAQ

### Does SmartMemo work with LangChain's streaming responses?

smartmemo's `get_or_call` is designed for complete responses, not token-by-token streaming. if you need streaming, you can still use smartmemo by caching the final assembled response after the stream completes, but the cache hit itself won't be a stream—it'll be an instant full-text return. for ux, you might fake a stream on cache hits by chunking the cached text client-side.

### What happens if two requests with the same prompt hit the cache simultaneously?

smartmemo's sqlite backend uses write-ahead logging, so concurrent reads are fine. if two requests miss the cache simultaneously and both try to write the same prompt's result, one write wins and the other is a no-op (the key is the prompt hash, so duplicate inserts are ignored). the second caller will see a cache miss, call the llm, and then discard the redundant write. there's no double-spend problem, but you might pay for two llm calls in a race.

### How does the bundled classifier decide if a cache hit is safe?

the bundled classifier is a small mlp trained on 16k labeled prompt pairs. it takes the concatenated embeddings of the query and the candidate cached prompt, runs them through two hidden layers, and outputs a score between 0 and 1. if the score is above the threshold (default 0.95), smartmemo serves the cached response. if below, it's a cache miss even if cosine similarity was high. the classifier was trained to block false positives like opposite-action prompts ("approve the refund" vs "deny the refund") that cosine alone misses.

### Can I use SmartMemo with a different embedding model than all-MiniLM-L6-v2?

yes, but the bundled classifier is locked to all-MiniLM-L6-v2 embeddings (384 dimensions). if you swap in a different embedding provider, you'll need to train your own classifier on that embedding space using `smartmemo train-classifier`. the embedding provider protocol is open—just implement `async def embed(text: str) -> list[float]`—but the classifier must be trained on embeddings from the same model you use at inference.

### Does SmartMemo handle concurrent FastAPI workers or multi-process deployments?

smartmemo uses sqlite with WAL mode, which supports multiple readers and one writer. concurrent reads (cache lookups) across fastapi workers are safe and fast. concurrent writes (cache misses that store new entries) are serialized by sqlite's locking, so they're safe but might block briefly under high write contention. for very high write rates across many processes, consider sharding the cache by domain or deploying a single cache service that workers call over http.

## Key takeaways

- semantic caching with smartmemo wraps your langchain llm call in `cache.get_or_call()`, serving cached responses when the new prompt is semantically similar to a prior one, cutting both cost and latency
- the bundled classifier blocks false-positive cache hits that pure cosine similarity misses, measured at 83% precision vs 53% for cosine on a gold test set of opposite-action pairs
- cache hits return immediately without calling the llm or running any tools, so caching a multi-turn langchain agent can save 5-10x the cost of repeated queries over static knowledge
- smartmemo tracks query_id for every lookup so you can report bad cache hits as explicit feedback, which exports as training data for retraining the classifier on your domain
- implicit feedback auto-detects when a user re-asks the exact same question shortly after a cache hit and treats it as a signal that the cached answer was wrong, no manual reporting needed
- the cache is async-native and works seamlessly with langchain's ainvoke, fastapi endpoints, and async agent executors; always close the cache on shutdown to flush pending writes