# Lessons

Stuff that didn't work and what we did about it. Roughly in the order
it happened, because most of the bad decisions came from not
understanding the last one.

## The LLM intent router that ate 114 seconds

First version of intent decomposition was an LLM call — Gemma4:31b-cloud,
Pydantic-validated JSON, 3 retries on schema failure with the validation
error prepended. Looked clean on paper.

In practice, Gemma4 would sometimes emit JSON that hit `max_tokens=512`
mid-object. Truncated. Validator failed. Retry. Same truncation. Retry
again. By the time the hard fallback kicked in, ~114 seconds had gone by
with nothing on screen except a spinner.

The retries were the problem. We'd treated them like a cheap safety net,
but each one was a 30+ second round-trip. And when the first attempt fails
with a specific failure mode (truncation), the retry almost always hits
the same wall. One clean try with a sensible fallback beats three slow
tries that all die the same way.

We briefly eyed `response_format=json_schema` as a fix but Ollama Cloud
doesn't expose it yet.

## So we ripped out the LLM. Then we broke multi-intent.

Second pass: kill the LLM, use keyword matching on the whole query. Fast,
deterministic, done.

Except "how are you and what is the capital of France?" now became a
single PERSONAL sub-intent. The whole point of decomposition — routing
the two halves to different pools — was gone. We'd deleted the feature
to make it fast.

Speed isn't the only axis. "Agentic" here means splitting the query
into typed sub-queries *and* routing them; not just "an LLM is involved
somewhere." A faster solution that doesn't do what the spec asks is
not a solution.

## What actually worked: split + zero-shot BGE

Regex-split the query on `and` / `but` / punctuation, classify each
fragment via cosine similarity against 5 seed sentences per class using
the BGE embedder we already had loaded for retrieval.

No LLM, no retries, no new dependencies. Median latency went from 114s
to ~30ms on the same input. All three intents routed correctly.

Moral of the story: the cheapest classifier that works is almost always
the right one for a prototype. We were already paying for BGE; using it
for classification too cost us nothing.

## The classifier over-matched CONTEXTUAL

Live test, turn 11 of a Forrest Gump session. Partner: "give me a
detailed introduction." Classifier: `CONTEXTUAL`. Retrieval searched
session history, found three weak matches, grounded the LLM prompt in
basically nothing about Forrest. The LLM flailed, guardrail caught
something, user got "I don't know."

Problem was that CONTEXTUAL exemplars like *"what were we talking
about"* cast too wide a net — any meta-shaped question slid into that
bucket. A single threshold (`> 0.35`) didn't guard against it.

Fix used two extra signals: CONTEXTUAL has to beat the runner-up by a
margin (`0.08`), *and* the fragment has to contain an actual discourse
word (earlier, mentioned, just, repeat, said) matched at word
boundaries. Low-confidence goes to PERSONAL, not OPEN_DOMAIN — safer
fallback for a persona bot.

One-dimensional thresholds are a weak guard. Adding a margin signal
and a structural word cue made wrong classifications much harder
without changing what the happy path does.

## CONTEXTUAL was fighting personal grounding

Originally we had CONTEXTUAL as a *replacement* for PERSONAL — "this
turn is about what we just said, so search history instead of
memories." Wrong. Even when the user asks about prior conversation,
the response still needs to sound like the persona. Session history
is extra context, not a source of truth.

Now CONTEXTUAL always pulls persona memory first, then layers on
relevant history (score ≥ 0.5). Never an empty personal prompt.

Think about where the LLM's source of truth is. For a persona bot,
that's the persona's memories, every time. Other signals go on top.

## Gemma4 started writing the character brief as output

Early testing, `THINKING_MODE=off`. Partner: "hi". LLM:

```
The user wants me to roleplay as Abed Nadir from Community.
Key characteristics:
- Autism spectrum (canonically coded, not explicitly diagnosed)
- Occasional selective mutism during sensory overload
...
```

It was writing the brief, not being the character. Our prompt
front-loaded Abed's condition and voice, then asked for a response at
the bottom. Gemma4 treated the whole thing as a writing assignment —
summarize first, respond second. We'd accidentally written a creative
writing prompt.

Two fixes stacked:

1. Anti-meta rules at the top *and* bottom of the prompt: "never
   narrate, analyze, describe, or list traits. Never say 'As an AI',
   'The user wants me to', 'Key characteristics'..." Models weight
   the start and end of a prompt more than the middle; saying the
   rule in both spots is cheap.
2. `THINKING_MODE=suppress` in `.env`. Ollama supports a `/no_think`
   prefix on the user message; this turns it on. Gemma4 stopped
   emitting the scratchpad entirely.

Instruction-tuned models will follow *whatever instruction looks most
like the task*. If your prompt looks like a character brief, the
model may complete the brief. State "do not narrate" explicitly, and
use the no-think flag when the model supports it.

## The guardrail saved us once

After the prompt and `/no_think` fixes, a test run *still* leaked —
`"The user wants me to roleplay as Raymond..."` came back from the LLM.
But the user never saw it. The output guardrail caught the phrase and
swapped in the safe fallback.

Belt-and-braces output checks are worth the effort. When the prompt
was wrong, the guardrail was still right.

## Open-domain tempted us to build a web search

First instinct when we added OPEN_DOMAIN was "great, now we need a
web search adapter." But the product isn't a search engine — it's an
AAC user's voice. If someone asks Mia for the capital of France, the
answer is "Paris" in her voice. The LLM already knows basic facts;
Mia's persona is the scarce thing. Piping in a Wikipedia snippet
would dilute her voice, not enrich it.

So OPEN_DOMAIN just emits a stub chunk that tells the LLM to answer
from its own knowledge. Cheap, aligned with the product, one less
thing to break.

When you see a retrieval-shaped problem, don't assume a retriever
is the right answer.

## Caching contextual embeddings was a waste of thought

At one point we worried about `retrieve_from_history` re-encoding the
session window every turn. Measured it: 43ms even with 80 turns of
history. The LLM call was taking 1.5–95 seconds. Shaving 30ms off a
20-second turn is 0.1%, invisible.

Measure before you optimize, even when the waste seems obvious.

## Monolithic prompts don't cache

Our planner built one giant user message with the character sheet
(stable per persona) and the retrieved chunks (different every turn)
mashed together. Prompt caches match prefixes exactly — one byte
change in the retrieval block invalidates the whole prompt, including
~300 tokens of character sheet that hadn't changed.

Split it: system message holds the stable character sheet and
answering rules, user message holds the per-turn retrieval and query.
Provider caches the system prefix → every turn after the first skips
prefill on ~300 tokens.

Structure matters as much as content once you care about latency.
Stable stuff goes in the system message, per-turn stuff goes in the
user message. Costs nothing in capability, compounds across turns.

---

## Principles we kept circling back to

**Measure first.** Every good decision here was triggered by a number
— 114s, 43ms, 30ms. Every bad decision was triggered by a hunch.

**Three pools because we have three sources.** Not four, not five. A
category without a real retriever behind it just confuses the
classifier.

**Short prompts behave better.** Every time we trimmed something,
the model was more consistent.

**Every path produces at least one chunk.** Empty retrieval blocks
were the fastest route to a hallucination. CONTEXTUAL with no
history, OPEN_DOMAIN with nothing wired up, a classifier returning
an empty sub-intent list — all have fallbacks now.
