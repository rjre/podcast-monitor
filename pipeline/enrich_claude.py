"""Optional Claude-powered enrichment layer -- OFF by default.

The default pipeline (fetch_feeds + extract_themes) costs zero tokens and
runs on title/show-notes text alone. This module is the deliberate
"spend a little to see further" upgrade for when show notes are too thin
to tag reliably, or when you have a full transcript and want nuance the
keyword matcher can't get (sarcasm, hedged views, a guest disagreeing
with the host, etc).

It is never called automatically. Wire it in explicitly, e.g.:

    if os.environ.get("PODCAST_MONITOR_ENABLE_LLM") == "1":
        enrichment = enrich_episode(episode, transcript_text)

Cost control:
  - Uses Haiku (cheapest current model) by default -- override via
    PODCAST_MONITOR_LLM_MODEL if you want more nuance for a subset of runs.
  - Truncates input text to keep each call small; this is meant to
    complement the free tagging pass, not replace it with a full-transcript
    read on every episode.
  - Nothing in pipeline/run.py imports this module by default, so simply
    not setting ANTHROPIC_API_KEY keeps the whole app at zero API cost.
"""
import json
import os

DEFAULT_MODEL = os.environ.get("PODCAST_MONITOR_LLM_MODEL", "claude-haiku-4-5-20251001")
MAX_INPUT_CHARS = 6000

PROMPT_TEMPLATE = """You are tagging a finance podcast episode for a research dashboard.
Return ONLY valid JSON matching this schema, no prose:

{{
  "themes": [string],      // short theme/topic labels, e.g. "private credit stress"
  "sectors": [string],     // GICS-style sector labels
  "stocks": [string],      // company names or tickers explicitly discussed
  "sentiment": number,     // -1.0 (bearish) to 1.0 (bullish) overall tone re: markets
  "summary": string        // one sentence on the episode's investment-relevant takeaway
}}

Episode title: {title}
Podcast: {podcast_name}

Text (transcript or show notes, may be truncated):
---
{text}
---
"""


def enrich_episode(episode, text, model=None):
    """Call Claude to extract richer tags from `text` (transcript or notes).

    Requires the `anthropic` package and ANTHROPIC_API_KEY. Raises
    ImportError/RuntimeError clearly if either is missing rather than
    silently no-op'ing, since this should only ever be invoked
    deliberately.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "pip install anthropic to use pipeline/enrich_claude.py"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; enrichment is opt-in and needs a key")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = PROMPT_TEMPLATE.format(
        title=episode.get("title", ""),
        podcast_name=episode.get("podcast_name", ""),
        text=text[:MAX_INPUT_CHARS],
    )

    response = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text
    return json.loads(raw)
