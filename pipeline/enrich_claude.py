"""Claude-powered analysis of episode transcripts.

The default pipeline (fetch_feeds + extract_themes) is a zero-token keyword
tagger -- reliable and free, but blunt: it can't catch sarcasm, a hedged
view, a guest disagreeing with the host, or a theme discussed without ever
using its exact keyword.

This module is the "spend tokens, get it right" path: it sends the full
transcript to Claude and asks it to tag the episode against the same
taxonomy (so results stay compatible with the dashboard's filters), plus a
free-text summary an LLM can do that a regex can't.

It runs automatically wherever ANTHROPIC_API_KEY is set (see
pipeline/transcribe.py) -- quality over cost, by explicit choice: use the
best available model rather than the cheapest one that mostly works. The
zero-token keyword tagger remains the automatic fallback only when no key
is configured, so the pipeline never hard-fails for lack of a key.
"""
import json
import os

DEFAULT_MODEL = os.environ.get("PODCAST_MONITOR_LLM_MODEL", "claude-opus-5")
DEFAULT_EFFORT = os.environ.get("PODCAST_MONITOR_LLM_EFFORT", "high")

# Real transcripts here run 2-4k words (~15-25k chars). This cap is a safety
# ceiling, not a normal-case truncation -- Claude Opus 5's 1M-token context
# has ample room for full transcripts.
MAX_INPUT_CHARS = 300_000

SYSTEM_PROMPT = """You are a financial research analyst tagging podcast episodes for an \
investment-research dashboard. You read the full transcript and identify what was \
actually discussed and how it was framed -- catching hedged views, sarcasm, and a guest \
disagreeing with the host, which keyword matching cannot."""

PROMPT_TEMPLATE = """Episode: {title}
Podcast: {podcast_name}

Tag this episode using ONLY ids from the allowed lists below (omit anything not \
genuinely discussed; do not force matches). If something clearly investment-relevant \
came up that isn't covered by the taxonomy, note it in `new_entities` instead of \
inventing an id.

Allowed sector ids: {sector_ids}
Allowed theme ids: {theme_ids}
Allowed stock ids: {stock_ids}

Transcript:
---
{text}
---
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sectors": {"type": "array", "items": {"type": "string"}},
        "themes": {"type": "array", "items": {"type": "string"}},
        "stocks": {"type": "array", "items": {"type": "string"}},
        "sentiment": {
            "type": "number",
            "description": "-1.0 (bearish) to 1.0 (bullish), overall investment-relevant tone",
        },
        "summary": {
            "type": "string",
            "description": "One or two sentences on the investment-relevant takeaway",
        },
        "new_entities": {
            "type": "array",
            "description": "Genuinely relevant themes/sectors/stocks not in the allowed lists",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["sector", "theme", "stock"]},
                    "label": {"type": "string"},
                },
                "required": ["kind", "label"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sectors", "themes", "stocks", "sentiment", "summary", "new_entities"],
    "additionalProperties": False,
}


def enrich_episode(episode, text, taxonomy_raw, model=None, effort=None):
    """Tag `text` (a full transcript) against `taxonomy_raw` (parsed
    config/taxonomy.json) using Claude. Returns a dict shaped like
    extract_themes.Taxonomy.tag()'s output, plus 'summary' and
    'new_entities'. IDs not present in the taxonomy are dropped defensively
    so dashboard filters never break on a hallucinated id.

    Raises ImportError/RuntimeError if the anthropic package or API key is
    missing -- this should only be called when both are confirmed present.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "pip install anthropic to use pipeline/enrich_claude.py"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)

    sector_ids = [e["id"] for e in taxonomy_raw["sectors"]]
    theme_ids = [e["id"] for e in taxonomy_raw["themes"]]
    stock_ids = [e["id"] for e in taxonomy_raw["stocks"]]

    prompt = PROMPT_TEMPLATE.format(
        title=episode.get("title", ""),
        podcast_name=episode.get("podcast_name", ""),
        sector_ids=", ".join(sector_ids),
        theme_ids=", ".join(theme_ids),
        stock_ids=", ".join(stock_ids),
        text=text[:MAX_INPUT_CHARS],
    )

    response = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={
            "effort": effort or DEFAULT_EFFORT,
            "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
        },
        messages=[{"role": "user", "content": prompt}],
    )

    if response.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined to analyze this episode: {response.stop_details}")

    raw_text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(raw_text)

    valid_sectors = set(sector_ids)
    valid_themes = set(theme_ids)
    valid_stocks = set(stock_ids)

    return {
        "sectors": [s for s in parsed["sectors"] if s in valid_sectors],
        "themes": [t for t in parsed["themes"] if t in valid_themes],
        "stocks": [s for s in parsed["stocks"] if s in valid_stocks],
        "sentiment": round(float(parsed["sentiment"]), 3),
        "summary": parsed.get("summary", ""),
        "new_entities": parsed.get("new_entities", []),
    }
