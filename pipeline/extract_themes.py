"""Zero-token tagging of episode text against config/taxonomy.json.

This is the default, always-on analysis path: pure regex/keyword matching
over each episode's title + show-notes summary. No network calls, no API
key, no per-run cost -- it can run on every episode, forever, for free.

Matching notes:
  - Every taxonomy term is compiled into a case-insensitive, word-boundary
    regex so "AI" doesn't match inside "said" and "T" (AT&T's ticker) is
    never used as a bare term.
  - Sentiment is a simple net count of bullish vs. bearish lexicon hits in
    the same text, attributed to every theme/sector/stock found in that
    episode. It's a coarse signal (episode-level, not per-mention), good
    enough for spotting rising/falling attention and directional tilt over
    time -- not a substitute for reading the episode.
  - `pipeline/enrich_claude.py` can optionally layer a real transcript +
    LLM pass on top of this for episodes worth digging into further.
"""
import json
import re
from functools import lru_cache


def _compile_terms(entries):
    """entries: list of {id, label, terms, ...}. Returns list of (entry, compiled_regex)."""
    compiled = []
    for entry in entries:
        patterns = [re.escape(t) for t in entry["terms"]]
        # Longest terms first so "electric vehicle market" wins over a
        # shorter substring before word-boundary matching even applies.
        patterns.sort(key=len, reverse=True)
        regex = re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)
        compiled.append((entry, regex))
    return compiled


class Taxonomy:
    def __init__(self, path="config/taxonomy.json"):
        with open(path) as f:
            data = json.load(f)
        self.sectors = _compile_terms(data["sectors"])
        self.themes = _compile_terms(data["themes"])
        self.stocks = _compile_terms(data["stocks"])
        self.bullish_re = self._lexicon_regex(data["sentiment_lexicon"]["bullish"])
        self.bearish_re = self._lexicon_regex(data["sentiment_lexicon"]["bearish"])

    @staticmethod
    def _lexicon_regex(words):
        patterns = sorted((re.escape(w) for w in words), key=len, reverse=True)
        return re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)

    def _matches(self, compiled, text):
        hits = []
        for entry, regex in compiled:
            if regex.search(text):
                hits.append(entry["id"])
        return hits

    def sentiment_score(self, text):
        bull = len(self.bullish_re.findall(text))
        bear = len(self.bearish_re.findall(text))
        total = bull + bear
        if total == 0:
            return 0.0
        return round((bull - bear) / total, 3)

    def tag(self, text):
        return {
            "sectors": self._matches(self.sectors, text),
            "themes": self._matches(self.themes, text),
            "stocks": self._matches(self.stocks, text),
            "sentiment": self.sentiment_score(text),
        }


@lru_cache(maxsize=1)
def _default_taxonomy():
    return Taxonomy()


def tag_episode(episode, taxonomy=None):
    """episode: dict with 'title' and 'summary'. Returns tag dict (see Taxonomy.tag)."""
    taxonomy = taxonomy or _default_taxonomy()
    text = f"{episode.get('title', '')}. {episode.get('summary', '')}"
    return taxonomy.tag(text)


if __name__ == "__main__":
    import sys

    tx = Taxonomy()
    sample = {
        "title": "Why Everyone Is Worried About a Private Credit Bubble",
        "summary": (
            "Regional banks and private credit funds have surged in importance, "
            "but concerns about a bubble in leveraged loans are growing as "
            "default risk rises and spreads widen."
        ),
    }
    print(json.dumps(tag_episode(sample, tx), indent=2))
