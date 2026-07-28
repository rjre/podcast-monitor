"""Zero-token tagging of episode text against config/taxonomy.json.

This is the default, always-on analysis path: pure regex/keyword matching
over each episode's title + show-notes summary (or, once transcribed, the
full transcript). No network calls, no API key, no per-run cost -- it can
run on every episode, forever, for free.

Matching notes:
  - Every taxonomy term is compiled into a case-insensitive, word-boundary
    regex so "AI" doesn't match inside "said" and "T" (AT&T's ticker) is
    never used as a bare term.
  - Each matched entity also gets a mention count (how many times its terms
    hit the text) -- a stronger salience/intensity signal than the plain
    presence/absence used previously, and what pipeline/run.py rolls up
    into aggregate "buzz" figures.
  - Sentiment has two layers:
      1. Overall episode sentiment: net bullish vs. bearish lexicon hits,
         same as before, but shrunk toward neutral when the hit count is
         low -- a single incidental bearish word in a 5,000-word transcript
         should not read as a full-strength -1.0. `sentiment_hits` and
         `sentiment_confidence` are exposed so downstream consumers can see
         how much signal the score is actually based on.
      2. Per-entity local sentiment: bullish/bearish words in the sentences
         immediately around each entity's mentions, not the episode as a
         whole -- so an episode that's bearish on regional banks but
         bullish on AI infrastructure spending doesn't get flattened into
         one misleading average.
  - Beyond tone, three more signals come out of that same per-entity local
    window (see config/taxonomy.json's conviction_lexicon/action_lexicon):
      - entity_conviction: confident ("no doubt", "high conviction") vs.
        hedged ("I think", "hard to say") language -- independent of
        whether the tone was bullish or bearish, a hedged take reads very
        differently from a confident one.
      - entity_stance: actual buy/sell recommendation language ("I'd buy
        this", "I'm avoiding it") -- a different axis from tone; someone
        can sound upbeat about a stock while still saying they wouldn't
        buy it here.
      - entity_contested: flags an entity that got *meaningful* bullish
        AND bearish language in the same episode -- a net sentiment near
        0.0 is otherwise ambiguous between "nobody discussed tone" and
        "views clashed and canceled out".
  - entity_mention_density normalizes entity_mentions by transcript length
    (mentions per 1,000 words), so a stock mentioned 3 times in a 5-minute
    segment isn't scored as equally salient as one mentioned 3 times across
    a 90-minute episode.
  - `pipeline/enrich_claude.py` (or the manual-review workflow) can
    optionally layer a real transcript + LLM/human pass on top of this for
    episodes worth digging into further.
"""
import json
import re
from collections import defaultdict
from functools import lru_cache

# Below this many total bull/bear lexicon hits, the raw bull/bear ratio is
# too noisy to trust at face value: shrink it toward neutral in proportion
# to how few hits it's based on, instead of reporting a full-strength
# +/-1.0 off a single incidental word.
MIN_CONFIDENT_HITS = 6

# Sentences within this many positions of a mention count as "near" it, for
# per-entity (rather than whole-episode) sentiment/conviction/stance.
LOCAL_WINDOW = 1

# An entity needs at least this many bullish AND this many bearish hits in
# its own local window to count as genuinely "contested" -- a stray 1-1
# isn't a real clash of views, just noise.
CONTESTED_MIN_HITS = 2

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text):
    return [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


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
        self.confident_re = self._lexicon_regex(data["conviction_lexicon"]["confident"])
        self.hedged_re = self._lexicon_regex(data["conviction_lexicon"]["hedged"])
        self.buy_re = self._lexicon_regex(data["action_lexicon"]["buy_signals"])
        self.sell_re = self._lexicon_regex(data["action_lexicon"]["sell_signals"])

    @staticmethod
    def _lexicon_regex(words):
        patterns = sorted((re.escape(w) for w in words), key=len, reverse=True)
        return re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)

    @staticmethod
    def _mention_counts(compiled, text):
        """entity_id -> number of term hits in the whole text."""
        counts = {}
        for entry, regex in compiled:
            n = len(regex.findall(text))
            if n:
                counts[entry["id"]] = n
        return counts

    @staticmethod
    def _sentence_hits(compiled, sentences):
        """entity_id -> list of sentence indices where it was mentioned."""
        positions = defaultdict(list)
        for idx, sentence in enumerate(sentences):
            for entry, regex in compiled:
                if regex.search(sentence):
                    positions[entry["id"]].append(idx)
        return positions

    @staticmethod
    def _window_text(sentences, indices):
        """Concatenated text of every sentence within LOCAL_WINDOW of any of
        `indices`, deduped -- the shared context all per-entity signals
        (sentiment, conviction, stance) are drawn from."""
        seen = set()
        parts = []
        for idx in indices:
            lo, hi = max(0, idx - LOCAL_WINDOW), min(len(sentences) - 1, idx + LOCAL_WINDOW)
            for j in range(lo, hi + 1):
                if j not in seen:
                    seen.add(j)
                    parts.append(sentences[j])
        return " ".join(parts)

    def _window_signals(self, window_text):
        """Raw lexicon hit counts in one entity's local window."""
        return {
            "bull": len(self.bullish_re.findall(window_text)),
            "bear": len(self.bearish_re.findall(window_text)),
            "confident": len(self.confident_re.findall(window_text)),
            "hedged": len(self.hedged_re.findall(window_text)),
            "buy": len(self.buy_re.findall(window_text)),
            "sell": len(self.sell_re.findall(window_text)),
        }

    def sentiment_score(self, text):
        """Returns (sentiment, hits, confidence). `sentiment` is shrunk
        toward 0.0 when `hits` (total bull+bear lexicon matches) is below
        MIN_CONFIDENT_HITS, so a lone stray word can't swing a long
        transcript to a full-strength +/-1.0."""
        bull = len(self.bullish_re.findall(text))
        bear = len(self.bearish_re.findall(text))
        total = bull + bear
        if total == 0:
            return 0.0, 0, 0.0
        raw = (bull - bear) / total
        confidence = min(1.0, total / MIN_CONFIDENT_HITS)
        return round(raw * confidence, 3), total, round(confidence, 3)

    def tag(self, text):
        sentences = _split_sentences(text)
        word_count = len(text.split())

        sector_mentions = self._mention_counts(self.sectors, text)
        theme_mentions = self._mention_counts(self.themes, text)
        stock_mentions = self._mention_counts(self.stocks, text)

        entity_sentiment = {}
        entity_conviction = {}
        entity_stance = {}
        entity_contested = []
        for compiled in (self.sectors, self.themes, self.stocks):
            for entity_id, indices in self._sentence_hits(compiled, sentences).items():
                window_text = self._window_text(sentences, indices)
                sig = self._window_signals(window_text)

                tone_total = sig["bull"] + sig["bear"]
                if tone_total:
                    entity_sentiment[entity_id] = round((sig["bull"] - sig["bear"]) / tone_total, 3)
                if sig["bull"] >= CONTESTED_MIN_HITS and sig["bear"] >= CONTESTED_MIN_HITS:
                    entity_contested.append(entity_id)

                conviction_total = sig["confident"] + sig["hedged"]
                if conviction_total:
                    entity_conviction[entity_id] = round(
                        (sig["confident"] - sig["hedged"]) / conviction_total, 3)

                if sig["buy"] != sig["sell"] and (sig["buy"] or sig["sell"]):
                    entity_stance[entity_id] = "buy" if sig["buy"] > sig["sell"] else "sell"

        entity_mentions = {**sector_mentions, **theme_mentions, **stock_mentions}
        entity_mention_density = {
            entity_id: round(count / word_count * 1000, 1)
            for entity_id, count in entity_mentions.items()
        } if word_count else {}

        sentiment, hits, confidence = self.sentiment_score(text)

        def ranked(mentions):
            return sorted(mentions, key=mentions.get, reverse=True)

        return {
            "sectors": ranked(sector_mentions),
            "themes": ranked(theme_mentions),
            "stocks": ranked(stock_mentions),
            "sentiment": sentiment,
            "sentiment_hits": hits,
            "sentiment_confidence": confidence,
            "entity_mentions": entity_mentions,
            "entity_mention_density": entity_mention_density,
            "entity_sentiment": entity_sentiment,
            "entity_conviction": entity_conviction,
            "entity_stance": entity_stance,
            "entity_contested": entity_contested,
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
