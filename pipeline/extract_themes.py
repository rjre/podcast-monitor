"""Zero-token tagging of episode text against config/taxonomy.json.

This is the default, always-on analysis path: pure regex/keyword matching
over each episode's title + show-notes summary (or, once transcribed, the
full transcript). No network calls, no API key, no per-run cost -- it can
run on every episode, forever, for free.

That said, plain keyword counting is a genuinely blunt instrument: "I
don't think there's a semiconductor bubble" contains the word "bubble" and
would naively score bearish, even though the speaker is saying the
opposite. Three targeted (still zero-token, still just regex) fixes for
the most common ways that goes wrong:

  1. Negation: a lookback window before each bullish/bearish match is
     checked for a negation cue ("not", "isn't", "never", "without", ...).
     If found, the match's polarity is FLIPPED rather than just discarded
     -- "not bearish" is a real (if modest) bullish signal, not neutral.
     Only applied to the bullish/bearish pair, where "positive" and
     "negative" are genuine antonyms of each other. It is deliberately NOT
     applied to confident/hedged: negating a hedge phrase ("I don't think
     X") doesn't make it confident, it's arguably still a hedge, so
     flipping there would make things worse, not better.
  2. Attribution: "some people think", "critics say", "bears argue" and
     similar precede someone else's view, not the speaker's own -- matches
     in that scope are down-weighted (ATTRIBUTION_WEIGHT), not dropped
     entirely, since the episode is still *about* that view.
  3. Contrast: in "X, but Y", Y is usually the actual point being made. A
     match before a contrast conjunction ("but"/"however"/"though"/...) is
     down-weighted (PRE_CONTRAST_WEIGHT); the clause after it keeps full
     weight.

None of this is real language understanding -- sarcasm is still invisible,
and a negation more than ~6 words from its target is still missed. It is a
meaningfully better heuristic than raw counting, not a substitute for
`pipeline/enrich_claude.py` or a human/Claude manual-review pass actually
reading the transcript.

Matching notes:
  - Every taxonomy term is compiled into a case-insensitive, word-boundary
    regex so "AI" doesn't match inside "said" and "T" (AT&T's ticker) is
    never used as a bare term.
  - Each matched entity also gets a mention count (how many times its terms
    hit the text) -- a stronger salience/intensity signal than the plain
    presence/absence used previously, and what pipeline/run.py rolls up
    into aggregate "buzz" figures.
  - Sentiment has two layers:
      1. Overall episode sentiment: net bullish vs. bearish lexicon hits
         (negation/attribution/contrast-adjusted, see above), shrunk toward
         neutral when the hit count is low -- a single incidental bearish
         word in a 5,000-word transcript should not read as a full-strength
         -1.0. `sentiment_hits` and `sentiment_confidence` are exposed so
         downstream consumers can see how much signal the score is based on.
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

# How far back (in characters, roughly 6-7 words) to look for a negation
# cue before a bullish/bearish match.
NEGATION_LOOKBACK_CHARS = 40

# Matches attributed to someone else's view ("some people think...") or
# appearing before a contrast conjunction's punchline ("X, but Y") count
# for less than a direct, un-hedged statement.
ATTRIBUTION_WEIGHT = 0.5
PRE_CONTRAST_WEIGHT = 0.4

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|cannot|can't|don't|doesn't|didn't|isn't|wasn't|aren't|"
    r"weren't|won't|wouldn't|shouldn't|couldn't|without|hardly|rarely|barely)\b",
    re.IGNORECASE,
)
_CONTRAST_RE = re.compile(r"\b(?:but|however|though|although|yet)\b", re.IGNORECASE)
_ATTRIBUTION_RE = re.compile(
    r"\b(?:some (?:people |analysts |investors )?(?:say|think|believe|argue|worry)|"
    r"critics (?:say|argue|warn)|bears (?:say|argue)|bulls (?:say|argue)|"
    r"skeptics (?:say|warn)|according to|there(?:'s| is| are) (?:a )?concerns? that)\b",
    re.IGNORECASE,
)


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
    def _window_sentences(sentences, indices):
        """Every sentence within LOCAL_WINDOW of any of `indices`, deduped,
        in original order -- the shared context all per-entity signals
        (sentiment, conviction, stance) are drawn from. Kept as a list of
        whole sentences (not concatenated into one blob) so negation and
        contrast-clause scope stay meaningful."""
        seen = set()
        result = []
        for idx in indices:
            lo, hi = max(0, idx - LOCAL_WINDOW), min(len(sentences) - 1, idx + LOCAL_WINDOW)
            for j in range(lo, hi + 1):
                if j not in seen:
                    seen.add(j)
                    result.append(sentences[j])
        return result

    @staticmethod
    def _weighted_hits(sentence, positive_re, negative_re, flip_on_negation):
        """One sentence's contribution to a positive/negative pair, with
        negation (optional), attribution, and contrast-clause weighting
        applied. Returns (positive_weight, negative_weight, raw_hit_count)
        -- raw_hit_count is unweighted, for confidence/shrinkage purposes.

        Attribution is scoped to the clause the match is actually in: in
        "some people think X, but I think Y", an attribution cue before
        "but" must not discount Y, which is the speaker's own statement,
        not the attributed one.
        """
        contrast_match = _CONTRAST_RE.search(sentence)
        contrast_at = contrast_match.end() if contrast_match else None

        positive = negative = 0.0
        raw = 0
        for regex, is_positive in ((positive_re, True), (negative_re, False)):
            for m in regex.finditer(sentence):
                raw += 1
                start = m.start()
                in_post_clause = contrast_at is not None and start >= contrast_at

                negated = False
                if flip_on_negation:
                    lookback = sentence[max(0, start - NEGATION_LOOKBACK_CHARS):start]
                    negated = bool(_NEGATION_RE.search(lookback))
                effective_positive = (not is_positive) if negated else is_positive

                clause_start = contrast_at if in_post_clause else 0
                weight = 1.0
                if _ATTRIBUTION_RE.search(sentence[clause_start:start]):
                    weight *= ATTRIBUTION_WEIGHT
                if contrast_at is not None and not in_post_clause:
                    weight *= PRE_CONTRAST_WEIGHT

                if effective_positive:
                    positive += weight
                else:
                    negative += weight
        return positive, negative, raw

    @classmethod
    def _signal_over_sentences(cls, sentences, positive_re, negative_re, flip_on_negation):
        positive = negative = 0.0
        raw = 0
        for sentence in sentences:
            p, n, r = cls._weighted_hits(sentence, positive_re, negative_re, flip_on_negation)
            positive += p
            negative += n
            raw += r
        return positive, negative, raw

    def _window_signals(self, window_sentences):
        """Weighted lexicon signal in one entity's local window. bull/bear
        get negation-flipping (clean antonym pair); confident/hedged and
        buy/sell don't (see module docstring for why), but all three pairs
        still get attribution/contrast weighting."""
        bull, bear, _ = self._signal_over_sentences(window_sentences, self.bullish_re, self.bearish_re, True)
        confident, hedged, _ = self._signal_over_sentences(
            window_sentences, self.confident_re, self.hedged_re, False)
        buy, sell, _ = self._signal_over_sentences(window_sentences, self.buy_re, self.sell_re, False)
        return {"bull": bull, "bear": bear, "confident": confident, "hedged": hedged, "buy": buy, "sell": sell}

    def sentiment_score(self, text):
        """Returns (sentiment, hits, confidence). `sentiment` is shrunk
        toward 0.0 when `hits` (total bull+bear lexicon matches, unweighted)
        is below MIN_CONFIDENT_HITS, so a lone stray word can't swing a
        long transcript to a full-strength +/-1.0."""
        sentences = _split_sentences(text)
        bull, bear, raw_hits = self._signal_over_sentences(sentences, self.bullish_re, self.bearish_re, True)
        total = bull + bear
        if raw_hits == 0:
            return 0.0, 0, 0.0
        raw = (bull - bear) / total if total else 0.0
        confidence = min(1.0, raw_hits / MIN_CONFIDENT_HITS)
        return round(raw * confidence, 3), raw_hits, round(confidence, 3)

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
                window_sentences = self._window_sentences(sentences, indices)
                sig = self._window_signals(window_sentences)

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
            "Regional banks and private credit funds have surged in importance. "
            "Some people think there's a bubble in leveraged loans, but I don't "
            "think that's actually true -- default risk looks contained to me."
        ),
    }
    print(json.dumps(tag_episode(sample, tx), indent=2))
