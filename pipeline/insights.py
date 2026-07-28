"""Cross-cutting "find the needle in the haystack" signals -- the kind of
thing a per-entity rollup (pipeline/run.py's build_aggregates) can't surface
because it isn't shaped around one entity at a time:

  - podcast_baselines / surprising_episodes: a show's OWN historical tone,
    and episodes that deviate sharply from it. A naturally-bearish show
    turning bullish is a bigger tell than a naturally-bullish one being
    bullish, which a dataset-wide baseline can't tell you.
  - contrarian_calls: within an entity+month where most episodes clearly
    agree on tone, the episode(s) that disagreed -- the lone dissenting
    voice, not just the loudest opinion.
  - entity_cooccurrence: entity pairs that show up together far more than
    chance would predict (lift = observed / expected co-occurrence) --
    emergent narrative linkages nobody hand-coded into the taxonomy.
  - guests: best-effort extraction of named guests from episode titles
    ("w/ Kyle Grieve", "with Ben Felix", "feat. ..."), tracked across
    appearances. Deliberately noisy -- title parsing can't distinguish a
    real name from a capitalized phrase -- so results are gated on a guest
    name recurring across >= GUEST_MIN_EPISODES episodes verbatim, which a
    one-off false match essentially never does.

Called once from pipeline/run.py's build_aggregates() and merged into its
return value, so every existing entry point (run.py, transcribe.py,
retag.py, manual_review.py) picks these up for free without needing its
own changes.
"""
import re
from collections import Counter, defaultdict
from itertools import combinations

# Minimum episodes from a podcast before we trust its own average tone as
# a meaningful personal baseline (rather than noise from 1-2 episodes).
BASELINE_MIN_EPISODES = 3
SURPRISE_MIN_MAGNITUDE = 0.4

# An entity+month needs at least this many mentions, with at least this
# fraction agreeing on direction, before dissent from the rest counts as
# "contrarian" rather than just an ordinary split of opinion.
CONSENSUS_MIN_GROUP = 5
CONSENSUS_MIN_FRACTION = 0.7
CONTRARIAN_MIN_MAGNITUDE = 0.3

# A pair needs at least this many joint appearances before its lift ratio
# is trusted -- two rare entities that happen to co-occur once or twice
# would otherwise post enormous (and meaningless) lift.
MIN_PAIR_COUNT = 5

# A guest name needs to recur across at least this many episodes verbatim
# -- the main defense against title-parsing false positives, since a
# random capitalized phrase essentially never repeats exactly.
GUEST_MIN_EPISODES = 2

_GUEST_NAME_RE = re.compile(
    r"(?:\bw/\s*|\bfeat\.?\s+|\bft\.?\s+|\bwith\s+)"
    r"([A-Z][A-Za-z’'.-]+(?:\s+[A-Z][A-Za-z’'.-]+){0,3})"
)
# Common capitalized words that follow "with"/"w/" without being a guest
# credit (show branding, generic phrases) -- filtered before the
# recurrence-across-episodes check even gets a chance to run.
_GUEST_STOPWORDS = {
    "the", "this", "that", "these", "those", "blockbuster", "earnings",
    "wall", "street", "market", "markets", "debt", "interest", "rates",
    "highest", "lowest", "everything", "everyone", "anything", "something",
    "nothing", "friends", "benefits",
}


def extract_guest_name(title):
    """Best-effort single-guest extraction from an episode title. Returns
    None if nothing matches or the match looks like a stopword phrase
    rather than a name. Multi-guest titles ("w/ A, B & C") only capture
    the first name -- comma/ampersand-separated names aren't joined."""
    match = _GUEST_NAME_RE.search(title or "")
    if not match:
        return None
    name = match.group(1).strip().rstrip(".,:;-")
    if len(name) < 4:
        return None
    words = name.lower().split()
    if any(w in _GUEST_STOPWORDS for w in words):
        return None
    return name


def _entity_ids(tags):
    return sorted(set(tags.get("sectors", []) + tags.get("themes", []) + tags.get("stocks", [])))


def build_podcast_baselines(episodes):
    """podcast_id -> {avg_sentiment, avg_conviction, episode_count}, from
    every tagged episode that podcast has (not just transcribed ones)."""
    sums = defaultdict(lambda: {"sentiment_sum": 0.0, "n": 0, "conviction_sum": 0.0, "conviction_n": 0})
    for ep in episodes:
        tags = ep.get("tags", {})
        pid = ep.get("podcast_id")
        if not pid or not _entity_ids(tags):
            continue
        agg = sums[pid]
        agg["sentiment_sum"] += tags.get("sentiment", 0.0)
        agg["n"] += 1
        for conviction in (tags.get("entity_conviction") or {}).values():
            agg["conviction_sum"] += conviction
            agg["conviction_n"] += 1

    baselines = {}
    for pid, agg in sums.items():
        if agg["n"] < BASELINE_MIN_EPISODES:
            continue
        baselines[pid] = {
            "avg_sentiment": round(agg["sentiment_sum"] / agg["n"], 3),
            "avg_conviction": round(agg["conviction_sum"] / agg["conviction_n"], 3) if agg["conviction_n"] else None,
            "episode_count": agg["n"],
        }
    return baselines


def find_surprising_episodes(episodes, podcast_baselines, top_n=20):
    out = []
    for ep in episodes:
        tags = ep.get("tags", {})
        pid = ep.get("podcast_id")
        baseline = podcast_baselines.get(pid)
        if not baseline or not _entity_ids(tags):
            continue
        surprise = round(tags.get("sentiment", 0.0) - baseline["avg_sentiment"], 3)
        if abs(surprise) < SURPRISE_MIN_MAGNITUDE:
            continue
        out.append({
            "guid": ep.get("guid"),
            "title": ep.get("title"),
            "podcast_id": pid,
            "podcast_name": ep.get("podcast_name"),
            "published_at": ep.get("published_at"),
            "sentiment": tags.get("sentiment", 0.0),
            "podcast_baseline_sentiment": baseline["avg_sentiment"],
            "surprise": surprise,
        })
    out.sort(key=lambda x: abs(x["surprise"]), reverse=True)
    return out[:top_n]


def find_contrarian_calls(episodes, labels_by_id, top_n=20):
    groups = defaultdict(list)
    for ep in episodes:
        tags = ep.get("tags", {})
        month = (ep.get("published_at") or "")[:7]
        if not month:
            continue
        sentiment = tags.get("sentiment", 0.0)
        entity_sentiment = tags.get("entity_sentiment") or {}
        for entity_id in _entity_ids(tags):
            groups[(entity_id, month)].append((entity_sentiment.get(entity_id, sentiment), ep))

    out = []
    for (entity_id, month), items in groups.items():
        total = len(items)
        if total < CONSENSUS_MIN_GROUP:
            continue
        bulls = sum(1 for s, _ in items if s > 0.15)
        bears = sum(1 for s, _ in items if s < -0.15)
        if bulls / total >= CONSENSUS_MIN_FRACTION:
            consensus, is_dissent = "bullish", (lambda s: s < -CONTRARIAN_MIN_MAGNITUDE)
            consensus_fraction = round(bulls / total, 2)
        elif bears / total >= CONSENSUS_MIN_FRACTION:
            consensus, is_dissent = "bearish", (lambda s: s > CONTRARIAN_MIN_MAGNITUDE)
            consensus_fraction = round(bears / total, 2)
        else:
            continue
        for s, ep in items:
            if not is_dissent(s):
                continue
            out.append({
                "entity_id": entity_id,
                "entity_label": labels_by_id.get(entity_id, entity_id),
                "month": month,
                "consensus": consensus,
                "consensus_fraction": consensus_fraction,
                "group_size": total,
                "guid": ep.get("guid"),
                "title": ep.get("title"),
                "podcast_name": ep.get("podcast_name"),
                "published_at": ep.get("published_at"),
                "sentiment": round(s, 3),
            })
    out.sort(key=lambda x: x["consensus_fraction"] * x["group_size"], reverse=True)
    return out[:top_n]


def build_cooccurrence(episodes, labels_by_id, kind_by_id, top_n=20):
    entity_counts = Counter()
    pair_counts = Counter()
    total = 0
    for ep in episodes:
        entities = _entity_ids(ep.get("tags", {}))
        if not entities:
            continue
        total += 1
        entity_counts.update(entities)
        pair_counts.update(combinations(entities, 2))

    out = []
    for (a, b), count in pair_counts.items():
        if count < MIN_PAIR_COUNT:
            continue
        expected = entity_counts[a] * entity_counts[b] / total
        if expected <= 0:
            continue
        out.append({
            "a": a, "a_label": labels_by_id.get(a, a), "a_kind": kind_by_id.get(a, ""),
            "b": b, "b_label": labels_by_id.get(b, b), "b_kind": kind_by_id.get(b, ""),
            "co_occurrences": count,
            "lift": round(count / expected, 2),
        })
    out.sort(key=lambda x: x["lift"], reverse=True)
    return out[:top_n]


def build_guest_index(episodes, top_n=20):
    by_guest = defaultdict(list)
    for ep in episodes:
        name = extract_guest_name(ep.get("title", ""))
        if name:
            by_guest[name].append(ep)

    out = []
    for name, eps in by_guest.items():
        if len(eps) < GUEST_MIN_EPISODES:
            continue
        sentiments = [e.get("tags", {}).get("sentiment", 0.0) for e in eps]
        entity_counter = Counter()
        for e in eps:
            entity_counter.update(_entity_ids(e.get("tags", {})))
        out.append({
            "name": name,
            "episode_count": len(eps),
            "avg_sentiment": round(sum(sentiments) / len(sentiments), 3),
            "top_entities": [eid for eid, _ in entity_counter.most_common(5)],
            "podcasts": sorted({e.get("podcast_name") for e in eps if e.get("podcast_name")}),
        })
    out.sort(key=lambda x: x["episode_count"], reverse=True)
    return out[:top_n]


def build_insights(episodes, taxonomy_raw):
    labels_by_id, kind_by_id = {}, {}
    for kind in ("sectors", "themes", "stocks"):
        for entry in taxonomy_raw.get(kind, []):
            labels_by_id[entry["id"]] = entry["label"]
            kind_by_id[entry["id"]] = kind[:-1]  # "sectors" -> "sector"

    podcast_baselines = build_podcast_baselines(episodes)
    return {
        "podcast_baselines": podcast_baselines,
        "surprising_episodes": find_surprising_episodes(episodes, podcast_baselines),
        "contrarian_calls": find_contrarian_calls(episodes, labels_by_id),
        "entity_cooccurrence": build_cooccurrence(episodes, labels_by_id, kind_by_id),
        "guests": build_guest_index(episodes),
    }
