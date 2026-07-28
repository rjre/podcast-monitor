#!/usr/bin/env python3
"""Pipeline orchestrator: fetch new episodes, tag them, write data files.

Usage:
    python3 pipeline/run.py                 # incremental: only new episodes
    python3 pipeline/run.py --backfill-days 90   # (re)pull a wider window

Idempotent and safe to run repeatedly (e.g. on a daily GitHub Actions
schedule): episodes are de-duplicated by RSS guid, and existing tags are
kept unless --retag is passed.

Outputs (all committed to the repo so other projects/tools can just read
the JSON without re-running anything):
    data/episodes.json    - every episode + its tags, the reusable dataset
    data/aggregates.json  - precomputed mention counts, for quick reuse
    data/state.json       - last run metadata
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# Momentum window: mentions in the trailing 30 days vs. the 30 days before
# that. Short enough to surface a real shift in what shows are talking
# about, long enough that a single week's episode backlog doesn't read as
# a "trend".
MOMENTUM_WINDOW_DAYS = 30
TREND_RISING_PCT = 20.0
TREND_FALLING_PCT = -20.0

# How much avg_sentiment/avg_conviction has to move between the same two
# windows to count as a real shift rather than noise, on their native -1..1
# scale.
SENTIMENT_SHIFT_THRESHOLD = 0.25
CONVICTION_SHIFT_THRESHOLD = 0.3

# Minimum months of history (from the monthly time series) before
# sentiment_volatility is reported -- variance off 1-2 data points isn't a
# real "how much does opinion swing" signal.
MIN_MONTHS_FOR_VOLATILITY = 3

# An entity with no mentions in this many days is "dormant" -- gone quiet
# after being discussed, as opposed to merely "falling" (still mentioned,
# just less).
DORMANT_AFTER_DAYS = 2 * MOMENTUM_WINDOW_DAYS

sys.path.insert(0, os.path.dirname(__file__))

from fetch_feeds import fetch_episodes  # noqa: E402
from extract_themes import Taxonomy, tag_episode  # noqa: E402
from insights import build_insights  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "config")
DATA_DIR = os.path.join(ROOT, "data")


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def _parse_published_at(ep):
    raw = ep.get("published_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_aggregates(episodes, taxonomy_raw, now=None):
    """Roll episodes up into per-entity mention counts + sentiment, and a
    monthly time series per entity, so downstream tools/dashboards don't
    need to recompute this themselves.

    Beyond the plain episode-count "mentions" figure, this also surfaces:
      - total_hits: sum of in-episode term-hit counts (from the keyword
        tagger's entity_mentions, when present) -- a salience/intensity
        signal, so an entity obsessed over for 20 minutes outweighs one
        name-dropped once. Falls back to 1 per episode for tags without
        per-entity counts (llm/claude-manual/legacy).
      - avg_sentiment / sentiment_divergence: per-entity local sentiment
        (from entity_sentiment, when present, else the episode's overall
        sentiment) averaged, and how far that sits from the dataset-wide
        baseline -- flags entities that are unusually bullish/bearish
        relative to everything else being discussed right now.
      - momentum_pct / trend: mentions in the trailing MOMENTUM_WINDOW_DAYS
        vs. the window before that, as a rising/falling/flat signal.
      - avg_conviction: mean of entity_conviction (confident vs. hedged
        language), when present -- separate from tone.
      - buy_mentions / sell_mentions: episode counts where entity_stance
        called it a buy vs. a sell -- an actionable tally, not just tone.
      - contested_episodes: how many episodes flagged this entity as
        entity_contested (meaningful bullish AND bearish language at once).

    momentum_pct/trend track a shift in how MUCH an entity is discussed.
    These three track a shift in HOW it's discussed -- the actual ask
    behind this feature ("tease out changes in tone/sentiment"):
      - sentiment_shift / sentiment_trend: avg_sentiment in the trailing
        MOMENTUM_WINDOW_DAYS vs. the window before that. A sentiment of,
        say, +0.1 tells you nothing about direction of travel -- this
        does: "turning_bullish"/"turning_bearish"/"stable"/
        "insufficient-data".
      - conviction_shift / conviction_trend: same idea, for avg_conviction
        -- "growing_confidence"/"growing_doubt" independent of whether the
        tone itself moved.
      - sentiment_volatility: standard deviation of the monthly avg_sentiment
        series -- a topic with wide swings month to month (contested over
        time) reads very differently from one that's been steadily +0.3
        for a year, even if their overall averages match.
      - lifecycle_stage (using first_seen/last_seen alongside trend):
        emerging/growing/steady/declining/dormant -- "dormant" in
        particular (gone quiet after being discussed) isn't visible from
        trend alone, which only compares two 30-day windows.

    Also merges in pipeline/insights.py's build_insights() output
    (podcast_baselines, surprising_episodes, contrarian_calls,
    entity_cooccurrence, guests) -- cross-cutting signals that don't fit a
    per-entity rollup.
    """
    now = now or datetime.now(timezone.utc)
    recent_start = now - timedelta(days=MOMENTUM_WINDOW_DAYS)
    prior_start = now - timedelta(days=2 * MOMENTUM_WINDOW_DAYS)

    def label_map(section):
        return {e["id"]: e["label"] for e in taxonomy_raw[section]}

    labels = {
        "sectors": label_map("sectors"),
        "themes": label_map("themes"),
        "stocks": label_map("stocks"),
    }

    def new_agg():
        return {"mentions": 0, "total_hits": 0, "sentiment_sum": 0.0, "sentiment_n": 0,
                 "recent": 0, "prior": 0, "conviction_sum": 0.0, "conviction_n": 0,
                 "buy_mentions": 0, "sell_mentions": 0, "contested_episodes": 0,
                 "recent_sentiment_sum": 0.0, "prior_sentiment_sum": 0.0,
                 "recent_conviction_sum": 0.0, "recent_conviction_n": 0,
                 "prior_conviction_sum": 0.0, "prior_conviction_n": 0,
                 "first_seen": None, "last_seen": None}

    totals = {k: defaultdict(new_agg) for k in labels}
    monthly = {k: defaultdict(lambda: defaultdict(lambda: {"mentions": 0, "sentiment_sum": 0.0})) for k in labels}

    global_sentiment_sum, global_sentiment_n = 0.0, 0

    for ep in episodes:
        month = (ep.get("published_at") or "")[:7]  # YYYY-MM
        tags = ep.get("tags", {})
        sentiment = tags.get("sentiment", 0.0)
        entity_mentions = tags.get("entity_mentions") or {}
        entity_sentiment = tags.get("entity_sentiment") or {}
        entity_conviction = tags.get("entity_conviction") or {}
        entity_stance = tags.get("entity_stance") or {}
        entity_contested = tags.get("entity_contested") or []
        pub_dt = _parse_published_at(ep)
        touched = tags.get("sectors") or tags.get("themes") or tags.get("stocks")

        if touched:
            global_sentiment_sum += sentiment
            global_sentiment_n += 1

        for kind in ("sectors", "themes", "stocks"):
            for entity_id in tags.get(kind, []):
                agg = totals[kind][entity_id]
                agg["mentions"] += 1
                agg["total_hits"] += entity_mentions.get(entity_id, 1)
                local_sentiment = entity_sentiment.get(entity_id, sentiment)
                agg["sentiment_sum"] += local_sentiment
                agg["sentiment_n"] += 1
                local_conviction = entity_conviction.get(entity_id)
                if local_conviction is not None:
                    agg["conviction_sum"] += local_conviction
                    agg["conviction_n"] += 1
                if pub_dt is not None:
                    if agg["first_seen"] is None or pub_dt < agg["first_seen"]:
                        agg["first_seen"] = pub_dt
                    if agg["last_seen"] is None or pub_dt > agg["last_seen"]:
                        agg["last_seen"] = pub_dt
                    if pub_dt >= recent_start:
                        agg["recent"] += 1
                        agg["recent_sentiment_sum"] += local_sentiment
                        if local_conviction is not None:
                            agg["recent_conviction_sum"] += local_conviction
                            agg["recent_conviction_n"] += 1
                    elif pub_dt >= prior_start:
                        agg["prior"] += 1
                        agg["prior_sentiment_sum"] += local_sentiment
                        if local_conviction is not None:
                            agg["prior_conviction_sum"] += local_conviction
                            agg["prior_conviction_n"] += 1
                stance = entity_stance.get(entity_id)
                if stance == "buy":
                    agg["buy_mentions"] += 1
                elif stance == "sell":
                    agg["sell_mentions"] += 1
                if entity_id in entity_contested:
                    agg["contested_episodes"] += 1
                if month:
                    monthly[kind][entity_id][month]["mentions"] += 1
                    monthly[kind][entity_id][month]["sentiment_sum"] += local_sentiment

    global_avg_sentiment = round(global_sentiment_sum / global_sentiment_n, 3) if global_sentiment_n else 0.0

    def momentum(recent, prior):
        if recent == 0 and prior == 0:
            return None, "insufficient-data"
        if prior == 0:
            return None, "new"
        pct = round(((recent - prior) / prior) * 100, 1)
        if pct > TREND_RISING_PCT:
            trend = "rising"
        elif pct < TREND_FALLING_PCT:
            trend = "falling"
        else:
            trend = "flat"
        return pct, trend

    def windowed_shift(recent_sum, recent_n, prior_sum, prior_n, threshold, rising_label, falling_label):
        """Same before/after-window comparison as momentum(), but for an
        average (sentiment or conviction) instead of a raw count."""
        recent_avg = round(recent_sum / recent_n, 3) if recent_n else None
        prior_avg = round(prior_sum / prior_n, 3) if prior_n else None
        if recent_avg is None or prior_avg is None:
            return recent_avg, prior_avg, None, "insufficient-data"
        shift = round(recent_avg - prior_avg, 3)
        if shift > threshold:
            label = rising_label
        elif shift < -threshold:
            label = falling_label
        else:
            label = "stable"
        return recent_avg, prior_avg, shift, label

    def lifecycle_stage(agg, trend):
        """Where an entity sits in its own arc, using first/last mention
        date alongside the volume trend already computed -- richer than a
        single rising/falling snapshot: emerging (brand new and active),
        growing/declining (trend says so), dormant (was discussed, has
        gone quiet), or steady (neither rising nor falling, still active)."""
        if agg["last_seen"] is None:
            return "unknown"
        if (now - agg["last_seen"]).days > DORMANT_AFTER_DAYS:
            return "dormant"
        if agg["first_seen"] is not None and (now - agg["first_seen"]).days <= MOMENTUM_WINDOW_DAYS \
                and agg["recent"] > 0:
            return "emerging"
        if trend == "rising":
            return "growing"
        if trend == "falling":
            return "declining"
        return "steady"

    def finalize(kind):
        out = []
        for entity_id, agg in totals[kind].items():
            mentions = agg["mentions"]
            avg_sent = round(agg["sentiment_sum"] / agg["sentiment_n"], 3) if agg["sentiment_n"] else 0.0
            avg_conviction = round(agg["conviction_sum"] / agg["conviction_n"], 3) if agg["conviction_n"] else None
            momentum_pct, trend = momentum(agg["recent"], agg["prior"])
            series = [
                {"month": m, "mentions": v["mentions"], "avg_sentiment": round(v["sentiment_sum"] / v["mentions"], 3)}
                for m, v in sorted(monthly[kind][entity_id].items())
            ]

            recent_avg_sentiment, prior_avg_sentiment, sentiment_shift, sentiment_trend = windowed_shift(
                agg["recent_sentiment_sum"], agg["recent"], agg["prior_sentiment_sum"], agg["prior"],
                SENTIMENT_SHIFT_THRESHOLD, "turning_bullish", "turning_bearish")
            _, _, conviction_shift, conviction_trend = windowed_shift(
                agg["recent_conviction_sum"], agg["recent_conviction_n"],
                agg["prior_conviction_sum"], agg["prior_conviction_n"],
                CONVICTION_SHIFT_THRESHOLD, "growing_confidence", "growing_doubt")

            monthly_sentiments = [m["avg_sentiment"] for m in series]
            sentiment_volatility = (
                round(statistics.pstdev(monthly_sentiments), 3)
                if len(monthly_sentiments) >= MIN_MONTHS_FOR_VOLATILITY else None
            )

            out.append({
                "id": entity_id,
                "label": labels[kind].get(entity_id, entity_id),
                "mentions": mentions,
                "total_hits": agg["total_hits"],
                "avg_sentiment": avg_sent,
                "sentiment_divergence": round(avg_sent - global_avg_sentiment, 3),
                "recent_avg_sentiment": recent_avg_sentiment,
                "prior_avg_sentiment": prior_avg_sentiment,
                "sentiment_shift": sentiment_shift,
                "sentiment_trend": sentiment_trend,
                "sentiment_volatility": sentiment_volatility,
                "recent_30d_mentions": agg["recent"],
                "prior_30d_mentions": agg["prior"],
                "momentum_pct": momentum_pct,
                "trend": trend,
                "avg_conviction": avg_conviction,
                "conviction_shift": conviction_shift,
                "conviction_trend": conviction_trend,
                "buy_mentions": agg["buy_mentions"],
                "sell_mentions": agg["sell_mentions"],
                "contested_episodes": agg["contested_episodes"],
                "first_seen": agg["first_seen"].isoformat() if agg["first_seen"] else None,
                "last_seen": agg["last_seen"].isoformat() if agg["last_seen"] else None,
                "lifecycle_stage": lifecycle_stage(agg, trend),
                "monthly": series,
            })
        out.sort(key=lambda x: x["mentions"], reverse=True)
        return out

    result = {
        "generated_at": None,  # filled in by caller
        "global_avg_sentiment": global_avg_sentiment,
        "sectors": finalize("sectors"),
        "themes": finalize("themes"),
        "stocks": finalize("stocks"),
    }
    result.update(build_insights(episodes, taxonomy_raw))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill-days", type=int, default=90,
                         help="How far back to pull on a cold start (default 90 = ~3 months)")
    parser.add_argument("--retag", action="store_true",
                         help="Re-run tagging on all existing episodes too (e.g. after editing taxonomy.json)")
    args = parser.parse_args()

    podcasts_cfg = load_json(os.path.join(CONFIG_DIR, "podcasts.json"), {"podcasts": []})
    taxonomy_raw = load_json(os.path.join(CONFIG_DIR, "taxonomy.json"), {})
    taxonomy = Taxonomy(os.path.join(CONFIG_DIR, "taxonomy.json"))

    episodes = load_json(os.path.join(DATA_DIR, "episodes.json"), [])
    by_guid = {ep["guid"]: ep for ep in episodes}

    since = datetime.now(timezone.utc) - timedelta(days=args.backfill_days)

    new_count = 0
    errors = []
    for podcast in podcasts_cfg["podcasts"]:
        if not podcast.get("active", True):
            continue
        try:
            fetched = fetch_episodes(podcast, since=since)
        except Exception as exc:  # keep going even if one feed is down
            errors.append(f"{podcast['name']}: {exc}")
            continue

        for ep in fetched:
            existing = by_guid.get(ep["guid"])
            if existing is not None and not args.retag:
                continue
            ep["tags"] = tag_episode(ep, taxonomy)
            ep["tags_source"] = "keyword"
            by_guid[ep["guid"]] = ep
            new_count += 1

    all_episodes = sorted(by_guid.values(), key=lambda e: e.get("published_at") or "", reverse=True)

    now = datetime.now(timezone.utc).isoformat()
    save_json(os.path.join(DATA_DIR, "episodes.json"), all_episodes)

    aggregates = build_aggregates(all_episodes, taxonomy_raw)
    aggregates["generated_at"] = now
    save_json(os.path.join(DATA_DIR, "aggregates.json"), aggregates)

    state = {
        "last_run_at": now,
        "episode_count": len(all_episodes),
        "new_or_retagged_this_run": new_count,
        "podcasts": [p["id"] for p in podcasts_cfg["podcasts"] if p.get("active", True)],
        "errors": errors,
    }
    save_json(os.path.join(DATA_DIR, "state.json"), state)

    print(f"Episodes total: {len(all_episodes)} (new/retagged this run: {new_count})")
    if errors:
        print("Errors:", errors, file=sys.stderr)


if __name__ == "__main__":
    main()
