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
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))

from fetch_feeds import fetch_episodes  # noqa: E402
from extract_themes import Taxonomy, tag_episode  # noqa: E402

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


def build_aggregates(episodes, taxonomy_raw):
    """Roll episodes up into per-entity mention counts + sentiment, and a
    monthly time series per entity, so downstream tools/dashboards don't
    need to recompute this themselves."""

    def label_map(section):
        return {e["id"]: e["label"] for e in taxonomy_raw[section]}

    labels = {
        "sectors": label_map("sectors"),
        "themes": label_map("themes"),
        "stocks": label_map("stocks"),
    }

    totals = {k: defaultdict(lambda: {"mentions": 0, "sentiment_sum": 0.0}) for k in labels}
    monthly = {k: defaultdict(lambda: defaultdict(lambda: {"mentions": 0, "sentiment_sum": 0.0})) for k in labels}

    for ep in episodes:
        month = (ep.get("published_at") or "")[:7]  # YYYY-MM
        tags = ep.get("tags", {})
        sentiment = tags.get("sentiment", 0.0)
        for kind in ("sectors", "themes", "stocks"):
            for entity_id in tags.get(kind, []):
                totals[kind][entity_id]["mentions"] += 1
                totals[kind][entity_id]["sentiment_sum"] += sentiment
                if month:
                    monthly[kind][entity_id][month]["mentions"] += 1
                    monthly[kind][entity_id][month]["sentiment_sum"] += sentiment

    def finalize(kind):
        out = []
        for entity_id, agg in totals[kind].items():
            mentions = agg["mentions"]
            avg_sent = round(agg["sentiment_sum"] / mentions, 3) if mentions else 0.0
            series = [
                {"month": m, "mentions": v["mentions"], "avg_sentiment": round(v["sentiment_sum"] / v["mentions"], 3)}
                for m, v in sorted(monthly[kind][entity_id].items())
            ]
            out.append({
                "id": entity_id,
                "label": labels[kind].get(entity_id, entity_id),
                "mentions": mentions,
                "avg_sentiment": avg_sent,
                "monthly": series,
            })
        out.sort(key=lambda x: x["mentions"], reverse=True)
        return out

    return {
        "generated_at": None,  # filled in by caller
        "sectors": finalize("sectors"),
        "themes": finalize("themes"),
        "stocks": finalize("stocks"),
    }


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
