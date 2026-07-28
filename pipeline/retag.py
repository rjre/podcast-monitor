#!/usr/bin/env python3
"""Re-run zero-token tagging over already-tagged episodes without
re-fetching RSS feeds or re-downloading/re-transcribing audio -- for when
config/taxonomy.json or the tagging engine in extract_themes.py changes and
the existing keyword-tagged backlog should pick up the improvement.

pipeline/run.py --retag can't do this offline: it only retags episodes that
come back from a live RSS fetch, bounded by --backfill-days, so episodes
that have since scrolled out of a feed's window are silently skipped. This
script instead works directly from data/episodes.json (title/summary) and
the already-saved data/transcripts/*.md files (full transcript text), so it
needs no network access and covers the whole backlog every time.

Only touches episodes tagged 'keyword' or untagged (tags_source is None).
'llm' and 'claude-manual' episodes were tagged by Claude actually reading
the content and are left alone.

Usage:
    python3 pipeline/retag.py            # every eligible episode
    python3 pipeline/retag.py --limit 50
"""
import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from extract_themes import Taxonomy, tag_episode  # noqa: E402
from manual_review import parse_frontmatter  # noqa: E402
from run import CONFIG_DIR, DATA_DIR, ROOT, build_aggregates, load_json, save_json  # noqa: E402
from transcribe import write_transcript_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max episodes to retag (0 = all eligible)")
    args = parser.parse_args()

    taxonomy = Taxonomy(os.path.join(CONFIG_DIR, "taxonomy.json"))
    taxonomy_raw = load_json(os.path.join(CONFIG_DIR, "taxonomy.json"), {})
    episodes = load_json(os.path.join(DATA_DIR, "episodes.json"), [])
    by_guid = {ep["guid"]: ep for ep in episodes}

    eligible = [ep for ep in episodes if ep.get("tags_source") in (None, "keyword")]
    if args.limit:
        eligible = eligible[:args.limit]

    retranscribed, plain, missing = 0, 0, 0
    for ep in eligible:
        transcript_path = ep.get("transcript_path")
        if ep.get("transcript_status") == "done" and transcript_path:
            full_path = os.path.join(ROOT, transcript_path)
            if not os.path.exists(full_path):
                missing += 1
                continue
            with open(full_path) as f:
                content = f.read()
            data, body = parse_frontmatter(content)
            combined_text = f"{ep['title']}. {ep.get('summary', '')} {body}"
            tags = taxonomy.tag(combined_text)
            write_transcript_file(full_path, ep, body, tags, "keyword", language=data.get("language"))
            retranscribed += 1
        else:
            tags = tag_episode(ep, taxonomy)
            plain += 1

        ep["tags"] = {k: tags[k] for k in ("sectors", "themes", "stocks", "sentiment")}
        for extra in ("sentiment_hits", "sentiment_confidence", "entity_mentions", "entity_sentiment"):
            if extra in tags:
                ep["tags"][extra] = tags[extra]
        ep["tags_source"] = "keyword"
        by_guid[ep["guid"]] = ep

    all_episodes = sorted(by_guid.values(), key=lambda e: e.get("published_at") or "", reverse=True)
    save_json(os.path.join(DATA_DIR, "episodes.json"), all_episodes)

    aggregates = build_aggregates(all_episodes, taxonomy_raw)
    aggregates["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_json(os.path.join(DATA_DIR, "aggregates.json"), aggregates)

    print(f"Retagged {retranscribed} transcribed + {plain} title/summary-only episode(s) "
          f"({retranscribed + plain} total) with the current taxonomy/engine.")
    if missing:
        print(f"Skipped {missing} episode(s) whose transcript_path was missing on disk.", file=sys.stderr)


if __name__ == "__main__":
    main()
