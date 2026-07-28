#!/usr/bin/env python3
"""Free alternative to pipeline/enrich_claude.py: Claude-quality tagging
without an ANTHROPIC_API_KEY or any metered API charge.

enrich_claude.py calls the Anthropic API directly, which is billed
per-token on a separate account. This script instead stages a batch of
transcribed episodes for a *Claude Code session* to tag by hand -- that
usage is covered by your existing Claude plan, not a per-call charge, so
it's meant to be run on whatever cadence suits you (e.g. weekly).

Workflow:
    1. python3 pipeline/manual_review.py list --limit 15
       Writes data/manual_review/pending.json: the oldest not-yet-Claude-
       tagged transcribed episodes, plus the allowed taxonomy ids and an
       empty tags template for each.

    2. Open a Claude Code session and ask Claude to fill in
       data/manual_review/pending.json: read each episode's
       transcript_path and write sectors/themes/stocks/sentiment/summary
       for it, using only ids from the allowed lists in the file.

    3. python3 pipeline/manual_review.py apply data/manual_review/pending.json
       Validates ids against config/taxonomy.json, updates
       data/episodes.json and the matching data/transcripts/*.md
       frontmatter (tags_source: "claude-manual"), and rebuilds
       data/aggregates.json.

Repeat weekly (or whenever) until the backlog is clear; newly transcribed
episodes join the pending list automatically.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from run import CONFIG_DIR, DATA_DIR, build_aggregates, load_json, save_json  # noqa: E402

REVIEW_DIR = os.path.join(DATA_DIR, "manual_review")
DEFAULT_BATCH_PATH = os.path.join(REVIEW_DIR, "pending.json")


def _yaml_scalar(value):
    return json.dumps(value)


def parse_frontmatter(content):
    """Transcript files are written by transcribe.py with a simple
    `key: <json value>` frontmatter (see write_transcript_file) -- every
    value is a valid JSON literal, so parsing is just JSON-per-line."""
    assert content.startswith("---\n"), "not a recognized transcript file"
    end = content.index("\n---\n", 4)
    front_text = content[4:end]
    body = content[end + 5:]
    data = {}
    for line in front_text.splitlines():
        if not line.strip():
            continue
        key, _, raw = line.partition(":")
        data[key.strip()] = json.loads(raw.strip())
    return data, body


FRONTMATTER_ORDER = [
    "title", "podcast", "podcast_id", "published_at", "guid", "link",
    "duration", "word_count", "language", "tags_source", "sectors",
    "themes", "stocks", "sentiment", "summary",
]


def rewrite_transcript_tags(path, tags, source):
    """Update an existing transcript .md file's frontmatter tags in place,
    leaving the transcript body untouched."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        content = f.read()
    data, body = parse_frontmatter(content)

    data["tags_source"] = source
    data["sectors"] = tags["sectors"]
    data["themes"] = tags["themes"]
    data["stocks"] = tags["stocks"]
    data["sentiment"] = tags["sentiment"]
    if tags.get("summary"):
        data["summary"] = tags["summary"]
    else:
        data.pop("summary", None)
    # A claude-manual pass is one holistic read, not a per-sentence regex
    # scan -- drop any per-entity fields a prior keyword pass may have left,
    # since they'd no longer match this episode's (possibly revised) tags.
    for stale_key in ("sentiment_hits", "sentiment_confidence", "entity_mentions", "entity_mention_density",
                      "entity_sentiment", "entity_conviction", "entity_stance", "entity_contested"):
        data.pop(stale_key, None)

    lines = ["---"]
    for key in FRONTMATTER_ORDER:
        if key in data:
            lines.append(f"{key}: {_yaml_scalar(data[key])}")
    lines.append("---")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write(body)
    return True


def cmd_list(args):
    episodes = load_json(os.path.join(DATA_DIR, "episodes.json"), [])
    taxonomy_raw = load_json(os.path.join(CONFIG_DIR, "taxonomy.json"), {})

    pending = [
        ep for ep in episodes
        if ep.get("transcript_status") == "done"
        and ep.get("tags_source") in (None, "keyword")
        and ep.get("transcript_path")
    ]
    pending.sort(key=lambda ep: ep.get("published_at") or "")  # oldest first: clear backlog in order

    if not pending:
        print("Nothing pending -- every transcribed episode already has a Claude tagging pass.")
        return

    batch = pending[:args.limit] if args.limit else pending

    episodes_field = {}
    for ep in batch:
        episodes_field[ep["guid"]] = {
            "podcast": ep.get("podcast_name"),
            "title": ep.get("title"),
            "transcript_path": ep.get("transcript_path"),
            "sectors": [],
            "themes": [],
            "stocks": [],
            "sentiment": 0.0,
            "summary": "",
        }

    out = {
        "_instructions": (
            "Fill in sectors/themes/stocks/sentiment/summary for each episode below "
            "by reading its transcript_path. Use ONLY ids from the allowed lists. "
            "sentiment is -1.0 (bearish) to 1.0 (bullish). summary is one or two "
            "sentences on the investment-relevant takeaway. Leave an array empty if "
            "genuinely not discussed -- don't force matches. Then run: "
            "python3 pipeline/manual_review.py apply data/manual_review/pending.json"
        ),
        "_allowed_sector_ids": [e["id"] for e in taxonomy_raw.get("sectors", [])],
        "_allowed_theme_ids": [e["id"] for e in taxonomy_raw.get("themes", [])],
        "_allowed_stock_ids": [e["id"] for e in taxonomy_raw.get("stocks", [])],
        "episodes": episodes_field,
    }

    os.makedirs(REVIEW_DIR, exist_ok=True)
    save_json(DEFAULT_BATCH_PATH, out)

    print(f"{len(pending)} episode(s) pending in total; staged {len(batch)} in {DEFAULT_BATCH_PATH}")
    for ep in batch:
        print(f"  - {ep['podcast_name']}: {ep['title']}  ({ep['transcript_path']})")
    print(f"\nNext: open a Claude Code session and ask Claude to fill in {DEFAULT_BATCH_PATH}, "
          f"then run:\n  python3 pipeline/manual_review.py apply {DEFAULT_BATCH_PATH}")


def cmd_apply(args):
    episodes = load_json(os.path.join(DATA_DIR, "episodes.json"), [])
    taxonomy_raw = load_json(os.path.join(CONFIG_DIR, "taxonomy.json"), {})
    by_guid = {ep["guid"]: ep for ep in episodes}

    with open(args.batch_path) as f:
        batch = json.load(f)

    valid_sectors = {e["id"] for e in taxonomy_raw.get("sectors", [])}
    valid_themes = {e["id"] for e in taxonomy_raw.get("themes", [])}
    valid_stocks = {e["id"] for e in taxonomy_raw.get("stocks", [])}

    applied, skipped, dropped_ids = 0, [], []
    for guid, entry in batch.get("episodes", {}).items():
        ep = by_guid.get(guid)
        if ep is None:
            skipped.append((guid, "no matching episode in data/episodes.json"))
            continue

        sectors = [s for s in entry.get("sectors", []) if s in valid_sectors]
        themes = [t for t in entry.get("themes", []) if t in valid_themes]
        stocks = [s for s in entry.get("stocks", []) if s in valid_stocks]
        for kind, raw, valid in (("sector", entry.get("sectors", []), valid_sectors),
                                  ("theme", entry.get("themes", []), valid_themes),
                                  ("stock", entry.get("stocks", []), valid_stocks)):
            for x in raw:
                if x not in valid:
                    dropped_ids.append(f"{kind}:{x}")

        try:
            sentiment = round(float(entry.get("sentiment", 0.0)), 3)
        except (TypeError, ValueError):
            sentiment = 0.0

        tags = {"sectors": sectors, "themes": themes, "stocks": stocks, "sentiment": sentiment,
                "summary": entry.get("summary", "")}

        ep["tags"] = {k: tags[k] for k in ("sectors", "themes", "stocks", "sentiment")}
        ep["tags_source"] = "claude-manual"
        if tags["summary"]:
            ep["llm_summary"] = tags["summary"]

        transcript_path = ep.get("transcript_path")
        if transcript_path:
            full_path = os.path.join(os.path.dirname(DATA_DIR), transcript_path)
            rewrite_transcript_tags(full_path, tags, "claude-manual")

        applied += 1

    all_episodes = sorted(by_guid.values(), key=lambda e: e.get("published_at") or "", reverse=True)
    save_json(os.path.join(DATA_DIR, "episodes.json"), all_episodes)

    aggregates = build_aggregates(all_episodes, taxonomy_raw)
    import datetime
    aggregates["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_json(os.path.join(DATA_DIR, "aggregates.json"), aggregates)

    print(f"Applied Claude tags to {applied} episode(s).")
    if dropped_ids:
        print(f"Dropped {len(dropped_ids)} id(s) not in the taxonomy: {sorted(set(dropped_ids))}")
    if skipped:
        print(f"Skipped {len(skipped)} entr(y/ies): {skipped}")
    print("data/episodes.json, data/aggregates.json and the transcript frontmatter are updated. "
          "Commit and push when ready.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="Stage a batch of pending episodes for manual Claude tagging")
    p_list.add_argument("--limit", type=int, default=15, help="Max episodes to stage (0 = all pending)")
    p_list.set_defaults(func=cmd_list)

    p_apply = sub.add_parser("apply", help="Apply a filled-in batch file back into the dataset")
    p_apply.add_argument("batch_path", nargs="?", default=DEFAULT_BATCH_PATH)
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
