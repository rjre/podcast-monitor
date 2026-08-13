"""Reconcile transcript .md files that exist on disk but aren't reflected in
episodes.json yet -- happens when transcribe.py's process is killed mid-batch
(it only saves episodes.json once, after the whole batch loop finishes)."""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, "pipeline")
from run import load_json, save_json, build_aggregates, CONFIG_DIR, DATA_DIR

TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")


def parse_frontmatter(path):
    text = open(path, encoding="utf-8").read()
    fm_match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    fm_text = fm_match.group(1)

    def get(key, default=None):
        m = re.search(rf'^{key}:\s*"?(.*?)"?\s*$', fm_text, re.MULTILINE)
        return m.group(1) if m else default

    def get_list(key):
        m = re.search(rf'^{key}:\s*(\[.*?\])\s*$', fm_text, re.MULTILINE)
        return json.loads(m.group(1)) if m else []

    def get_num(key):
        m = re.search(rf'^{key}:\s*(-?[\d.]+)\s*$', fm_text, re.MULTILINE)
        return float(m.group(1)) if m else None

    def get_dict(key):
        m = re.search(rf'^{key}:\s*(\{{.*?\}})\s*$', fm_text, re.MULTILINE)
        return json.loads(m.group(1)) if m else {}

    return {
        "guid": get("guid"),
        "word_count": int(get_num("word_count")) if get_num("word_count") is not None else None,
        "language": get("language"),
        "sectors": get_list("sectors"),
        "themes": get_list("themes"),
        "stocks": get_list("stocks"),
        "sentiment": get_num("sentiment"),
        "sentiment_hits": get_num("sentiment_hits"),
        "sentiment_confidence": get_num("sentiment_confidence"),
        "entity_mentions": get_dict("entity_mentions"),
        "entity_mention_density": get_dict("entity_mention_density"),
        "entity_conviction": get_dict("entity_conviction"),
        "entity_sentiment": get_dict("entity_sentiment"),
    }


def find_orphans():
    out = subprocess.run(
        ["git", "status", "--porcelain", "data/transcripts/"],
        capture_output=True, text=True, check=True,
    ).stdout
    orphans = []
    for line in out.splitlines():
        status, path = line[:2], line[3:]
        if "?" in status or "M" in status:
            if path.endswith(".md"):
                orphans.append(path)
    return orphans


def main():
    orphans = find_orphans()
    if not orphans:
        print("No orphaned transcript files.")
        return

    eps = load_json(os.path.join(DATA_DIR, "episodes.json"), [])
    by_guid = {e["guid"]: e for e in eps}
    reconciled = 0
    for path in orphans:
        fm = parse_frontmatter(path)
        guid = fm["guid"]
        e = by_guid.get(guid)
        if not e:
            print(f"SKIP (guid not found in episodes.json): {path}")
            continue
        e["transcript_status"] = "done"
        e["transcript_path"] = path
        e["transcript_word_count"] = fm["word_count"]
        e["transcript_language"] = fm["language"]
        e["tags"] = {
            "sectors": fm["sectors"],
            "themes": fm["themes"],
            "stocks": fm["stocks"],
            "sentiment": fm["sentiment"],
            "sentiment_hits": fm["sentiment_hits"],
            "sentiment_confidence": fm["sentiment_confidence"],
            "entity_mentions": fm["entity_mentions"],
            "entity_mention_density": fm["entity_mention_density"],
            "entity_sentiment": fm["entity_sentiment"],
            "entity_conviction": fm["entity_conviction"],
            "entity_stance": {},
            "entity_contested": [],
        }
        e["tags_source"] = "keyword"
        reconciled += 1
        print(f"Reconciled {guid} ({fm['word_count']} words): {path}")

    save_json(os.path.join(DATA_DIR, "episodes.json"), eps)
    taxonomy_raw = load_json(os.path.join(CONFIG_DIR, "taxonomy.json"), {})
    aggregates = build_aggregates(eps, taxonomy_raw)
    save_json(os.path.join(DATA_DIR, "aggregates.json"), aggregates)
    print(f"\nReconciled {reconciled}/{len(orphans)} orphaned transcripts. Rebuilt aggregates.json fresh.")


if __name__ == "__main__":
    main()
