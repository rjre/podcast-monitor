#!/usr/bin/env python3
"""One-off backfill: populate episode_summary/episode_summary_source for
transcribed episodes that predate this feature.

- claude-manual/llm episodes: promote the existing genuine ep["llm_summary"].
- keyword episodes: generate a template summary from their existing tags
  (same zero-token generate_template_summary() used going forward by
  transcribe.py/retag.py, so re-running this later is a no-op unless tags
  changed).

Safe to re-run: only touches episodes whose computed value differs from
what's already stored. Untranscribed episodes (no tags yet) are skipped --
nothing to summarize from.

Usage:
    python3 pipeline/backfill_summaries.py
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from insights import build_label_maps, generate_template_summary  # noqa: E402
from run import CONFIG_DIR, DATA_DIR, build_aggregates, load_json, save_json  # noqa: E402


def main():
    taxonomy_raw = load_json(os.path.join(CONFIG_DIR, "taxonomy.json"), {})
    labels = build_label_maps(taxonomy_raw)
    episodes = load_json(os.path.join(DATA_DIR, "episodes.json"), [])

    updated_genuine, updated_template = 0, 0
    for ep in episodes:
        if ep.get("transcript_status") != "done":
            continue
        source = ep.get("tags_source")
        tags = ep.get("tags") or {}

        if source in ("claude-manual", "llm"):
            genuine = ep.get("llm_summary")
            if genuine and (ep.get("episode_summary") != genuine or ep.get("episode_summary_source") != source):
                ep["episode_summary"] = genuine
                ep["episode_summary_source"] = source
                updated_genuine += 1
        elif source == "keyword":
            generated = generate_template_summary(tags, labels)
            if ep.get("episode_summary") != generated or ep.get("episode_summary_source") != "template":
                ep["episode_summary"] = generated
                ep["episode_summary_source"] = "template"
                updated_template += 1

    save_json(os.path.join(DATA_DIR, "episodes.json"), episodes)

    aggregates = build_aggregates(episodes, taxonomy_raw)
    aggregates["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_json(os.path.join(DATA_DIR, "aggregates.json"), aggregates)

    print(f"Backfilled episode_summary: {updated_genuine} promoted from genuine llm_summary, "
          f"{updated_template} generated from tags.")


if __name__ == "__main__":
    main()
