#!/usr/bin/env python3
"""Transcribe episode audio locally and re-tag from the full transcript.

Show notes are a thin signal -- a title and two sentences. The actual
transcript is where the real detail lives (what was actually said about a
stock, how a guest hedged their view, etc). This script downloads each
episode's audio (from the RSS <enclosure> URL) and runs it through a local
Whisper model (via faster-whisper, CPU-friendly, no API key/cost), then
re-runs the zero-token tagger over the *full transcript text* instead of
just the title/summary.

Transcripts are written as portable, self-contained Markdown files under
data/transcripts/ -- YAML frontmatter (episode metadata + tags) followed by
the transcript body -- so any other project can read a single file and get
everything, no database or cross-reference needed. See
data/transcripts/README.md for the format. This repo is private, so these
are committed alongside the rest of the data.

Analysis quality: when ANTHROPIC_API_KEY is set, each transcript is tagged
by Claude (pipeline/enrich_claude.py) -- full transcript, full reasoning,
real understanding of hedged views and sarcasm that keyword matching can't
catch. Without a key, it falls back automatically to the zero-token
keyword tagger (pipeline/extract_themes.py). The fallback exists so the
pipeline never hard-fails for lack of a key -- but Claude is the intended,
default-quality path, not an optional upgrade.

Usage:
    python3 pipeline/transcribe.py --limit 5           # shortest untranscribed episodes first
    python3 pipeline/transcribe.py --limit 3 --model small
    python3 pipeline/transcribe.py --guid "<specific-guid>"

Requires: pip install faster-whisper (not in the zero-dependency default
path -- this is the deliberate "spend local compute for more detail" step).
With ANTHROPIC_API_KEY set, also requires: pip install anthropic.
"""
import argparse
import json
import os
import random
import re
import sys
import tempfile
import urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from extract_themes import Taxonomy  # noqa: E402
from run import CONFIG_DIR, DATA_DIR, build_aggregates, load_json, save_json  # noqa: E402

TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
USER_AGENT = "podcast-monitor/1.0 (+https://github.com/rjre/podcast-monitor)"


def duration_seconds(raw):
    if not raw:
        return None
    raw = raw.strip()
    if re.fullmatch(r"\d+", raw):
        return int(raw)
    parts = raw.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return h * 3600 + m * 60 + s


def download_audio(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)


def transcribe_file(path, model_size="base", model=None):
    from faster_whisper import WhisperModel

    if model is None:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, beam_size=1, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip(), info


def slugify(text, max_len=60):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-")


def transcript_filename(ep):
    date = (ep.get("published_at") or "")[:10] or "undated"
    guid_suffix = re.sub(r"[^a-zA-Z0-9]", "", ep["guid"])[-6:]
    return f"{date}-{ep['podcast_id']}-{slugify(ep['title'])}-{guid_suffix}.md"


def _yaml_scalar(value):
    if isinstance(value, str):
        return json.dumps(value)  # reuse JSON string escaping -- valid YAML too
    return json.dumps(value)


def write_transcript_file(path, ep, text, tags, source, language=None):
    """Write a self-contained Markdown file: YAML frontmatter (metadata +
    tags) followed by the transcript body. Portable -- readable standalone
    by any tool, without needing data/episodes.json alongside it.
    """
    front = [
        "---",
        f"title: {_yaml_scalar(ep['title'])}",
        f"podcast: {_yaml_scalar(ep['podcast_name'])}",
        f"podcast_id: {_yaml_scalar(ep['podcast_id'])}",
        f"published_at: {_yaml_scalar(ep.get('published_at'))}",
        f"guid: {_yaml_scalar(ep['guid'])}",
        f"link: {_yaml_scalar(ep.get('link'))}",
        f"duration: {_yaml_scalar(ep.get('duration'))}",
        f"word_count: {len(text.split())}",
        f"language: {_yaml_scalar(language)}",
        f"tags_source: {_yaml_scalar(source)}",
        f"sectors: {json.dumps(tags['sectors'])}",
        f"themes: {json.dumps(tags['themes'])}",
        f"stocks: {json.dumps(tags['stocks'])}",
        f"sentiment: {tags['sentiment']}",
    ]
    # Only the keyword tagger produces these -- llm/claude-manual tags are a
    # single holistic read of the episode, not a per-sentence regex scan.
    if "sentiment_hits" in tags:
        front.append(f"sentiment_hits: {tags['sentiment_hits']}")
    if "sentiment_confidence" in tags:
        front.append(f"sentiment_confidence: {tags['sentiment_confidence']}")
    if tags.get("entity_mentions"):
        front.append(f"entity_mentions: {json.dumps(tags['entity_mentions'])}")
    if tags.get("entity_sentiment"):
        front.append(f"entity_sentiment: {json.dumps(tags['entity_sentiment'])}")
    if tags.get("summary"):
        front.append(f"summary: {_yaml_scalar(tags['summary'])}")
    front.append("---")
    front.append("")

    with open(path, "w") as f:
        f.write("\n".join(front))
        f.write(text)
        f.write("\n")


def tag_with_best_available(episode, text, taxonomy, taxonomy_raw):
    """Tag `text` with Claude if a key is configured, else the keyword
    tagger. Returns (tags_dict, source) where source is 'llm' or 'keyword'.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from enrich_claude import enrich_episode
            return enrich_episode(episode, text, taxonomy_raw), "llm"
        except Exception as exc:
            print(f"    Claude analysis failed ({exc}); falling back to keyword tagging", file=sys.stderr)
    return taxonomy.tag(text), "keyword"


def select_candidates(episodes, podcasts_cfg, limit, only_guid=None):
    """Pick episodes to transcribe next.

    Round-robin across podcasts (shortest-first within each show), not a
    flat global sort -- at 100+ podcasts, a handful of shows with naturally
    short episodes (daily 15-minute personal-finance shows, "At The Money"
    segments, etc.) would otherwise monopolize every batch, and shows with
    longer-format episodes would never get reached. Round-robin means a
    `--limit` at or above the number of active shows covers every show
    once before starting on anyone's second episode.
    """
    active_ids = {p["id"] for p in podcasts_cfg["podcasts"] if p.get("active", True)}
    candidates = [
        ep for ep in episodes
        if ep.get("podcast_id") in active_ids
        and ep.get("audio_url")
        and ep.get("transcript_status") != "done"
    ]
    if only_guid:
        return [ep for ep in candidates if ep["guid"] == only_guid]

    by_podcast = defaultdict(list)
    for ep in candidates:
        by_podcast[ep["podcast_id"]].append(ep)
    for pid in by_podcast:
        by_podcast[pid].sort(key=lambda ep: (duration_seconds(ep.get("duration")) or 1 << 30))

    # Shuffle which podcasts go first each run so that when --limit is
    # smaller than the number of shows with a backlog, coverage rotates
    # across runs instead of always favoring the same alphabetical shows.
    podcast_ids = list(by_podcast.keys())
    random.shuffle(podcast_ids)

    result = []
    round_idx = 0
    while len(result) < limit:
        added_any = False
        for pid in podcast_ids:
            if round_idx < len(by_podcast[pid]):
                result.append(by_podcast[pid][round_idx])
                added_any = True
                if len(result) >= limit:
                    break
        if not added_any:
            break
        round_idx += 1
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default="base", help="faster-whisper model size (tiny/base/small/medium)")
    parser.add_argument("--guid", default=None, help="Transcribe one specific episode by guid")
    args = parser.parse_args()

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

    podcasts_cfg = load_json(os.path.join(CONFIG_DIR, "podcasts.json"), {"podcasts": []})
    episodes = load_json(os.path.join(DATA_DIR, "episodes.json"), [])
    by_guid = {ep["guid"]: ep for ep in episodes}
    taxonomy_raw = load_json(os.path.join(CONFIG_DIR, "taxonomy.json"), {})
    taxonomy = Taxonomy(os.path.join(CONFIG_DIR, "taxonomy.json"))

    todo = select_candidates(episodes, podcasts_cfg, args.limit, only_guid=args.guid)
    if not todo:
        print("Nothing to transcribe (no untranscribed episodes with an audio_url).")
        return

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    for i, ep in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {ep['podcast_name']}: {ep['title']}")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            download_audio(ep["audio_url"], tmp_path)
            text, info = transcribe_file(tmp_path, model=model)
        except Exception as exc:
            print(f"    failed: {exc}", file=sys.stderr)
            continue
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not text:
            print("    empty transcript, skipping")
            continue

        combined_text = f"{ep['title']}. {ep.get('summary', '')} {text}"
        tags, source = tag_with_best_available(ep, combined_text, taxonomy, taxonomy_raw)
        language = getattr(info, "language", None)

        filename = transcript_filename(ep)
        transcript_path = os.path.join(TRANSCRIPT_DIR, filename)
        write_transcript_file(transcript_path, ep, text, tags, source, language=language)

        ep["tags"] = {k: tags[k] for k in ("sectors", "themes", "stocks", "sentiment")}
        for extra in ("sentiment_hits", "sentiment_confidence", "entity_mentions", "entity_sentiment"):
            if extra in tags:
                ep["tags"][extra] = tags[extra]
        ep["tags_source"] = source
        if tags.get("summary"):
            ep["llm_summary"] = tags["summary"]
        if tags.get("new_entities"):
            ep["llm_new_entities"] = tags["new_entities"]
        ep["transcript_status"] = "done"
        ep["transcript_path"] = f"data/transcripts/{filename}"
        ep["transcript_word_count"] = len(text.split())
        ep["transcript_language"] = language
        by_guid[ep["guid"]] = ep
        print(f"    [{source}] {ep['transcript_word_count']} words -> "
              f"sectors={tags['sectors']} themes={tags['themes']} stocks={tags['stocks']} sentiment={tags['sentiment']}")

    all_episodes = sorted(by_guid.values(), key=lambda e: e.get("published_at") or "", reverse=True)
    save_json(os.path.join(DATA_DIR, "episodes.json"), all_episodes)

    aggregates = build_aggregates(all_episodes, taxonomy_raw)
    import datetime
    aggregates["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_json(os.path.join(DATA_DIR, "aggregates.json"), aggregates)

    print(f"\nDone. Transcripts saved under {TRANSCRIPT_DIR} as portable Markdown files.")


if __name__ == "__main__":
    main()
