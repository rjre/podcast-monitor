#!/usr/bin/env python3
"""Transcribe episode audio locally and re-tag from the full transcript.

Show notes are a thin signal -- a title and two sentences. The actual
transcript is where the real detail lives (what was actually said about a
stock, how a guest hedged their view, etc). This script downloads each
episode's audio (from the RSS <enclosure> URL) and runs it through a local
Whisper model (via faster-whisper, CPU-friendly, no API key/cost), then
re-runs the zero-token tagger over the *full transcript text* instead of
just the title/summary.

Copyright note: full transcript text is intentionally kept OUT of git (see
data/transcripts/ in .gitignore) -- these are copyrighted commercial
podcasts and this repo is public. Transcripts stay local for your own
research; only the derived tags/sentiment (facts about what was discussed,
not the podcast's own words) are committed to data/episodes.json. If you
make this repo private, you can safely stop gitignoring data/transcripts/.

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
import os
import re
import sys
import tempfile
import urllib.request

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


def safe_filename(guid):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", guid)[:120]


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
    active_ids = {p["id"] for p in podcasts_cfg["podcasts"] if p.get("active", True)}
    candidates = [
        ep for ep in episodes
        if ep.get("podcast_id") in active_ids
        and ep.get("audio_url")
        and ep.get("transcript_status") != "done"
    ]
    if only_guid:
        return [ep for ep in candidates if ep["guid"] == only_guid]
    candidates.sort(key=lambda ep: (duration_seconds(ep.get("duration")) or 1 << 30))
    return candidates[:limit]


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

        transcript_path = os.path.join(TRANSCRIPT_DIR, safe_filename(ep["guid"]) + ".txt")
        with open(transcript_path, "w") as f:
            f.write(text)

        combined_text = f"{ep['title']}. {ep.get('summary', '')} {text}"
        tags, source = tag_with_best_available(ep, combined_text, taxonomy, taxonomy_raw)
        ep["tags"] = {k: tags[k] for k in ("sectors", "themes", "stocks", "sentiment")}
        ep["tags_source"] = source
        if "summary" in tags:
            ep["llm_summary"] = tags["summary"]
        if tags.get("new_entities"):
            ep["llm_new_entities"] = tags["new_entities"]
        ep["transcript_status"] = "done"
        ep["transcript_word_count"] = len(text.split())
        ep["transcript_language"] = getattr(info, "language", None)
        by_guid[ep["guid"]] = ep
        print(f"    [{source}] {ep['transcript_word_count']} words -> "
              f"sectors={tags['sectors']} themes={tags['themes']} stocks={tags['stocks']} sentiment={tags['sentiment']}")

    all_episodes = sorted(by_guid.values(), key=lambda e: e.get("published_at") or "", reverse=True)
    save_json(os.path.join(DATA_DIR, "episodes.json"), all_episodes)

    aggregates = build_aggregates(all_episodes, taxonomy_raw)
    import datetime
    aggregates["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_json(os.path.join(DATA_DIR, "aggregates.json"), aggregates)

    print(f"\nDone. Transcripts saved locally under {TRANSCRIPT_DIR} (not committed -- see .gitignore).")


if __name__ == "__main__":
    main()
