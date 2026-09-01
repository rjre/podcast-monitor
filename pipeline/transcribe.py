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
from insights import build_label_maps, generate_template_summary  # noqa: E402
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
    # Every podcast in config/podcasts.json is English-language. Without
    # language="en", Whisper's language auto-detection occasionally locks
    # onto a short noisy/musical segment and misdetects the whole episode
    # as some other language, which then decodes as a handful of garbled
    # words instead of a real transcript (near-total transcription failure
    # that still silently reports success).
    segments, info = model.transcribe(path, beam_size=1, vad_filter=True, language="en")
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip(), info


MIN_AUDIO_COVERAGE = 0.85  # decoded audio must cover at least this fraction of the RSS-declared duration


def download_and_transcribe(ep, model, max_attempts=3):
    """Download + transcribe with a retry loop guarding against truncated
    downloads. A stalled/short-circuited HTTP transfer produces a valid but
    short mp3 file -- transcribe_file() then "succeeds" on just the first
    few minutes (e.g. an intro ad) while reporting transcript_status: done,
    silently discarding the rest of the episode. faster-whisper's TranscriptionInfo
    exposes the decoded audio's actual duration, so compare that against the
    RSS <duration> and retry the download if it's suspiciously short instead
    of trusting the transcription just because it didn't raise.
    """
    expected = duration_seconds(ep.get("duration"))
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            download_audio(ep["audio_url"], tmp_path)
            text, info = transcribe_file(tmp_path, model=model)
        except Exception as exc:
            last_exc = exc
            print(f"    attempt {attempt}/{max_attempts} failed ({exc}); retrying", file=sys.stderr)
            continue
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        decoded_duration = getattr(info, "duration", None)
        if expected and decoded_duration and decoded_duration < MIN_AUDIO_COVERAGE * expected:
            last_exc = RuntimeError(
                f"truncated download: decoded {decoded_duration:.0f}s of {expected}s expected"
            )
            print(f"    attempt {attempt}/{max_attempts}: {last_exc}; retrying", file=sys.stderr)
            continue

        return text, info
    raise last_exc or RuntimeError("transcription failed after retries")


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
    if tags.get("entity_mention_density"):
        front.append(f"entity_mention_density: {json.dumps(tags['entity_mention_density'])}")
    if tags.get("entity_sentiment"):
        front.append(f"entity_sentiment: {json.dumps(tags['entity_sentiment'])}")
    if tags.get("entity_conviction"):
        front.append(f"entity_conviction: {json.dumps(tags['entity_conviction'])}")
    if tags.get("entity_stance"):
        front.append(f"entity_stance: {json.dumps(tags['entity_stance'])}")
    if tags.get("entity_contested"):
        front.append(f"entity_contested: {json.dumps(tags['entity_contested'])}")
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


def select_candidates(episodes, podcasts_cfg, limit, only_guid=None, only_podcast_ids=None):
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
    if only_podcast_ids:
        active_ids &= set(only_podcast_ids)
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
    parser.add_argument("--podcast-ids", default=None,
                         help="Comma-separated podcast_id values to restrict this batch to")
    args = parser.parse_args()
    only_podcast_ids = [p.strip() for p in args.podcast_ids.split(",")] if args.podcast_ids else None

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

    # OpenRouter ASR (transcribe_openrouter.py) loads .env itself and is used
    # automatically whenever OPENROUTER_API_KEY ends up set -- same
    # "automatic if the key exists" convention as ANTHROPIC_API_KEY above.
    # Cloud, so it doesn't compete with local Whisper for this machine's CPU,
    # and runs in seconds/episode instead of minutes.
    try:
        import transcribe_openrouter
        use_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    except ImportError:
        use_openrouter = False

    podcasts_cfg = load_json(os.path.join(CONFIG_DIR, "podcasts.json"), {"podcasts": []})
    episodes = load_json(os.path.join(DATA_DIR, "episodes.json"), [])
    by_guid = {ep["guid"]: ep for ep in episodes}
    taxonomy_raw = load_json(os.path.join(CONFIG_DIR, "taxonomy.json"), {})
    taxonomy = Taxonomy(os.path.join(CONFIG_DIR, "taxonomy.json"))
    labels = build_label_maps(taxonomy_raw)

    todo = select_candidates(episodes, podcasts_cfg, args.limit, only_guid=args.guid,
                              only_podcast_ids=only_podcast_ids)
    if not todo:
        print("Nothing to transcribe (no untranscribed episodes with an audio_url).")
        return

    if use_openrouter:
        model = None
        print(f"Using OpenRouter ASR ({transcribe_openrouter.MODEL}) -- OPENROUTER_API_KEY is set")
    else:
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device="cpu", compute_type="int8")

    for i, ep in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {ep['podcast_name']}: {ep['title']}")
        try:
            if use_openrouter:
                text, info = transcribe_openrouter.download_and_transcribe(ep)
            else:
                text, info = download_and_transcribe(ep, model)
        except Exception as exc:
            print(f"    failed: {exc}", file=sys.stderr)
            continue

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
        for extra in ("sentiment_hits", "sentiment_confidence", "entity_mentions", "entity_mention_density",
                      "entity_sentiment", "entity_conviction", "entity_stance", "entity_contested"):
            if extra in tags:
                ep["tags"][extra] = tags[extra]
        ep["tags_source"] = source
        if tags.get("summary"):
            ep["llm_summary"] = tags["summary"]
            ep["episode_summary"] = tags["summary"]
            ep["episode_summary_source"] = source
        else:
            ep["episode_summary"] = generate_template_summary(ep["tags"], labels)
            ep["episode_summary_source"] = "template"
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
