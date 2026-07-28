# Podcast Monitor

Turns financial podcasts into a structured, filterable signal feed: what
themes, sectors and stocks are being talked about, how often, and with what
tone — across shows, over time.

**Live idea**: podcasters surface what's on institutional/retail minds before
it shows up in flows data. Tracking mention frequency + tone by
theme/sector/stock over time is a cheap, repeatable way to watch for that.

**Tracked out of the box:** 101 podcasts across 6 categories (see
`config/categories.json`) — macro/markets commentary (Odd Lots, Money Stuff,
Unhedged, Masters in Business), personal finance & budgeting, investing &
stock markets, economics/macro/corporate finance, retirement & FIRE, and real
estate/crypto/alternative assets. Full list in `config/podcasts.json`.

Add or remove shows any time in `config/podcasts.json` — see [Managing podcasts](#managing-podcasts).

## Two analysis paths: free keyword tagging, or Claude for real understanding

**Show notes + keyword tagging** (`pipeline/run.py`) is the always-on, zero-cost
baseline. It fetches each podcast's public RSS feed, then tags every episode's
**title + show notes** against a hand-built financial taxonomy
(`config/taxonomy.json`) using plain regex/keyword matching — sectors,
macro/market themes, and named stocks, plus a bullish/bearish lexicon for a
coarse sentiment score. All stdlib Python: no API key, no paid dependency, no
per-run cost. It's what keeps the dashboard populated with zero ongoing spend,
and it's the automatic fallback whenever a better option isn't configured.

**Full transcript + Claude analysis** is the real analysis path — it catches a
hedged view, sarcasm, or a guest disagreeing with the host, none of which
keyword matching can do, plus a one/two-sentence summary of the
investment-relevant takeaway. There are two ways to get it, and this project
deliberately uses the free one:

- `pipeline/enrich_claude.py` calls the Anthropic API directly whenever
  `ANTHROPIC_API_KEY` is set — fully automatic, but billed per token on a
  separate, metered Anthropic Console account (a different bill from a
  Claude.ai subscription). **Not used here** — no budget for a second metered
  bill on top of an existing Claude plan.
- `pipeline/manual_review.py` gets the same quality of tagging for free by
  having a Claude Code session (covered by your existing Claude plan, not
  billed per token) do the analysis by hand, in batches, on whatever cadence
  you like. **This is the path this repo uses** — see
  [Getting Claude-quality tags for free](#getting-claude-quality-tags-for-free-no-api-key)
  below.

Local transcription itself (`pipeline/transcribe.py`) is always free either
way — it downloads an episode's audio and transcribes it locally with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CPU, no API, no
cost), then tags the transcript with the free keyword tagger (or Claude, if
`ANTHROPIC_API_KEY` happens to be set):

```bash
pip install -r pipeline/requirements.txt   # faster-whisper (anthropic is unused unless you set a key)

python3 pipeline/transcribe.py --limit 5           # shortest untranscribed episodes first
python3 pipeline/transcribe.py --limit 3 --model small   # more accurate Whisper pass, slower
python3 pipeline/transcribe.py --guid "<guid>"     # one specific episode
```

It picks the shortest not-yet-transcribed episodes first (fastest path to
broad coverage) and writes each transcript as a self-contained Markdown file
under `data/transcripts/` (frontmatter + full text — see
[`data/transcripts/README.md`](data/transcripts/README.md) for the format and
how to consume it from another project), and re-tags the episode in
`data/episodes.json` (`"transcript_status": "done"`, `"tags_source"` set to
`"keyword"` until a manual review pass upgrades it to `"claude-manual"`, and
`"transcript_path"` pointing at its file). The scheduled GitHub Action
transcribes a few more episodes every day for free, so the transcript backlog
fills in on its own — the weekly manual pass below is what upgrades those
transcripts' *tags* from keyword-grade to Claude-grade.

**Copyright note**: `data/transcripts/` is committed because this repo is
**private** — the underlying audio is copyrighted commercial podcast content,
so this only works because access is restricted to you (and anyone you
invite). Don't flip this repo back to public without reconsidering that.

**Important caveat**: even with Claude, treat every signal here as directional
and worth a second look, not a source of truth on its own — always check the
source episode before acting.

## Getting Claude-quality tags for free (no API key)

`pipeline/manual_review.py` stages a batch of transcribed-but-not-yet-Claude-
tagged episodes for a Claude Code session to tag by hand — same taxonomy,
same fields `enrich_claude.py` would have produced, just written by Claude
directly during a session instead of via a metered API call. That usage is
covered by your existing Claude plan, so this path has no incremental cost
beyond however often you choose to run it. Weekly works well:
`pipeline/transcribe.py` only adds a handful of new transcripts per day, so
the backlog rarely grows faster than a weekly pass can clear it.

```bash
python3 pipeline/manual_review.py list --limit 15
```

This writes `data/manual_review/pending.json`: each pending episode's
metadata and `transcript_path`, the taxonomy's allowed sector/theme/stock
ids, and an empty `sectors`/`themes`/`stocks`/`sentiment`/`summary` template
per episode. Then, in a Claude Code session, ask something like:

> Read the transcripts listed in `data/manual_review/pending.json` and fill
> in sectors/themes/stocks/sentiment/summary for each, using only ids from
> the allowed lists in that file.

Once it's filled in, apply it back into the dataset:

```bash
python3 pipeline/manual_review.py apply data/manual_review/pending.json
```

This validates every id against `config/taxonomy.json` (dropping anything
hallucinated), updates `data/episodes.json` and the matching
`data/transcripts/*.md` frontmatter (`tags_source: "claude-manual"`, plus the
summary), and rebuilds `data/aggregates.json`. Commit and push the result
(`data/episodes.json`, `data/aggregates.json`, `data/transcripts/`) same as
any other pipeline run. Repeat weekly (or whenever) until the backlog is
clear — newly transcribed episodes join the pending list automatically.

## Architecture

```
config/
  podcasts.json     - which shows to track (add/remove here)
  taxonomy.json     - sectors, themes, stocks + bullish/bearish lexicon
pipeline/
  fetch_feeds.py    - stdlib-only RSS parsing
  extract_themes.py - zero-token regex tagging + sentiment scoring (fallback)
  transcribe.py     - local Whisper transcription + re-tagging orchestration
  enrich_claude.py  - Claude API tagging from full transcripts (only used if
                      you set ANTHROPIC_API_KEY -- not used by default, see
                      "Getting Claude-quality tags for free" above)
  manual_review.py  - stages/applies a free, no-API-key Claude Code tagging
                      pass over transcribed episodes (the recommended path)
  run.py            - orchestrator: fetch -> tag -> write data/*.json
data/
  episodes.json     - every episode + its tags (the reusable dataset)
  aggregates.json   - precomputed mention counts / trends per entity
  state.json        - last run metadata
  manual_review/    - working file for the manual Claude tagging pass
                      (pending.json; not meaningful once applied)
  transcripts/      - full transcript text, one Markdown file per episode
                      (committed -- repo is private; see its own README)
index.html, assets/ - static dashboard (HTML/CSS/JS, no build step)
streamlit_app.py    - Streamlit dashboard (same data, deploy on Streamlit Cloud)
.github/workflows/update-podcasts.yml - scheduled fetch + transcribe + commit
```

### `data/episodes.json` — reuse this elsewhere

This is the file meant to be reused in other projects, so nobody has to
re-run the fetch/tag pipeline to answer "what's been said about X":

```json
{
  "guid": "...",
  "podcast_id": "odd-lots",
  "podcast_name": "Odd Lots",
  "title": "...",
  "link": "https://...",
  "published_at": "2026-07-24T09:00:00+00:00",
  "summary": "...show notes text...",
  "duration": "38:12",
  "tags": {
    "sectors": ["semiconductors"],
    "themes": ["ai-boom"],
    "stocks": ["nvidia"],
    "sentiment": 0.4,
    "sentiment_hits": 9,
    "sentiment_confidence": 1.0,
    "entity_mentions": {"semiconductors": 3, "ai-boom": 5, "nvidia": 7},
    "entity_sentiment": {"nvidia": 0.6}
  }
}
```

`sentiment_hits`/`sentiment_confidence` (keyword tagger only) say how much
lexicon signal the overall `sentiment` score is based on -- the raw
bull/bear ratio is shrunk toward 0.0 below ~6 hits so one stray word in a
long transcript can't read as a full-strength +/-1.0. `entity_mentions` is
a per-entity hit count within the episode (salience/intensity, not just
presence), and `entity_sentiment` is the tone in the sentences immediately
around each entity's mentions, not just the episode-wide average -- so an
episode that's bearish on banks but bullish on AI infra doesn't flatten
into one misleading number. Both are populated by the keyword tagger; an
`llm`/`claude-manual` tagging pass gives one holistic `sentiment` +
`summary` instead.

`data/aggregates.json` gives you the same thing pre-rolled-up per entity:
total mentions, `total_hits` (salience-weighted, sums entity_mentions
rather than counting one per episode), average sentiment,
`sentiment_divergence` (how far an entity's average tone sits from the
dataset-wide baseline -- flags what's unusually bullish/bearish right now),
`momentum_pct`/`trend` (mentions in the trailing 30 days vs. the 30 days
before that: `rising`/`falling`/`flat`/`new`/`insufficient-data`), and the
monthly time series.

## Running it

```bash
# one-off: pull the last ~90 days across all active podcasts
python3 pipeline/run.py --backfill-days 90

# incremental (what the scheduled job does): only new episodes since last run
python3 pipeline/run.py

# after editing config/taxonomy.json, re-tag everything with the new rules
python3 pipeline/run.py --retag
```

No dependencies to install for the fetch+tag path — it's pure Python 3
standard library. `pipeline/requirements.txt` covers transcription
(`faster-whisper`) and Claude analysis (`anthropic`); the root
`requirements.txt` is just what the Streamlit dashboard needs to run.

### Keeping it current automatically

`.github/workflows/update-podcasts.yml` runs the fetch+tag job twice a day
and the transcribe job once a day, committing any new episodes/transcripts
back to the repo. Enable GitHub Pages (Settings → Pages → deploy from the
`main` branch, root) and the static dashboard at `index.html` always
reflects the latest committed data — open it and hit **Refresh** to pull
the newest JSON.

### Coverage timeline for the transcript backlog

Episode selection is round-robin across shows (see `select_candidates` in
`pipeline/transcribe.py`), shortest-episode-first per show — not a flat
global sort, so a handful of naturally-short shows can't monopolize every
batch. At the workflow's current `--limit 15`, every one of the ~97 active
podcasts gets at least one episode transcribed within about 7 days; after
that first pass, it moves on to each show's next-shortest episode, and so
on. Clearing the *entire* backlog (all episodes across all shows, not just
one per show) is a much bigger number — with ~3,300+ episodes total across
101 podcasts, even at 15/day that's the better part of a year; the
round-robin pass is what gives broad coverage quickly, not full history per
show. Run `pipeline/transcribe.py --limit N` manually any time you want to
speed that up further — it doesn't consume GitHub Actions minutes at all
when run locally.

Transcription itself is free (local Whisper), so the only thing that gates
tag *quality* going from keyword-grade to Claude-grade is running
`pipeline/manual_review.py` (see above) on whatever cadence you like — it
doesn't need to keep pace with transcription day-by-day, just often enough
that the backlog doesn't grow unbounded.

**GitHub Actions minutes.** This repo is private, so Actions minutes count
against your plan's monthly allowance (GitHub Free includes 2,000 min/month
for private repos; Pro/Team get 3,000; Enterprise gets 50,000). Transcribing
on the `base` Whisper model runs roughly 5.5–9 minutes/episode, so:

| `--limit` | minutes/day | minutes/month (~30 days) |
|---|---|---|
| 8  | 45–70   | ~1,350–2,100 |
| 15 (current) | 84–131 | **~2,500–3,900** |

At 15/day this is likely to **exceed** GitHub Free's 2,000 min/month
allowance, and may exceed Team's 3,000 too in a bad month. What happens
next depends on your GitHub billing settings (Settings → Billing → Plans
and usage, or the Actions usage page):
- **No spending limit configured (GitHub's default)**: once the included
  minutes run out, further workflow runs simply fail to start until the
  allowance resets next month — no charge, but the transcribe job silently
  stops working for the rest of the billing cycle.
- **A spending limit above $0 is configured**: extra minutes are billed at
  roughly $0.008/minute for Linux runners — at this pace, on the order of
  **$4–15/month** in overage on top of whatever plan you're on.

Worth checking your actual billing settings before relying on 15/day
running every day of the month. If you'd rather stay safely inside the
free allowance, lower `--limit` back down (e.g. to 8–10) and the coverage
timeline above just stretches out proportionally.

## Two dashboards, same data

- **`index.html`** — static HTML/CSS/JS, no build step, deploy via GitHub Pages.
- **`streamlit_app.py`** — a Streamlit app with the same filters/charts, plus
  richer Altair visualizations. Run locally:

  ```bash
  pip install -r requirements.txt
  streamlit run streamlit_app.py
  ```

  **Deploying to Streamlit Community Cloud** (share.streamlit.io): sign in with
  GitHub, click "New app", pick `rjre/podcast-monitor`, branch `main`, main
  file path `streamlit_app.py`, and deploy — it installs `requirements.txt`
  automatically. This step needs your Streamlit account, so it isn't something
  that can be done on your behalf; everything in the repo is ready for it.

## Using the dashboard

- **Overview** — episode/mention counts, net sentiment, fastest-rising theme
  in the last 30 days, top themes/sectors/stocks, and a 6-month mentions
  trend by podcast.
- **Episodes** — the filtered episode list with tags and tone per episode,
  linking out to the original.
- **Filters** (top bar, apply everywhere): toggle individual podcasts, filter
  to one theme/sector/stock (or click any bar/tag to jump straight to it),
  a date-range preset, and free-text search over titles/show notes.
- **Refresh** — re-pulls the JSON files (cache-busted) and re-renders; it
  does not itself run the pipeline (that happens via the scheduled Action or
  `pipeline/run.py`).

## Managing podcasts

Open **Manage podcasts** in the dashboard:
- The toggle next to each show hides/shows it *on your device only*
  (localStorage) — handy for personal decluttering without affecting anyone
  else or the underlying data.
- The **Add a podcast** form generates the JSON entry to paste into
  `config/podcasts.json`; commit it and the next pipeline run picks it up.
- To permanently remove a show, delete its entry from `config/podcasts.json`
  (or set `"active": false` to stop fetching new episodes while keeping its
  history in `data/episodes.json`).

Finding an RSS feed URL for a new show: search `<podcast name> RSS feed`, or
grab its Apple Podcasts ID from the podcasts.apple.com URL and query
`https://itunes.apple.com/lookup?id=<id>&entity=podcast` — the `feedUrl`
field in the response is what goes in `feed_url`.

### Categories

Every podcast entry needs a `category` matching one of the ids in
`config/categories.json` (`personal-finance`, `investing-markets`,
`macro-economics`, `retirement-fire`, `real-estate-crypto`, or
`markets-commentary`). Both dashboards color mentions-by-month by category
rather than by individual show — at 100+ podcasts, a distinct color per show
stops being readable, but 6 categories works fine as a stacked chart. Add a
new category by adding an entry to `config/categories.json` (give it a hex
color not already in use) before referencing its id from `podcasts.json`.

### Not found / excluded from the 100-podcast batch add

Three requested shows couldn't be added confidently and were left out rather
than guessed wrong:
- **You Need A Budget (YNAB)** — no official standalone feed found on Apple
  Podcasts under that name.
- **Joney Talks** — no matching podcast found; possibly a typo of a
  different title.
- **Bitcoin Fundamentals** — search kept resolving to an unrelated show
  (The Investor's Podcast); no standalone feed found under this name.

If you have a direct link (Apple Podcasts, Spotify, or the show's own site)
for any of these, share it and I'll add it properly.

## Extending the taxonomy

`config/taxonomy.json` is plain, editable JSON — add a `terms` phrase to an
existing sector/theme/stock to improve recall, or add a whole new entry.
Terms match case-insensitively as whole words/phrases (so "AI" won't match
inside "said", and multi-word phrases like "private credit" match as a
unit). Run `pipeline/run.py --retag` afterwards to apply the new rules to
existing episodes.

## Disclaimer

This is a research/attention-tracking tool built from public podcast show
notes. It is not investment advice, and keyword-based sentiment is a blunt
instrument — always read the episode before acting on a signal.
