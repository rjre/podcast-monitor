# Podcast Monitor

Turns financial podcasts into a structured, filterable signal feed: what
themes, sectors and stocks are being talked about, how often, and with what
tone — across shows, over time.

**Live idea**: podcasters surface what's on institutional/retail minds before
it shows up in flows data. Tracking mention frequency + tone by
theme/sector/stock over time is a cheap, repeatable way to watch for that.

**Tracked out of the box:**
- [Odd Lots](https://www.bloomberg.com/podcasts/series/odd-lots) (Bloomberg)
- [Money Stuff: The Podcast](https://www.bloomberg.com/podcasts/series/money-stuff) (Bloomberg)
- [Unhedged](https://shows.acast.com/unhedged) (Financial Times)
- [Masters in Business](https://www.bloomberg.com/podcasts/series/master-in-business) (Bloomberg)

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

**Full transcript + Claude** (`pipeline/transcribe.py` +
`pipeline/enrich_claude.py`) is the real analysis path, used whenever
`ANTHROPIC_API_KEY` is set: it downloads an episode's audio, transcribes it
locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CPU,
free), then sends the **full transcript** to Claude Opus 5 to tag it against
the same taxonomy — catching a hedged view, sarcasm, or a guest disagreeing
with the host, none of which keyword matching can do — plus a one/two-sentence
summary of the investment-relevant takeaway. This is intentionally the
higher-quality default when a key is available, not a cost-saving compromise:
it runs at Claude's full reasoning effort rather than a cheaper, shallower
model.

```bash
pip install -r pipeline/requirements.txt   # faster-whisper + anthropic

export ANTHROPIC_API_KEY=sk-ant-...        # enables the Claude analysis path

python3 pipeline/transcribe.py --limit 5           # shortest untranscribed episodes first
python3 pipeline/transcribe.py --limit 3 --model small   # more accurate Whisper pass, slower
python3 pipeline/transcribe.py --guid "<guid>"     # one specific episode
```

It picks the shortest not-yet-transcribed episodes first (fastest path to
broad coverage) and writes each transcript as a self-contained Markdown file
under `data/transcripts/` (frontmatter + full text — see
[`data/transcripts/README.md`](data/transcripts/README.md) for the format and
how to consume it from another project), and re-tags the episode in
`data/episodes.json` (`"transcript_status": "done"`, `"tags_source": "llm"`
or `"keyword"`, `"transcript_path"` pointing at its file, and — with Claude —
`"llm_summary"`). The scheduled GitHub Action transcribes a few more episodes
every day, so the backlog across all four shows fills in gradually with no
manual work; add `ANTHROPIC_API_KEY` as a repository secret to have it use
Claude too.

**Copyright note**: `data/transcripts/` is committed because this repo is
**private** — the underlying audio is copyrighted commercial podcast content,
so this only works because access is restricted to you (and anyone you
invite). Don't flip this repo back to public without reconsidering that.

**Important caveat**: even with Claude, treat every signal here as directional
and worth a second look, not a source of truth on its own — always check the
source episode before acting.

## Architecture

```
config/
  podcasts.json     - which shows to track (add/remove here)
  taxonomy.json     - sectors, themes, stocks + bullish/bearish lexicon
pipeline/
  fetch_feeds.py    - stdlib-only RSS parsing
  extract_themes.py - zero-token regex tagging + sentiment scoring (fallback)
  transcribe.py     - local Whisper transcription + re-tagging orchestration
  enrich_claude.py  - Claude-powered tagging from full transcripts (default
                      whenever ANTHROPIC_API_KEY is set)
  run.py            - orchestrator: fetch -> tag -> write data/*.json
data/
  episodes.json     - every episode + its tags (the reusable dataset)
  aggregates.json   - precomputed mention counts / trends per entity
  state.json        - last run metadata
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
    "sentiment": 0.4
  }
}
```

`data/aggregates.json` gives you the same thing pre-rolled-up per entity
(total mentions, average sentiment, monthly time series) if you just want
the summary.

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

`.github/workflows/update-podcasts.yml` runs the pipeline twice a day and
commits any new episodes back to the repo. Enable GitHub Pages (Settings →
Pages → deploy from the `main` branch, root) and the dashboard at `index.html`
always reflects the latest committed data — open it and hit **Refresh** to
pull the newest JSON.

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
