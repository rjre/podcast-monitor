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

## Why this is (almost) free to run forever

The default pipeline never calls an LLM. It fetches each podcast's public RSS
feed, then tags every episode's **title + show notes** against a hand-built
financial taxonomy (`config/taxonomy.json`) using plain regex/keyword
matching — sectors, macro/market themes, and named stocks, plus a bullish/
bearish lexicon for a coarse sentiment score. All of that is stdlib Python:
no API key, no paid dependency, no per-run cost.

The "superior processing" only happened once, up front, to *build* the
taxonomy and lexicon thoughtfully. From here on, running it daily (or 10x a
day) costs nothing but a GitHub Actions minute.

An optional `pipeline/enrich_claude.py` module exists for when you want more
nuance than keyword-matching can give (e.g. a full transcript, sarcasm, a
guest disagreeing with the host) — it's wired for the Claude API but is never
called automatically. Turn it on deliberately, per-episode, when it's worth
the tokens.

**Important caveat**: tagging runs on episode titles and show notes, not full
audio transcripts (these feeds don't publish transcripts, and transcribing
hundreds of hours of audio isn't something to do by default). Treat the
output as a directional attention/tone indicator — worth investigating
further, not a source of truth on its own.

## Architecture

```
config/
  podcasts.json     - which shows to track (add/remove here)
  taxonomy.json     - sectors, themes, stocks + bullish/bearish lexicon
pipeline/
  fetch_feeds.py    - stdlib-only RSS parsing
  extract_themes.py - zero-token regex tagging + sentiment scoring
  enrich_claude.py  - OPTIONAL Claude API enrichment (off by default)
  run.py            - orchestrator: fetch -> tag -> write data/*.json
data/
  episodes.json     - every episode + its tags (the reusable dataset)
  aggregates.json   - precomputed mention counts / trends per entity
  state.json        - last run metadata
index.html, assets/ - the dashboard (static HTML/CSS/JS, no build step)
.github/workflows/update-podcasts.yml - scheduled fetch + commit
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

No dependencies to install for the default path — it's pure Python 3
standard library. `requirements.txt` only matters if you turn on
`enrich_claude.py` (needs the `anthropic` package + `ANTHROPIC_API_KEY`).

### Keeping it current automatically

`.github/workflows/update-podcasts.yml` runs the pipeline twice a day and
commits any new episodes back to the repo. Enable GitHub Pages (Settings →
Pages → deploy from the `main` branch, root) and the dashboard at `index.html`
always reflects the latest committed data — open it and hit **Refresh** to
pull the newest JSON.

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
