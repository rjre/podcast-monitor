"""Podcast Monitor -- Streamlit dashboard.

Reads the same static JSON the zero-token pipeline produces
(config/podcasts.json, config/taxonomy.json, data/episodes.json) -- no
API calls, no database. Deploy as-is on Streamlit Community Cloud by
pointing it at this repo/file; locally: `streamlit run streamlit_app.py`.

Note on transcripts: data/transcripts/ holds the full episode text as
self-contained Markdown files (see its own README) -- committed because
this repo is private. This app itself only reads the derived analysis
(tags, sentiment, word counts) from data/episodes.json, not the transcript
files directly.
"""
import json
import os
from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")
DATA_DIR = os.path.join(ROOT, "data")

# Validated categorical palette (dataviz skill reference palette), four
# slots chosen to skip the one pairing flagged as unsafe (yellow/orange).
PODCAST_FALLBACK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
NEUTRAL = "#898781"

# Same reference palette, extended to 8 slots (in validated order) for
# entity time-series lines, where more than 4 series can appear at once.
ENTITY_LINE_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

st.set_page_config(page_title="Podcast Monitor", page_icon="\U0001F4C8", layout="wide")


def inject_style():
    st.markdown(
        """
        <style>
        .stApp { background: #f7f5ef; }
        h1, h2, h3 { font-family: Georgia, "Times New Roman", serif !important; font-weight: 400 !important; }
        div[data-testid="stMetricValue"] { font-size: 28px; }
        .tag-pill {
            display:inline-block; font-size:11.5px; padding:2px 9px; margin:2px 3px 2px 0;
            border-radius:999px; background:#ffffff; border:1px solid #e1e0d9; color:#52514e;
        }
        .sent-badge { font-size:11.5px; font-weight:600; padding:2px 9px; border-radius:999px; }
        .sent-bullish { color:#0ca30c; background:rgba(12,163,12,0.12); }
        .sent-bearish { color:#d03b3b; background:rgba(208,59,59,0.12); }
        .sent-neutral { color:#898781; background:rgba(137,135,129,0.12); }
        .ep-row { border-bottom:1px solid #e1e0d9; padding:10px 0; }
        .transcript-badge {
            font-size: 10.5px; color:#1f4d3a; background:rgba(31,77,58,0.10);
            padding:1px 7px; border-radius:999px; margin-left:6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300)
def load_data():
    with open(os.path.join(CONFIG_DIR, "podcasts.json")) as f:
        podcasts = json.load(f)["podcasts"]
    with open(os.path.join(CONFIG_DIR, "taxonomy.json")) as f:
        taxonomy = json.load(f)
    with open(os.path.join(DATA_DIR, "episodes.json")) as f:
        episodes = json.load(f)
    state = {}
    state_path = os.path.join(DATA_DIR, "state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    return podcasts, taxonomy, episodes, state


def label_maps(taxonomy):
    return {kind: {e["id"]: e["label"] for e in taxonomy[kind]} for kind in ("sectors", "themes", "stocks")}


def podcast_color(podcast, idx):
    return podcast.get("color") or PODCAST_FALLBACK_COLORS[idx % len(PODCAST_FALLBACK_COLORS)]


def sentiment_word(score):
    if score > 0.15:
        return "Bullish", "sent-bullish"
    if score < -0.15:
        return "Bearish", "sent-bearish"
    return "Neutral", "sent-neutral"


def episodes_to_df(episodes, podcasts_by_id):
    rows = []
    for ep in episodes:
        tags = ep.get("tags", {})
        rows.append({
            "guid": ep["guid"],
            "podcast_id": ep["podcast_id"],
            "podcast_name": podcasts_by_id.get(ep["podcast_id"], {}).get("name", ep["podcast_id"]),
            "title": ep["title"],
            "link": ep.get("link") or "",
            "published_at": pd.to_datetime(ep["published_at"]) if ep.get("published_at") else pd.NaT,
            "summary": ep.get("summary", ""),
            "sectors": tags.get("sectors", []),
            "themes": tags.get("themes", []),
            "stocks": tags.get("stocks", []),
            "sentiment": tags.get("sentiment", 0.0),
            "tag_count": len(tags.get("sectors", [])) + len(tags.get("themes", [])) + len(tags.get("stocks", [])),
            "transcript_status": ep.get("transcript_status", "pending"),
            "transcript_words": ep.get("transcript_word_count", 0),
        })
    return pd.DataFrame(rows)


def aggregate_entity(df, kind, labels):
    exploded = df[[kind, "sentiment"]].explode(kind).dropna(subset=[kind])
    if exploded.empty:
        return pd.DataFrame(columns=["id", "label", "mentions", "avg_sentiment"])
    grouped = exploded.groupby(kind).agg(mentions=("sentiment", "count"), avg_sentiment=("sentiment", "mean")).reset_index()
    grouped = grouped.rename(columns={kind: "id"})
    grouped["label"] = grouped["id"].map(labels[kind]).fillna(grouped["id"])
    return grouped.sort_values("mentions", ascending=False)


def sentiment_bucket(score):
    if score > 0.15:
        return "Bullish"
    if score < -0.15:
        return "Bearish"
    return "Neutral"


def explode_monthly(df, kind, labels, entity_ids=None):
    """One row per (episode, entity-of-`kind`, month). `entity_ids`
    optionally restricts to a specific set of ids (else all mentioned)."""
    sub = df[["published_at", kind, "sentiment"]].explode(kind).dropna(subset=[kind, "published_at"])
    if entity_ids is not None:
        sub = sub[sub[kind].isin(entity_ids)]
    if sub.empty:
        return sub
    sub = sub.copy()
    sub["month_period"] = sub["published_at"].dt.to_period("M")
    sub["month_label"] = sub["month_period"].dt.strftime("%b %Y")
    sub["label"] = sub[kind].map(labels[kind]).fillna(sub[kind])
    return sub


def render_mentions_over_time(sub, height=280):
    """Multi-line chart: mention count per month, one line per entity."""
    if sub.empty:
        st.caption("No mentions in this window.")
        return
    counts = sub.groupby(["month_period", "month_label", "label"]).size().reset_index(name="mentions")
    month_order = [m for m in counts.sort_values("month_period")["month_label"].unique()]
    entity_order = list(sub.groupby("label").size().sort_values(ascending=False).index)
    color_range = [ENTITY_LINE_COLORS[i % len(ENTITY_LINE_COLORS)] for i in range(len(entity_order))]

    chart = (
        alt.Chart(counts)
        .mark_line(point=alt.OverlayMarkDef(size=45, filled=True), strokeWidth=2)
        .encode(
            x=alt.X("month_label:N", title=None, sort=month_order),
            y=alt.Y("mentions:Q", title="Mentions"),
            color=alt.Color("label:N", title=None, scale=alt.Scale(domain=entity_order, range=color_range)),
            tooltip=[alt.Tooltip("label:N", title="Name"), alt.Tooltip("month_label:N", title="Month"),
                     alt.Tooltip("mentions:Q", title="Mentions")],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def render_sentiment_over_time(sub, height=140):
    """Faceted stacked bars: bullish/neutral/bearish mention counts per
    month, one small chart per entity, independent y-scales (the shape
    over time matters more than comparing raw magnitude across entities)."""
    if sub.empty:
        st.caption("No mentions in this window.")
        return
    sub = sub.copy()
    sub["bucket"] = sub["sentiment"].apply(sentiment_bucket)
    counts = sub.groupby(["month_period", "month_label", "label", "bucket"]).size().reset_index(name="count")
    month_order = [m for m in counts.sort_values("month_period")["month_label"].unique()]
    entity_order = list(sub.groupby("label").size().sort_values(ascending=False).index)

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("month_label:N", title=None, sort=month_order),
            y=alt.Y("count:Q", title="Mentions"),
            color=alt.Color("bucket:N", title="Tone",
                             scale=alt.Scale(domain=["Bullish", "Neutral", "Bearish"], range=[GOOD, NEUTRAL, CRITICAL])),
            tooltip=[alt.Tooltip("label:N", title="Name"), alt.Tooltip("month_label:N", title="Month"),
                     alt.Tooltip("bucket:N", title="Tone"), alt.Tooltip("count:Q", title="Mentions")],
        )
        .properties(height=height, width=220)
        .facet(facet=alt.Facet("label:N", title=None, sort=entity_order), columns=2)
        .resolve_scale(y="independent")
    )
    st.altair_chart(chart, use_container_width=True)


def _sentiment_color(score):
    if score > 0.15:
        return GOOD
    if score < -0.15:
        return CRITICAL
    return NEUTRAL


def render_bar_chart(agg_df, title):
    if agg_df.empty:
        st.caption("No mentions in this window.")
        return
    top = agg_df.head(8).copy()
    top["color"] = top["avg_sentiment"].apply(_sentiment_color)
    chart = (
        alt.Chart(top)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=16)
        .encode(
            y=alt.Y("label:N", sort="-x", title=None),
            x=alt.X("mentions:Q", title=None),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[alt.Tooltip("label:N", title="Name"), alt.Tooltip("mentions:Q", title="Mentions"),
                     alt.Tooltip("avg_sentiment:Q", title="Avg sentiment", format=".2f")],
        )
        .properties(height=28 * len(top))
    )
    st.altair_chart(chart, use_container_width=True)


def render_trend_chart(df, podcasts, podcasts_by_id):
    if df.empty:
        st.caption("No episodes in this window.")
        return
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=6)
    recent = df[df["published_at"] >= cutoff].copy()
    if recent.empty:
        st.caption("No episodes in the last 6 months.")
        return
    recent["month_period"] = recent["published_at"].dt.to_period("M")
    recent["month_label"] = recent["month_period"].dt.strftime("%b %Y")
    counts = recent.groupby(["month_period", "month_label", "podcast_name"]).size().reset_index(name="episodes")
    month_order = [m for m in counts.sort_values("month_period")["month_label"].unique()]

    color_domain = [p["name"] for p in podcasts]
    color_range = [podcast_color(p, i) for i, p in enumerate(podcasts)]

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("month_label:N", title=None, sort=month_order),
            y=alt.Y("episodes:Q", title="Episodes"),
            color=alt.Color("podcast_name:N", title="Podcast",
                             scale=alt.Scale(domain=color_domain, range=color_range)),
            tooltip=[alt.Tooltip("podcast_name:N", title="Podcast"),
                     alt.Tooltip("month_label:N", title="Month"),
                     alt.Tooltip("episodes:Q", title="Episodes")],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def main():
    inject_style()
    podcasts, taxonomy, episodes, run_state = load_data()
    podcasts_by_id = {p["id"]: p for p in podcasts}
    labels = label_maps(taxonomy)

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.title("Podcast Monitor")
        st.caption("Financial themes, sectors & stocks — tracked across podcasts, from full transcripts where available.")
    with top_r:
        if st.button("\U0001F504 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        if run_state.get("last_run_at"):
            last = pd.to_datetime(run_state["last_run_at"])
            st.caption(f"Last updated {last.strftime('%Y-%m-%d %H:%M UTC')} · {run_state.get('episode_count', len(episodes))} episodes")

    # ---------------- Filters ----------------
    st.markdown("---")
    f1, f2, f3, f4 = st.columns([2, 2, 1.4, 2])
    with f1:
        podcast_names = [p["name"] for p in podcasts]
        selected_podcasts = st.multiselect("Podcasts", podcast_names, default=podcast_names)
    with f2:
        entity_options = ["All"] + [f"Theme: {v}" for v in sorted(labels["themes"].values())] \
            + [f"Sector: {v}" for v in sorted(labels["sectors"].values())] \
            + [f"Stock: {v}" for v in sorted(labels["stocks"].values())]
        entity_choice = st.selectbox("Filter to a theme / sector / stock", entity_options)
    with f3:
        days = st.selectbox("Date range", ["30", "90", "180", "All"], index=1,
                             format_func=lambda v: "All time" if v == "All" else f"Last {v} days")
    with f4:
        search = st.text_input("Search titles & show notes", "")

    df = episodes_to_df(episodes, podcasts_by_id)
    df = df[df["podcast_name"].isin(selected_podcasts)]

    if days != "All":
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(days))
        df = df[df["published_at"] >= cutoff]

    if entity_choice != "All":
        kind_label, value = entity_choice.split(": ", 1)
        kind = {"Theme": "themes", "Sector": "sectors", "Stock": "stocks"}[kind_label]
        entity_id = next((k for k, v in labels[kind].items() if v == value), None)
        df = df[df[kind].apply(lambda ids: entity_id in ids)]

    if search.strip():
        q = search.strip().lower()
        df = df[df["title"].str.lower().str.contains(q) | df["summary"].str.lower().str.contains(q)]

    tagged = df[df["tag_count"] > 0]

    # ---------------- Overview ----------------
    tab_overview, tab_trends, tab_episodes, tab_manage = st.tabs(
        ["Overview", "Trends", "Episodes", "Manage podcasts"]
    )

    with tab_overview:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Episodes in view", len(df))
        total_mentions = (
            int(sum(len(v) for col in ("sectors", "themes", "stocks") for v in tagged[col]))
            if not tagged.empty else 0
        )
        m2.metric("Tagged mentions", total_mentions, help=f"{len(tagged)} episodes with a signal")
        avg_sentiment = float(tagged["sentiment"].mean()) if not tagged.empty else 0.0
        word, _ = sentiment_word(avg_sentiment)
        m3.metric("Net sentiment", word, f"{avg_sentiment:+.2f}")
        transcribed = int((df["transcript_status"] == "done").sum())
        m4.metric("Transcribed episodes", f"{transcribed}/{len(df)}")

        st.subheader("Mentions by month")
        render_trend_chart(df, [p for p in podcasts if p["name"] in selected_podcasts], podcasts_by_id)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Top themes")
            render_bar_chart(aggregate_entity(df, "themes", labels), "themes")
        with c2:
            st.subheader("Top sectors")
            render_bar_chart(aggregate_entity(df, "sectors", labels), "sectors")
        with c3:
            st.subheader("Top stocks")
            render_bar_chart(aggregate_entity(df, "stocks", labels), "stocks")

    with tab_trends:
        st.caption(
            "Pick a category and up to 8 themes, sectors, or stocks to compare how often they've "
            "come up, and whether commentary about them has leaned bullish, neutral, or bearish, "
            "month by month."
        )
        kind_label = st.radio("Category", ["Themes", "Sectors", "Stocks"], horizontal=True)
        kind = {"Themes": "themes", "Sectors": "sectors", "Stocks": "stocks"}[kind_label]

        ranked = aggregate_entity(df, kind, labels)
        all_labels = list(ranked["label"])
        default_labels = all_labels[:5]
        chosen_labels = st.multiselect(
            f"Which {kind_label.lower()}?", all_labels, default=default_labels, max_selections=8,
        )
        chosen_ids = ranked[ranked["label"].isin(chosen_labels)]["id"].tolist()

        sub = explode_monthly(df, kind, labels, entity_ids=chosen_ids if chosen_ids else None)
        if not chosen_labels:
            st.caption(f"No {kind_label.lower()} selected — pick one or more above.")
        else:
            st.subheader("Mentions over time")
            render_mentions_over_time(sub)

            st.subheader("Tone over time")
            st.caption("Each episode's overall tone counted toward every theme/sector/stock it mentions.")
            render_sentiment_over_time(sub)

    with tab_episodes:
        st.caption(f"Showing {min(len(df), 200)} of {len(df)} matching episodes")
        shown = df.sort_values("published_at", ascending=False).head(200)
        for _, ep in shown.iterrows():
            word, cls = sentiment_word(ep["sentiment"])
            tag_labels = (
                [labels["themes"].get(t, t) for t in ep["themes"]]
                + [labels["sectors"].get(t, t) for t in ep["sectors"]]
                + [labels["stocks"].get(t, t) for t in ep["stocks"]]
            )
            pills = "".join(f'<span class="tag-pill">{t}</span>' for t in tag_labels[:6])
            date_str = ep["published_at"].strftime("%b %d, %Y") if pd.notna(ep["published_at"]) else "—"
            transcript_note = (
                f'<span class="transcript-badge">transcript · {ep["transcript_words"]:,} words</span>'
                if ep["transcript_status"] == "done" else ""
            )
            title_html = f'<a href="{ep["link"]}" target="_blank">{ep["title"]}</a>' if ep["link"] else ep["title"]
            st.markdown(
                f"""<div class="ep-row">
                <span style="color:#898781; font-size:12.5px;">{date_str}</span> ·
                <strong>{ep['podcast_name']}</strong>{transcript_note}
                <span class="sent-badge {cls}" style="float:right;">{word}</span>
                <div style="font-weight:600; margin:4px 0;">{title_html}</div>
                <div>{pills}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with tab_manage:
        st.subheader("Tracked podcasts")
        for i, p in enumerate(podcasts):
            cols = st.columns([0.5, 3, 2])
            cols[0].markdown(
                f'<div style="width:14px;height:14px;border-radius:50%;background:{podcast_color(p, i)};margin-top:6px;"></div>',
                unsafe_allow_html=True,
            )
            cols[1].markdown(f"**{p['name']}**  \n{p['publisher']}")
            cols[2].markdown("Active" if p.get("active", True) else "Inactive")
        st.info(
            "This view is read-only. To add, remove, or disable a podcast, edit "
            "`config/podcasts.json` in the repository and commit — the next pipeline "
            "run (scheduled GitHub Action or manual) picks it up. Finding a feed URL: "
            "look up the show's Apple Podcasts ID and query "
            "`https://itunes.apple.com/lookup?id=<id>&entity=podcast` for `feedUrl`."
        )
        st.subheader("How the data updates")
        st.markdown(
            "- **Fetch + tag** (`pipeline/run.py`): zero-token, stdlib-only RSS parsing + "
            "regex tagging against `config/taxonomy.json`. Runs on a schedule via GitHub Actions.\n"
            "- **Transcribe** (`pipeline/transcribe.py`): local Whisper transcription of episode "
            "audio (via `faster-whisper`), then re-tags from the *full transcript* for much better "
            "recall than title/show-notes alone. Full transcript text stays local/gitignored "
            "(copyrighted commercial audio, public repo) — only derived tags and word counts are "
            "committed.\n"
            "- **Optional LLM enrichment** (`pipeline/enrich_claude.py`): off by default, for when "
            "keyword tagging isn't nuanced enough."
        )

    st.markdown("---")
    st.caption(
        "Data from public podcast RSS feeds and locally-transcribed audio. Signals are a directional "
        "attention/tone indicator, not investment advice — always check the source episode."
    )


if __name__ == "__main__":
    main()
