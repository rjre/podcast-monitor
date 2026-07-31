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
    with open(os.path.join(CONFIG_DIR, "categories.json")) as f:
        categories = json.load(f)["categories"]
    with open(os.path.join(CONFIG_DIR, "taxonomy.json")) as f:
        taxonomy = json.load(f)
    with open(os.path.join(DATA_DIR, "episodes.json")) as f:
        episodes = json.load(f)
    state = {}
    state_path = os.path.join(DATA_DIR, "state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    aggregates = {}
    aggregates_path = os.path.join(DATA_DIR, "aggregates.json")
    if os.path.exists(aggregates_path):
        with open(aggregates_path) as f:
            aggregates = json.load(f)
    return podcasts, categories, taxonomy, episodes, state, aggregates


def label_maps(taxonomy):
    return {kind: {e["id"]: e["label"] for e in taxonomy[kind]} for kind in ("sectors", "themes", "stocks")}


def category_maps(categories):
    by_id = {c["id"]: c for c in categories}
    return by_id


# Podcasts are tagged with a "region" in config/podcasts.json (based on
# publisher/host location, not episode content) so the dashboard can
# highlight -- and let users filter for -- how geographically skewed the
# panel is. As of the last count, ~90% of tracked shows are US-based, which
# should color how any "consensus" or "crowd" read here is interpreted:
# it's largely a US-media consensus, not a global one.
REGION_LABELS = {
    "us": "United States", "uk": "United Kingdom", "canada": "Canada",
    "australia": "Australia", "europe": "Europe (other)",
}


def region_label(region_id):
    return REGION_LABELS.get(region_id, region_id.title() if region_id else "Unknown")


def podcast_color(podcast, categories_by_id, dark=False):
    cat = categories_by_id.get(podcast.get("category"))
    if not cat:
        return PODCAST_FALLBACK_COLORS[0]
    return cat["color_dark"] if dark else cat["color"]


def sentiment_word(score):
    if score > 0.15:
        return "Bullish", "sent-bullish"
    if score < -0.15:
        return "Bearish", "sent-bearish"
    return "Neutral", "sent-neutral"


def episodes_to_df(episodes, podcasts_by_id, categories_by_id):
    rows = []
    for ep in episodes:
        tags = ep.get("tags", {})
        podcast = podcasts_by_id.get(ep["podcast_id"], {})
        category = categories_by_id.get(podcast.get("category"), {})
        rows.append({
            "guid": ep["guid"],
            "podcast_id": ep["podcast_id"],
            "podcast_name": podcast.get("name", ep["podcast_id"]),
            "category_id": podcast.get("category", ""),
            "category_label": category.get("label", "Uncategorized"),
            "region_id": podcast.get("region", ""),
            "region_label": region_label(podcast.get("region", "")),
            "title": ep["title"],
            "link": ep.get("link") or "",
            "published_at": pd.to_datetime(ep["published_at"]) if ep.get("published_at") else pd.NaT,
            "summary": ep.get("summary", ""),
            "episode_summary": ep.get("episode_summary", ""),
            "episode_summary_source": ep.get("episode_summary_source", ""),
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


def render_show_theme_timeline(timeline, height=280):
    """Multi-line chart: a single show's own top themes, mention count per
    month -- 'how has this show's topic mix shifted', built from
    aggregates["podcast_theme_timeline"][podcast_id] (pipeline/insights.py's
    build_podcast_theme_timeline). Unlike render_mentions_over_time this
    doesn't explode data/episodes.json at render time -- the per-podcast
    monthly counts are already precomputed, so months a theme wasn't
    mentioned are filled in as 0 here for a continuous line instead of a gap.
    """
    months = timeline.get("months") or []
    theme_ids = timeline.get("top_theme_ids") or []
    theme_labels = timeline.get("top_theme_labels") or []
    if not months or not theme_ids:
        st.caption("Not enough tagged episodes yet for a theme timeline.")
        return

    rows = []
    for entry in months:
        month = entry["month"]
        month_label = pd.Period(month, freq="M").strftime("%b %Y")
        counts = entry.get("theme_counts", {})
        for theme_id, theme_label in zip(theme_ids, theme_labels):
            rows.append({
                "month": month, "month_label": month_label,
                "theme": theme_label, "mentions": counts.get(theme_id, 0),
            })
    df = pd.DataFrame(rows)
    month_order = [m for m in df.sort_values("month")["month_label"].unique()]
    color_range = [ENTITY_LINE_COLORS[i % len(ENTITY_LINE_COLORS)] for i in range(len(theme_labels))]

    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=40, filled=True), strokeWidth=2)
        .encode(
            x=alt.X("month_label:N", title=None, sort=month_order),
            y=alt.Y("mentions:Q", title="Episodes mentioning theme"),
            color=alt.Color("theme:N", title=None, scale=alt.Scale(domain=theme_labels, range=color_range)),
            tooltip=[alt.Tooltip("theme:N", title="Theme"), alt.Tooltip("month_label:N", title="Month"),
                     alt.Tooltip("mentions:Q", title="Episodes")],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def render_region_breakdown(df, height=90):
    """Horizontal bar of episode share by podcast region -- the point isn't
    the chart, it's the callout above it: readers should know how US-heavy
    the panel is before treating any 'crowd consensus' signal as global."""
    if df.empty:
        st.caption("No episodes in view.")
        return
    counts = df["region_label"].value_counts().reset_index()
    counts.columns = ["region", "episodes"]
    counts["share"] = counts["episodes"] / counts["episodes"].sum()
    counts = counts.sort_values("episodes", ascending=False)

    chart = (
        alt.Chart(counts)
        .mark_bar(color="#2a78d6")
        .encode(
            x=alt.X("episodes:Q", title="Episodes in view"),
            y=alt.Y("region:N", title=None, sort="-x"),
            tooltip=[alt.Tooltip("region:N", title="Region"), alt.Tooltip("episodes:Q", title="Episodes"),
                     alt.Tooltip("share:Q", title="Share", format=".0%")],
        )
        .properties(height=height)
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


def render_trend_chart(df, categories):
    """Episodes per month, colored by podcast *category* rather than
    individual show -- with 100+ podcasts tracked, per-show coloring stops
    being readable, but 5-6 categories works fine as a stacked bar."""
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
    counts = recent.groupby(["month_period", "month_label", "category_label"]).size().reset_index(name="episodes")
    month_order = [m for m in counts.sort_values("month_period")["month_label"].unique()]

    color_domain = [c["label"] for c in categories]
    color_range = [c["color"] for c in categories]

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("month_label:N", title=None, sort=month_order),
            y=alt.Y("episodes:Q", title="Episodes"),
            color=alt.Color("category_label:N", title="Category",
                             scale=alt.Scale(domain=color_domain, range=color_range)),
            tooltip=[alt.Tooltip("category_label:N", title="Category"),
                     alt.Tooltip("month_label:N", title="Month"),
                     alt.Tooltip("episodes:Q", title="Episodes")],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def all_aggregate_entities(aggregates):
    """Flatten aggregates.json's sectors/themes/stocks into one list, each
    tagged with a human-readable kind, for the "Things to Notice" panel."""
    kind_labels = {"sectors": "Sector", "themes": "Theme", "stocks": "Stock"}
    out = []
    for kind, kind_label in kind_labels.items():
        for entity in aggregates.get(kind, []):
            out.append({**entity, "kind": kind_label})
    return out


def rising_entities(entities, min_mentions=5, min_prior=3, top_n=5):
    # min_prior guards against a tiny denominator (1 mention -> 6) reading
    # as a dramatic +500% swing that isn't really a stable trend.
    cands = [e for e in entities if e.get("trend") == "rising" and e["mentions"] >= min_mentions
             and e["prior_30d_mentions"] >= min_prior]
    return sorted(cands, key=lambda e: e["momentum_pct"], reverse=True)[:top_n]


def newly_emerged_entities(entities, min_recent=2, top_n=5):
    cands = [e for e in entities if e.get("trend") == "new" and e["recent_30d_mentions"] >= min_recent]
    return sorted(cands, key=lambda e: e["recent_30d_mentions"], reverse=True)[:top_n]


def cooling_entities(entities, min_mentions=5, min_prior=3, top_n=3):
    cands = [e for e in entities if e.get("trend") == "falling" and e["mentions"] >= min_mentions
             and e["prior_30d_mentions"] >= min_prior]
    return sorted(cands, key=lambda e: e["momentum_pct"])[:top_n]


def divergent_entities(entities, min_mentions=5, top_n=5):
    cands = [e for e in entities if e["mentions"] >= min_mentions]
    return sorted(cands, key=lambda e: abs(e["sentiment_divergence"]), reverse=True)[:top_n]


def turning_point_entities(entities, min_mentions=5, min_prior=3, top_n=5):
    # Volume momentum (rising/falling) and tone momentum (turning_bullish/
    # turning_bearish) are independent axes -- an entity can be discussed
    # the same amount but with the mood flipping underneath.
    cands = [e for e in entities if e.get("sentiment_trend") in ("turning_bullish", "turning_bearish")
             and e["mentions"] >= min_mentions and e["prior_30d_mentions"] >= min_prior]
    return sorted(cands, key=lambda e: abs(e["sentiment_shift"]), reverse=True)[:top_n]


def most_volatile_entities(entities, min_mentions=8, top_n=5):
    cands = [e for e in entities if e["mentions"] >= min_mentions and e.get("sentiment_volatility") is not None]
    return sorted(cands, key=lambda e: e["sentiment_volatility"], reverse=True)[:top_n]


def dormant_entities(entities, top_n=5):
    cands = [e for e in entities if e.get("lifecycle_stage") == "dormant"]
    return sorted(cands, key=lambda e: e["last_seen"] or "", reverse=True)[:top_n]


def most_contested_entities(entities, min_mentions=5, min_contested=2, top_n=5):
    # min_contested avoids a single clashing episode out of a dozen ranking
    # above a topic genuinely fought over in several -- ratio alone is too
    # noisy on a count of 1.
    cands = [e for e in entities if e["mentions"] >= min_mentions and e.get("contested_episodes", 0) >= min_contested]
    return sorted(cands, key=lambda e: e["contested_episodes"] / e["mentions"], reverse=True)[:top_n]


def crowd_stance_entities(entities, top_n=5):
    # Only entities with an actual buy/sell recommendation tally, and only
    # stocks -- "buy the sector" isn't language people actually use.
    cands = [e for e in entities if e["kind"] == "Stock" and (e.get("buy_mentions") or e.get("sell_mentions"))
             and e["buy_mentions"] != e["sell_mentions"]]
    return sorted(cands, key=lambda e: abs(e["buy_mentions"] - e["sell_mentions"]), reverse=True)[:top_n]


def render_notice_row(kind, label, detail, badge_text, badge_cls):
    st.markdown(
        f"""<div class="ep-row">
        <span class="tag-pill">{kind}</span> <strong>{label}</strong>
        <span class="sent-badge {badge_cls}" style="float:right;">{badge_text}</span>
        <div style="color:#898781; font-size:12.5px; margin-top:4px;">{detail}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_things_to_notice(aggregates):
    st.subheader("\U0001F50E Things to notice")
    if not aggregates:
        st.caption("Run the pipeline at least once to populate data/aggregates.json.")
        return
    st.caption(
        "Automatically surfaced from mention volume and tone across all tracked podcasts — "
        "independent of the filters above, so nothing gets missed by only looking at what "
        "you already searched for."
    )
    entities = all_aggregate_entities(aggregates)

    col_rising, col_new, col_tone = st.columns(3)

    with col_rising:
        st.markdown("**\U0001F4C8 Rising fast**")
        rising = rising_entities(entities)
        if not rising:
            st.caption("Nothing trending up right now.")
        for e in rising:
            render_notice_row(
                e["kind"], e["label"],
                f"{e['recent_30d_mentions']} mentions last 30d vs {e['prior_30d_mentions']} before",
                f"↑ +{e['momentum_pct']:.0f}%", "sent-bullish",
            )

    with col_new:
        st.markdown("**\U0001F195 Newly emerged**")
        emerged = newly_emerged_entities(entities)
        if not emerged:
            st.caption("No brand-new topics in the last 30 days.")
        for e in emerged:
            render_notice_row(
                e["kind"], e["label"],
                f"{e['recent_30d_mentions']} mentions in the last 30 days, none before that",
                "NEW", "sent-bullish",
            )

    with col_tone:
        st.markdown("**\U0001F3AD Unusually toned**")
        divergent = divergent_entities(entities)
        if not divergent:
            st.caption("Nothing diverging from the overall tone right now.")
        for e in divergent:
            word, cls = sentiment_word(e["avg_sentiment"])
            sign = "+" if e["sentiment_divergence"] >= 0 else ""
            render_notice_row(
                e["kind"], e["label"],
                f"{sign}{e['sentiment_divergence']:.2f} vs. the {aggregates.get('global_avg_sentiment', 0.0):+.2f} baseline",
                word, cls,
            )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    col_cooling, col_contested, col_stance = st.columns(3)

    with col_cooling:
        st.markdown("**\U0001F4C9 Cooling off**")
        cooling = cooling_entities(entities)
        if not cooling:
            st.caption("Nothing trending down right now.")
        for e in cooling:
            render_notice_row(
                e["kind"], e["label"],
                f"{e['recent_30d_mentions']} mentions last 30d vs {e['prior_30d_mentions']} before",
                f"↓ {e['momentum_pct']:.0f}%", "sent-bearish",
            )

    with col_contested:
        st.markdown("**\U0001F94A Most contested**")
        st.caption("Meaningful bullish *and* bearish language in the same episodes -- not just quiet, actively debated.")
        contested = most_contested_entities(entities)
        if not contested:
            st.caption("Nothing clearly contested right now.")
        for e in contested:
            render_notice_row(
                e["kind"], e["label"],
                f"clashing views in {e['contested_episodes']} of {e['mentions']} episodes",
                f"{e['contested_episodes']}/{e['mentions']}", "sent-neutral",
            )

    with col_stance:
        st.markdown("**\U0001F4E3 Crowd stance**")
        st.caption("Actual buy/sell recommendation language, not just tone.")
        stance = crowd_stance_entities(entities)
        if not stance:
            st.caption("No clear buy/sell calls picked up yet.")
        for e in stance:
            leaning = "buy" if e["buy_mentions"] > e["sell_mentions"] else "sell"
            cls = "sent-bullish" if leaning == "buy" else "sent-bearish"
            render_notice_row(
                e["kind"], e["label"],
                f"{e['buy_mentions']} buy call(s) vs {e['sell_mentions']} sell call(s)",
                leaning.upper(), cls,
            )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    col_turning, col_volatile = st.columns(2)

    with col_turning:
        st.markdown("**\U0001F504 Turning point**")
        st.caption("Same volume of coverage, but the mood underneath it has flipped in the last 30 days.")
        turning = turning_point_entities(entities)
        if not turning:
            st.caption("No clear tone reversal right now.")
        for e in turning:
            bullish = e["sentiment_trend"] == "turning_bullish"
            cls = "sent-bullish" if bullish else "sent-bearish"
            arrow = "↗" if bullish else "↘"
            render_notice_row(
                e["kind"], e["label"],
                f"tone was {e['prior_avg_sentiment']:+.2f} 30-60 days ago, now {e['recent_avg_sentiment']:+.2f}",
                f"{arrow} {'bullish' if bullish else 'bearish'}", cls,
            )

    with col_volatile:
        st.markdown("**\U0001F3A2 Most volatile tone**")
        st.caption("Opinion swings wildly month to month, rather than settling on a consistent read.")
        volatile = most_volatile_entities(entities)
        if not volatile:
            st.caption("Nothing with enough monthly history to judge volatility yet.")
        for e in volatile:
            render_notice_row(
                e["kind"], e["label"],
                f"tone swings ±{e['sentiment_volatility']:.2f} across months (avg {e['avg_sentiment']:+.2f})",
                f"σ {e['sentiment_volatility']:.2f}", "sent-neutral",
            )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    col_surprise, col_contrarian, col_dormant = st.columns(3)

    with col_surprise:
        st.markdown("**\U0001F62E Surprising for this show**")
        st.caption("Tone that deviates sharply from THAT podcast's own usual read, not the dataset average.")
        surprising = aggregates.get("surprising_episodes", [])[:5]
        if not surprising:
            st.caption("Nothing standing out from any show's normal tone right now.")
        for e in surprising:
            bullish = e["surprise"] > 0
            cls = "sent-bullish" if bullish else "sent-bearish"
            render_notice_row(
                "Episode", f"{e['podcast_name']}: {e['title'][:50]}",
                f"usually {e['podcast_baseline_sentiment']:+.2f} for this show, this one is {e['sentiment']:+.2f}",
                f"{e['surprise']:+.2f}", cls,
            )

    with col_contrarian:
        st.markdown("**⚡ Contrarian calls**")
        st.caption("The lone dissenting episode against a strong same-month consensus on a topic.")
        contrarian = aggregates.get("contrarian_calls", [])[:5]
        if not contrarian:
            st.caption("No clear dissenting voices against a strong consensus right now.")
        for c in contrarian:
            dissent_bullish = c["consensus"] == "bearish"  # dissents the other way
            cls = "sent-bullish" if dissent_bullish else "sent-bearish"
            render_notice_row(
                "Theme/Sector/Stock", c["entity_label"],
                f"{c['podcast_name']}: {c['title'][:45]} -- vs {c['consensus_fraction']:.0%} "
                f"{c['consensus']} consensus ({c['group_size']} eps, {c['month']})",
                f"{c['sentiment']:+.2f}", cls,
            )

    with col_dormant:
        st.markdown("**\U0001F4A4 Gone quiet**")
        st.caption("Was discussed, hasn't come up anywhere in over 60 days -- distinct from just 'less' coverage.")
        dormant = dormant_entities(entities)
        if not dormant:
            st.caption("Nothing that was active has gone fully silent yet.")
        for e in dormant:
            last_seen = pd.to_datetime(e["last_seen"]) if e.get("last_seen") else None
            days_ago = (pd.Timestamp.now(tz="UTC") - last_seen).days if last_seen is not None else None
            render_notice_row(
                e["kind"], e["label"],
                f"{e['mentions']} mentions total, last one {days_ago} days ago" if days_ago is not None
                else f"{e['mentions']} mentions total",
                "quiet", "sent-neutral",
            )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    col_cooc, col_guests = st.columns(2)

    with col_cooc:
        st.markdown("**\U0001F517 Correlated topics**")
        st.caption("Pairs discussed together far more often than chance would predict -- links nobody hand-coded.")
        pairs = aggregates.get("entity_cooccurrence", [])[:5]
        if not pairs:
            st.caption("Not enough co-occurring pairs yet.")
        for p in pairs:
            render_notice_row(
                f"{p['a_kind'].title()} + {p['b_kind'].title()}", f"{p['a_label']} + {p['b_label']}",
                f"appeared together in {p['co_occurrences']} episodes",
                f"×{p['lift']:.0f}", "sent-neutral",
            )

    with col_guests:
        st.markdown("**\U0001F3A4 Notable voices**")
        st.caption(
            "People named in episode titles, recurring across multiple episodes -- often a guest, "
            "sometimes a host whose name is baked into the show's own title format."
        )
        guests = aggregates.get("guests", [])[:5]
        if not guests:
            st.caption("No recurring named voices picked up yet.")
        for g in guests:
            word, cls = sentiment_word(g["avg_sentiment"])
            shows = ", ".join(g["podcasts"][:2]) + ("…" if len(g["podcasts"]) > 2 else "")
            render_notice_row(
                "Person", g["name"],
                f"{g['episode_count']} episodes on {shows}",
                word, cls,
            )


def main():
    inject_style()
    podcasts, categories, taxonomy, episodes, run_state, aggregates = load_data()
    podcasts_by_id = {p["id"]: p for p in podcasts}
    categories_by_id = category_maps(categories)
    labels = label_maps(taxonomy)

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.title("Podcast Monitor")
        st.caption(f"Financial themes, sectors & stocks — tracked across {len(podcasts)} podcasts, from full transcripts where available.")
    with top_r:
        if st.button("\U0001F504 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        if run_state.get("last_run_at"):
            last = pd.to_datetime(run_state["last_run_at"])
            st.caption(f"Last updated {last.strftime('%Y-%m-%d %H:%M UTC')} · {run_state.get('episode_count', len(episodes))} episodes")

    # ---------------- Filters ----------------
    st.markdown("---")
    fc1, fc_region, fc2 = st.columns([1.8, 1.6, 2.6])
    with fc1:
        category_labels = [c["label"] for c in categories]
        selected_categories = st.multiselect("Categories", category_labels, default=category_labels)
    with fc_region:
        region_options = sorted({region_label(p.get("region", "")) for p in podcasts})
        selected_regions = st.multiselect(
            "Regions", region_options, default=region_options,
            help="Podcast is tagged by publisher/host location, not episode content. Most tracked "
                 "shows are US-based -- use this to check whether a reading changes once non-US "
                 "shows are isolated or excluded.",
        )
    with fc2:
        podcasts_in_categories = [p for p in podcasts
                                   if categories_by_id.get(p.get("category"), {}).get("label") in selected_categories
                                   and region_label(p.get("region", "")) in selected_regions]
        podcast_names = [p["name"] for p in podcasts_in_categories]
        selected_podcasts = st.multiselect("Podcasts (narrow further)", podcast_names, default=podcast_names)

    f2, f3, f4 = st.columns([2, 1.4, 2])
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

    df = episodes_to_df(episodes, podcasts_by_id, categories_by_id)
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
    tab_overview, tab_trends, tab_shows, tab_episodes, tab_manage = st.tabs(
        ["Overview", "Trends", "Show Explorer", "Episodes", "Manage podcasts"]
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

        if not df.empty:
            us_share = float((df["region_id"] == "us").mean())
            st.info(
                f"\U0001F310 **{us_share:.0%} of episodes in view are from US-based podcasts.** "
                "Treat any 'crowd consensus' or 'contrarian call' signal on this dashboard as a "
                "reading of US financial media specifically, not a global one -- non-US shows are "
                "a small enough slice that a genuinely different international view could be "
                "diluted out. Use the Regions filter above to isolate or exclude them."
            )
            with st.expander("Episodes by region"):
                render_region_breakdown(df)

        st.markdown("---")
        render_things_to_notice(aggregates)

        st.markdown("---")
        st.subheader("Mentions by month")
        active_categories = [c for c in categories if c["label"] in selected_categories]
        render_trend_chart(df, active_categories)

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

    with tab_shows:
        podcast_summaries = aggregates.get("podcast_summaries", {})
        theme_timelines = aggregates.get("podcast_theme_timeline", {})
        eligible_pids = [p["id"] for p in podcasts_in_categories if p["id"] in podcast_summaries]
        if not eligible_pids:
            st.caption("No show has enough tagged episodes yet for a show-level summary.")
        else:
            names_by_pid = {p["id"]: p["name"] for p in podcasts_in_categories}
            eligible_pids.sort(key=lambda pid: names_by_pid.get(pid, pid))
            chosen_name = st.selectbox("Show", [names_by_pid[pid] for pid in eligible_pids])
            chosen_pid = next(pid for pid in eligible_pids if names_by_pid[pid] == chosen_name)
            show = podcast_summaries[chosen_pid]

            st.subheader(chosen_name)
            st.markdown(show["summary_text"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Episodes tagged", show["episode_count"])
            word, _ = sentiment_word(show["avg_sentiment"])
            m2.metric("All-time tone", word, f"{show['avg_sentiment']:+.2f}")
            baseline = aggregates.get("podcast_baselines", {}).get(chosen_pid, {})
            m3.metric("Avg conviction", f"{baseline.get('avg_conviction'):+.2f}" if baseline.get("avg_conviction") is not None else "—")

            c1, c2 = st.columns(2)
            with c1:
                st.caption("Top themes")
                for t in show["top_themes"]:
                    st.markdown(f"- {t['label']} ({t['count']})")
            with c2:
                st.caption("Top sectors & stocks")
                for t in show["top_sectors"] + show["top_stocks"]:
                    st.markdown(f"- {t['label']} ({t['count']})")

            st.markdown("---")
            st.subheader("Themes over time")
            st.caption("How often this show's own top themes have come up, month by month.")
            timeline = theme_timelines.get(chosen_pid)
            if timeline:
                render_show_theme_timeline(timeline)
            else:
                st.caption("Not enough tagged episodes yet for a theme timeline.")

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
            summary_html = ""
            if ep["episode_summary"]:
                is_genuine = ep["episode_summary_source"] in ("claude-manual", "llm")
                badge = "SUMMARY" if is_genuine else "AUTO-SUMMARY (from tags, not a transcript read)"
                summary_html = (
                    f'<div style="font-size:12.5px; color:#65635f; margin:5px 0 2px;">'
                    f'<span style="font-size:9.5px; letter-spacing:.03em; color:#9b9992;">{badge}</span><br/>'
                    f'{ep["episode_summary"]}</div>'
                )
            st.markdown(
                f"""<div class="ep-row">
                <span style="color:#898781; font-size:12.5px;">{date_str}</span> ·
                <strong>{ep['podcast_name']}</strong>{transcript_note}
                <span class="sent-badge {cls}" style="float:right;">{word}</span>
                <div style="font-weight:600; margin:4px 0;">{title_html}</div>
                <div>{pills}</div>
                {summary_html}
                </div>""",
                unsafe_allow_html=True,
            )

    with tab_manage:
        st.subheader(f"Tracked podcasts ({len(podcasts)})")
        manage_search = st.text_input("Search podcasts", "", key="manage_search")
        by_category = {}
        for p in podcasts:
            if manage_search.strip() and manage_search.strip().lower() not in p["name"].lower():
                continue
            by_category.setdefault(p.get("category", ""), []).append(p)

        for cat in categories:
            shows = by_category.get(cat["id"], [])
            if not shows:
                continue
            with st.expander(f"{cat['label']} ({len(shows)})", expanded=len(categories) <= 2):
                for p in shows:
                    cols = st.columns([0.4, 3, 1.5, 1.5])
                    cols[0].markdown(
                        f'<div style="width:12px;height:12px;border-radius:50%;background:{cat["color"]};margin-top:6px;"></div>',
                        unsafe_allow_html=True,
                    )
                    cols[1].markdown(f"**{p['name']}**  \n{p.get('publisher', '')}")
                    cols[2].markdown(region_label(p.get("region", "")))
                    cols[3].markdown("Active" if p.get("active", True) else "Inactive")
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
