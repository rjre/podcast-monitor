"""Fetch podcast RSS feeds and return episode metadata.

Deliberately dependency-free (stdlib only: urllib + xml.etree) so the
pipeline has nothing extra to install in CI or break over time. Handles
plain RSS 2.0 feeds with iTunes/Podcast-namespace extensions, which is
what every feed in config/podcasts.json uses.
"""
import re
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

USER_AGENT = "podcast-monitor/1.0 (+https://github.com/rjre/podcast-monitor)"

NAMESPACES = {
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "content": "http://purl.org/rss/1.0/modules/content/",
}


def _strip_html(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_date(raw):
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def fetch_raw_feed(feed_url, timeout=30):
    req = urllib.request.Request(feed_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_feed(xml_bytes, podcast_id, podcast_name, since=None):
    """Return a list of episode dicts from raw RSS bytes.

    since: optional timezone-aware datetime; episodes published before it
    are skipped (keeps the payload to the trailing window we care about).
    """
    root = ElementTree.fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        return []

    episodes = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        guid = (item.findtext("guid") or item.findtext("link") or title).strip()
        link = (item.findtext("link") or "").strip()
        pub_date = _parse_date(item.findtext("pubDate"))

        description = item.findtext("description") or ""
        itunes_summary = item.findtext("itunes:summary", namespaces=NAMESPACES) or ""
        summary = _strip_html(itunes_summary or description)

        duration = item.findtext("itunes:duration", namespaces=NAMESPACES) or ""

        if since is not None and pub_date is not None and pub_date < since:
            continue

        episodes.append({
            "guid": guid,
            "podcast_id": podcast_id,
            "podcast_name": podcast_name,
            "title": title,
            "link": link,
            "published_at": pub_date.isoformat() if pub_date else None,
            "summary": summary,
            "duration": duration,
        })

    return episodes


def fetch_episodes(podcast, since=None):
    """Fetch + parse one podcast's feed. `podcast` is an entry from podcasts.json."""
    xml_bytes = fetch_raw_feed(podcast["feed_url"])
    return parse_feed(xml_bytes, podcast["id"], podcast["name"], since=since)


if __name__ == "__main__":
    import json
    import sys

    with open(sys.argv[1] if len(sys.argv) > 1 else "config/podcasts.json") as f:
        cfg = json.load(f)

    since = datetime.now(timezone.utc)
    since = since.replace(month=max(1, since.month - 3)) if since.month > 3 else since.replace(year=since.year - 1, month=since.month + 9)

    for pod in cfg["podcasts"]:
        if not pod.get("active", True):
            continue
        eps = fetch_episodes(pod, since=since)
        print(f"{pod['name']}: {len(eps)} episodes since {since.date()}")
