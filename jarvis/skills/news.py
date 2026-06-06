"""News skill — software-engineering headlines + hobby RSS feeds.

All sources are key-free: HN / Dev.to / Reddit are public JSON APIs; hobby
feeds are arbitrary RSS parsed with feedparser.
"""
from __future__ import annotations

import asyncio

import feedparser
import httpx

HN_TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM = "https://hacker-news.firebaseio.com/v0/item/{id}.json"
HN_LINK = "https://news.ycombinator.com/item?id={id}"
DEVTO = "https://dev.to/api/articles"
REDDIT = "https://www.reddit.com/r/{sub}/top.json"
UA = "JarvisAssistant/0.1 (personal use)"


async def _hackernews(client: httpx.AsyncClient, limit: int) -> list[dict]:
    r = await client.get(HN_TOP)
    r.raise_for_status()
    ids = r.json()[: limit * 2]  # over-fetch; some items are jobs/dead

    async def item(i: int):
        rr = await client.get(HN_ITEM.format(id=i))
        return rr.json() if rr.status_code == 200 else None

    out: list[dict] = []
    for it in await asyncio.gather(*[item(i) for i in ids]):
        if not it or it.get("type") != "story" or not it.get("title"):
            continue
        out.append({
            "title": it["title"],
            "url": it.get("url") or HN_LINK.format(id=it["id"]),
            "source": "Hacker News",
            "score": it.get("score", 0),
        })
        if len(out) >= limit:
            break
    return out


async def _devto(client: httpx.AsyncClient, limit: int) -> list[dict]:
    r = await client.get(DEVTO, params={"per_page": limit, "top": 1})
    r.raise_for_status()
    return [{
        "title": a["title"],
        "url": a["url"],
        "source": "DEV",
        "score": a.get("positive_reactions_count", 0),
    } for a in r.json()[:limit]]


async def _reddit(client: httpx.AsyncClient, sub: str, limit: int) -> list[dict]:
    r = await client.get(REDDIT.format(sub=sub), params={"t": "day", "limit": limit})
    r.raise_for_status()
    out: list[dict] = []
    for c in r.json().get("data", {}).get("children", []):
        d = c.get("data", {})
        out.append({
            "title": d.get("title", ""),
            "url": d.get("url") or ("https://reddit.com" + d.get("permalink", "")),
            "source": f"r/{sub}",
            "score": d.get("score", 0),
        })
    return out[:limit]


async def get_software_news(cfg: dict) -> list[dict]:
    sources = cfg.get("sources", ["hackernews"])
    limit = cfg.get("limit", 6)
    tasks = []
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": UA}) as client:
        if "hackernews" in sources:
            tasks.append(_hackernews(client, limit))
        if "devto" in sources:
            tasks.append(_devto(client, limit))
        if "reddit" in sources:
            for sub in cfg.get("reddit_subs", ["programming"]):
                tasks.append(_reddit(client, sub, limit))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    out: list[dict] = []
    for res in results:
        if isinstance(res, list):
            out.extend(res)
    # de-dup by title, most-upvoted first
    seen: set[str] = set()
    dedup: list[dict] = []
    for a in sorted(out, key=lambda x: x.get("score", 0), reverse=True):
        key = a["title"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            dedup.append(a)
    return dedup[: max(limit, 6)]


def _parse_feed(label: str, url: str, limit: int) -> list[dict]:
    feed = feedparser.parse(url)
    return [{
        "title": getattr(e, "title", "(untitled)"),
        "url": getattr(e, "link", ""),
        "source": label,
    } for e in feed.entries[:limit]]


async def get_hobby_news(hobbies: list[dict], per_feed: int = 3) -> list[dict]:
    out: list[dict] = []
    for h in hobbies:
        url = h.get("rss")
        if not url:
            continue
        # feedparser is blocking; run it off the event loop
        out.extend(await asyncio.to_thread(_parse_feed, h.get("label", "Hobby"), url, per_feed))
    return out
