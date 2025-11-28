from __future__ import annotations
import json
from typing import List, Dict, Any
from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from datetime import date, timedelta
import urllib.parse

def _build_youtube_links_for_meme(
    meme: Dict[str, Any],
    max_links: int = 3,
) -> List[str]:
    """
    Build stable YouTube search URLs for this meme.

    Strategy:
    - Use the meme title as the primary query.
    - Optionally mix in tags to create 2–3 slightly different search queries.
    - Return only YouTube search URLs (not specific videos) so they never 404.
    """
    base = "https://www.youtube.com/results?search_query="

    title = (meme.get("title") or "").strip()
    tags = meme.get("tags") or []

    queries: List[str] = []

    if title:
        # main search
        queries.append(f"{title} meme")

    # add a few variants using tags
    for tag in tags:
        tag = str(tag).strip()
        if not tag:
            continue
        if title:
            queries.append(f"{title} {tag} meme")
        else:
            queries.append(f"{tag} meme")

    # ensure we have at least one query, even if title/tags are missing
    if not queries:
        queries.append("viral meme")

    # de-duplicate while preserving order
    seen = set()
    links: List[str] = []
    for q in queries:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        encoded = urllib.parse.quote_plus(q)
        links.append(base + encoded)
        if len(links) >= max_links:
            break

    return links


def _normalize_memes_with_youtube_links(
    memes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Replace evidence_links of each meme with YouTube search URLs
    based on meme title/tags.
    """
    normalized: List[Dict[str, Any]] = []
    for meme in memes:
        # shallow copy so we don't mutate the original
        m = dict(meme)
        m["evidence_links"] = _build_youtube_links_for_meme(m)
        normalized.append(m)
    return normalized