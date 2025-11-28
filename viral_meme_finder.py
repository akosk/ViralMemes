# viral_meme_finder.py

from __future__ import annotations

import json
from typing import List, Dict, Any

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool


# Shared web-search tool for both agents
search_tool = SerperDevTool()


def _build_crew() -> Crew:
    """Create a crew with two agents that collaborate to find recent viral memes."""
    researcher = Agent(
        role="Viral Meme Researcher",
        goal=(
            "Find memes that went viral on the internet within the last "
            "{days_back} days."
        ),
        backstory=(
            "You track current internet culture and trends across platforms such as "
            "X/Twitter, Reddit, TikTok, Instagram, and YouTube. You know how to "
            "quickly identify which memes are 'viral' based on recency, reach, and "
            "engagement."
        ),
        tools=[search_tool],
        verbose=False,
        allow_delegation=False,
    )

    analyst = Agent(
        role="Meme Analyst and Summarizer",
        goal=(
            "Given raw research results about recent memes, filter and summarize "
            "only the truly viral ones, and return a clean JSON list."
        ),
        backstory=(
            "You specialize in understanding why memes go viral and can distill the "
            "most important details: name, where they went viral, why, and links."
        ),
        tools=[search_tool],
        verbose=False,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            "Search the web and social platforms for memes that went viral in the "
            "last {days_back} days. Consider platforms such as X/Twitter, Reddit, "
            "TikTok, Instagram, and YouTube.\n\n"
            "Focus only on memes that clearly show viral behavior, such as:\n"
            "- Large engagement numbers (likes, shares, comments).\n"
            "- Being widely discussed in multiple sources.\n"
            "- Being mentioned in news or culture articles.\n\n"
            "Collect significantly more candidates than {max_memes}, then pass "
            "your notes to the next agent."
        ),
        agent=researcher,
        expected_output=(
            "A bullet list of candidate viral memes from the last {days_back} days, "
            "with rough evidence and links for each."
        ),
    )

    aggregate_task = Task(
        description=(
            "From the research notes above, select up to {max_memes} memes that are "
            "clearly viral and recent (within the last {days_back} days).\n\n"
            "For each meme, produce a JSON object with the following fields:\n"
            "- title: short recognizable name of the meme\n"
            "- primary_platform: main platform where it went viral (e.g. 'TikTok')\n"
            "- summary: 2–3 sentence explanation of the meme and why it is viral\n"
            "- evidence_links: list of 2–5 URLs that show the meme or report on it\n"
            "- started_around: approximate start date as 'YYYY-MM-DD'\n"
            "- tags: list of simple tags (e.g. ['reaction', 'video', 'sound'])\n\n"
            "Return ONLY a JSON array of these objects, with no extra text, "
            "no markdown, and no comments."
        ),
        agent=analyst,
        expected_output=(
            "A pure JSON array (no surrounding text) of up to {max_memes} meme "
            "objects with the exact fields requested."
        ),
        # output_json=True,  # <-- REMOVE THIS LINE
    )

    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, aggregate_task],
        process=Process.sequential,
        verbose=False,
    )

    return crew


def get_recent_viral_memes(days_back: int = 14, max_memes: int = 10) -> List[Dict[str, Any]]:
    """
    Run the crew to search for viral memes from the last `days_back` days.

    Returns a list of dicts with:
        - title
        - primary_platform
        - summary
        - evidence_links
        - started_around
        - tags
    """
    crew = _build_crew()
    raw_result = crew.kickoff(
        inputs={
            "days_back": days_back,
            "max_memes": max_memes,
        }
    )

    # crewAI may already give us parsed JSON depending on version/config
    if isinstance(raw_result, list):
        return raw_result

    if isinstance(raw_result, dict):
        # sometimes the final result is nested
        # you can adjust this depending on how your crew is configured
        return [raw_result]

    if isinstance(raw_result, str):
        # Attempt to parse JSON from the string
        raw_result = raw_result.strip()
        try:
            parsed = json.loads(raw_result)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except json.JSONDecodeError:
            # Fallback: wrap in a single item if parsing fails
            return [{"raw_output": raw_result}]

    # Last-resort fallback
    return [{"raw_output": str(raw_result)}]
