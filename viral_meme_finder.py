# viral_meme_finder.py

from __future__ import annotations

import json
from typing import List, Dict, Any

from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from datetime import date, timedelta

from youtube import _normalize_memes_with_youtube_links

# Shared web-search tool for both agents
search_tool = SerperDevTool()
scraper = ScrapeWebsiteTool()

@CrewBase
class ViralMemeCrew:
    """Viral Meme Finder crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            tools=[search_tool],
            llm=LLM(model="gpt-4o-mini", temperature=0.2),
            verbose=True,
            max_iter=5
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst"],
            tools=[scraper],
            llm=LLM(model="gpt-4o-mini", temperature=0.2),
            verbose=True,
            max_iter=3
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
        )

    @task
    def aggregate_task(self) -> Task:
        return Task(
            config=self.tasks_config["aggregate_task"],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.researcher(), self.analyst()],
            tasks=[self.research_task(), self.aggregate_task()],
            process=Process.sequential,
            verbose=False,
        )



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

    today = date.today()
    cutoff = today - timedelta(days=days_back)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    viral_meme_crew = ViralMemeCrew().crew()
    raw_result = viral_meme_crew.kickoff(
        inputs={
            "days_back": days_back,
            "max_memes": max_memes,
            "cutoff_date": cutoff_str
        }
    )


    # Normalize to a list
    if isinstance(raw_result, list):
        memes = raw_result
    elif isinstance(raw_result, dict):
        memes = [raw_result]
    elif isinstance(raw_result, str):
        raw = raw_result.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                memes = parsed
            else:
                memes = [parsed]
        except json.JSONDecodeError:
            memes = [{"raw_output": raw}]
    else:
        memes = [{"raw_output": str(raw_result)}]

    # Last-resort fallback
    if len(memes) == 1 and "raw_output" in memes[0]:
        return memes

    # Replace evidence_links with YouTube search URLs
    return _normalize_memes_with_youtube_links(memes)
