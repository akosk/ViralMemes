# viral_meme_finder.py

from __future__ import annotations

import json
from typing import List, Dict, Any

from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool


# Shared web-search tool for both agents
search_tool = SerperDevTool()


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
            verbose=False,
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst"],
            tools=[search_tool],
            llm=LLM(model="gpt-4o-mini", temperature=0.2),
            verbose=False,
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
    viral_meme_crew = ViralMemeCrew().crew()
    raw_result = viral_meme_crew.kickoff(
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
