"""Minimal CrewAI scaffold that routes through the classifier and dispatches handlers.

This keeps things simple:
- Step 1: classify the message using the existing `classify` function.
- Step 2: pick the appropriate agent (feedback or query) based on the label.
- Step 3: run a single-task crew for that agent to produce a response.

Requirements: `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) set in the environment.
"""

import os
from typing import Optional

from crewai import Agent, Crew, LLM, Process, Task

from .classifier_agent import ClassificationLabel, classify


def _build_llm(model: str = "gpt-4o-mini") -> LLM:
    """Create a CrewAI LLM wrapper using OpenAI credentials from the environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY to run CrewAI agents.")
    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs = {"model": model, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return LLM(**kwargs)


def handle_message(message: str, *, trace_id: Optional[str] = None, model: str = "gpt-4o-mini") -> str:
    """Entry point: classify and delegate to the appropriate CrewAI agent."""
    classification = classify(message, trace_id=trace_id, model=model)

    if classification.label == ClassificationLabel.QUERY:
        agent = _query_agent(model)
        task = _query_task(agent, message, trace_id)
    else:
        agent = _feedback_agent(model)
        task = _feedback_task(agent, message, classification.label, trace_id)

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff({"message": message, "trace_id": trace_id, "classification": classification.label.value})
    return str(result)


def _feedback_agent(model: str) -> Agent:
    return Agent(
        role="Feedback Handler",
        goal="Respond to positive or negative customer feedback with empathy and clarity.",
        backstory="You help banking customers feel heard and provide next steps when they report issues.",
        llm=_build_llm(model),
        verbose=False,
    )


def _feedback_task(agent: Agent, message: str, label: ClassificationLabel, trace_id: Optional[str]) -> Task:
    return Task(
        description=(
            "Customer message: {message}\n"
            f"Feedback type: {label.value}\n"
            "If positive, thank the customer warmly. "
            "If negative, apologize and explain a ticket will be created and followed up.\n"
            "Keep it to 1-2 sentences. Include the trace_id if provided."
        ),
        expected_output="A concise, empathetic response appropriate to the feedback type.",
        agent=agent,
        tools=[],
        metadata={"trace_id": trace_id},
    )


def _query_agent(model: str) -> Agent:
    return Agent(
        role="Query Handler",
        goal="Answer customer status questions about tickets as clearly as possible.",
        backstory="You help banking customers understand the status of their support tickets.",
        llm=_build_llm(model),
        verbose=False,
    )


def _query_task(agent: Agent, message: str, trace_id: Optional[str]) -> Task:
    return Task(
        description=(
            "Customer message: {message}\n"
            "If a ticket number is mentioned, echo back that the ticket is being checked "
            "and provide a concise status placeholder. Keep it to 1-2 sentences.\n"
            "Include the trace_id if provided."
        ),
        expected_output="A concise status response to the customer's query.",
        agent=agent,
        tools=[],
        metadata={"trace_id": trace_id},
    )


if __name__ == "__main__":
    samples = [
        "Thanks for resolving my credit card issue.",
        "My debit card replacement still hasn't arrived.",
        "Could you check the status of ticket 650932?",
    ]
    for sample in samples:
        try:
            print(f"\nUser: {sample}")
            reply = handle_message(sample)
            print(f"Agent: {reply}")
        except Exception as exc:  # pragma: no cover - CLI convenience
            print(f"Error handling message: {exc}")
