"""Minimal CrewAI scaffold that routes through the classifier and dispatches handlers.

This keeps things simple:
- Step 1: classify the message using the existing `classify` function.
- Step 2: pick the appropriate agent (feedback or query) based on the label.
- Step 3: run a single-task crew for that agent to produce a response.

Requirements: `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) set in the environment.
"""

import re
import secrets
from typing import Optional

from crewai import Agent, Crew, Process, Task

from src.main.classifier import ClassificationLabel, classify
from src.main.openai_client_factory import get_openai_client
from src.main.openai_llm_adapter import OpenAIChatLLM
from src.main.support_store import create_ticket, get_ticket_status, init_db


def _build_llm(model: str = "gpt-4o-mini") -> OpenAIChatLLM:
    return OpenAIChatLLM(
        client=get_openai_client(),
        model=model,
        temperature=0.0,
    )
    


def handle_message(message: str, *, trace_id: Optional[str] = None, model: str = "gpt-4o-mini") -> str:
    """Entry point: classify and delegate to the appropriate CrewAI agent."""
    init_db()
    classification = classify(message, trace_id=trace_id, model=model)

    if classification.label == ClassificationLabel.QUERY:
        agent = _query_agent(model)
        ticket_number = _extract_ticket_number(message)
        ticket_status = get_ticket_status(ticket_number) if ticket_number else None
        task = _query_task(agent, message, trace_id, ticket_number, ticket_status)
    elif classification.label == ClassificationLabel.POSITIVE_FEEDBACK:
        agent = _positive_feedback_agent(model)
        task = _positive_feedback_task(agent, message, trace_id)
    else:
        agent = _negative_feedback_agent(model)
        ticket_number = _generate_ticket_number()
        create_ticket(ticket_number, message, status="Unresolved")
        task = _negative_feedback_task(agent, message, trace_id, ticket_number)

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff({"message": message, "trace_id": trace_id, "classification": classification.label.value})
    return str(result)


def _positive_feedback_agent(model: str) -> Agent:
    return Agent(
        role="Feedback Handler",
        goal="Respond warmly to positive customer feedback.",
        backstory="You help banking customers feel appreciated when they share praise.",
        llm=_build_llm(model),
        verbose=False,
    )


def _positive_feedback_task(agent: Agent, message: str, trace_id: Optional[str]) -> Task:
    return Task(
        description=(
            f"Customer message: {message}\n"
            "Respond with format: `Thank you for your kind words, [CustomerName]! We're delighted to assist you.` "
            "If no name is provided, omit the name gracefully.\n"
            "Keep it to 1-2 sentences. Include the trace_id if provided."
        ),
        expected_output="A concise, warm thank-you aligned to the provided format.",
        agent=agent,
        tools=[],
        metadata={"trace_id": trace_id},
    )


def _negative_feedback_agent(model: str) -> Agent:
    return Agent(
        role="Feedback Handler",
        goal="Respond empathetically to negative feedback and confirm ticket creation.",
        backstory="You apologize, provide reassurance, and share ticket details for banking customers reporting issues.",
        llm=_build_llm(model),
        verbose=False,
    )


def _negative_feedback_task(
    agent: Agent,
    message: str,
    trace_id: Optional[str],
    ticket_number: str,
) -> Task:
    return Task(
        description=(
            f"Customer message: {message}\n"
            f"Ticket number: {ticket_number}\n"
            "Respond with empathy, apologize, and include the ticket number.\n"
            "Format guidance: `We apologize for the inconvenience. A new ticket #[TicketNumber] has been generated, and our team will follow up shortly.`\n"
            "Keep it to 1-2 sentences. Include the trace_id if provided."
        ),
        expected_output="A concise, empathetic response that includes the ticket number.",
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


def _query_task(
    agent: Agent,
    message: str,
    trace_id: Optional[str],
    ticket_number: Optional[str],
    ticket_status: Optional[str],
) -> Task:
    status_text = (
        f"Ticket {ticket_number} status: {ticket_status}"
        if ticket_number and ticket_status
        else (
            f"Ticket {ticket_number} not found."
            if ticket_number
            else "No ticket number detected."
        )
    )
    return Task(
        description=(
            f"Customer message: {message}\n"
            f"{status_text}\n"
            "If ticket status is known, return it. If not found, state that. "
            "If no ticket number is present, ask politely for one. "
            "Keep it to 1-2 sentences. Include the trace_id if provided."
        ),
        expected_output="A concise status response to the customer's query.",
        agent=agent,
        tools=[],
        metadata={"trace_id": trace_id},
    )


def _generate_ticket_number() -> str:
    """Create a 6-digit ticket number."""
    return f"{secrets.randbelow(900000) + 100000}"


def _extract_ticket_number(text: str) -> Optional[str]:
    """Extract a 6-digit ticket number from text, if present."""
    match = re.search(r"\b(\d{6})\b", text)
    return match.group(1) if match else None


if __name__ == "__main__":
    samples = [
        # "Thanks for resolving my credit card issue.",
        # "My debit card replacement still hasn't arrived.",
        "Mark ticket number 381581 as resolved.",
        "Could you check the status of ticket 381581?",
    ]
    for sample in samples:
        try:
            print(f"\nUser: {sample}")
            reply = handle_message(sample)
            print(f"Agent: {reply}")
        except Exception as exc:  # pragma: no cover - CLI convenience
            print(f"Error handling message: {exc}")
