"""Classifier agent contract for banking support messages.

Step 1: Define inputs/outputs and a callable interface. Logic will be added next.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

from src.main.openai_client_factory import get_openai_client


DEFAULT_MODEL = "gpt-4o-mini"

class ClassificationLabel(str, Enum):
    """Allowed labels for incoming customer messages."""

    POSITIVE_FEEDBACK = "positive_feedback"
    NEGATIVE_FEEDBACK = "negative_feedback"
    QUERY = "query"


@dataclass
class ClassificationResult:
    """Structured output from the classifier agent."""

    label: ClassificationLabel
    rationale: Optional[str] = None  # brief reason/explanation (optional for now)

    @property
    def route(self) -> Literal["feedback_positive_handler", "feedback_negative_handler", "query_handler"]:
        """Downstream agent identifier based on label."""
        if self.label == ClassificationLabel.POSITIVE_FEEDBACK:
            return "feedback_positive_handler"
        if self.label == ClassificationLabel.NEGATIVE_FEEDBACK:
            return "feedback_negative_handler"
        return "query_handler"


def classify(
    message: str,
    *,
    model: str = DEFAULT_MODEL,
    client=None,
    trace_id: Optional[str] = None,
) -> ClassificationResult:
    """Classify a user message with an OpenAI chat model, with graceful fallback."""
    text = (message or "").strip()
    if not text:
        raise ValueError("Message must be a non-empty string.")

    client = client or get_openai_client()
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a banking customer support triage agent. "
                        "Classify the user's message into exactly one of: "
                        "positive_feedback, negative_feedback, query. "
                        "Return JSON with fields: label, rationale."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        raw = completion.choices[0].message.content or "{}"
        parsed = json.loads(raw)
    except Exception:
        rationale = "Classifier unavailable; routing to query handler."
        if trace_id:
            rationale += f" trace_id={trace_id}"
        return ClassificationResult(label=ClassificationLabel.QUERY, rationale=rationale)

    label_value = str(parsed.get("label", "")).strip().lower().replace(" ", "_")

    # Normalize model output to our enum values.
    if label_value in {"positive", "positive_feedback", "pos"}:
        label = ClassificationLabel.POSITIVE_FEEDBACK
    elif label_value in {"negative", "negative_feedback", "neg"}:
        label = ClassificationLabel.NEGATIVE_FEEDBACK
    else:
        label = ClassificationLabel.QUERY

    rationale = parsed.get("rationale") or "No rationale provided by model."
    if trace_id:
        rationale += f" trace_id={trace_id}"
    return ClassificationResult(label=label, rationale=rationale)

if __name__ == "__main__":
    # Simple test
    test_message = "My name is John. I love how easy it is to use your mobile app!"
    result = classify(test_message)
    print(f"Message: {test_message}")
    print(f"Classification: {result.label}, Rationale: {result.rationale}")