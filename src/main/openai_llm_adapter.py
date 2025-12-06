"""Adapter to use a shared OpenAI client with CrewAI agents."""

from __future__ import annotations

from typing import Any, List

from crewai.llms.base_llm import BaseLLM
from openai import OpenAI
from pydantic import BaseModel


class OpenAIChatLLM(BaseLLM):
    """Minimal adapter implementing CrewAI's BaseLLM interface."""

    is_litellm = False

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini", temperature: float = 0.0):
        super().__init__(model=model, temperature=temperature, api_key=None, base_url=None, provider="openai")
        self.client = client
        self.temperature = temperature

    def call(
        self,
        messages: str | List[dict],
        tools: List[dict] | None = None,
        callbacks: List[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Generate a chat completion and return the text content."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""
