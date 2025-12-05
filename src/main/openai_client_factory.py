"""Factory for sharing a single OpenAI client instance across agents."""

import os
from functools import lru_cache
from typing import Optional

from openai import OpenAI


@lru_cache(maxsize=1)
def get_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    Return a shared OpenAI client.

    Args:
        api_key: Optional explicit API key; defaults to env OPENAI_API_KEY.
        base_url: Optional override for custom endpoints; defaults to env OPENAI_BASE_URL.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("Set OPENAI_API_KEY or pass api_key to get_openai_client.")

    endpoint = base_url or os.getenv("OPENAI_BASE_URL")
    if endpoint:
        return OpenAI(api_key=key, base_url=endpoint)
    return OpenAI(api_key=key)

