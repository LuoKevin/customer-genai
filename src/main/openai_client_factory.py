"""Factory for sharing a single OpenAI client instance across agents."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from openai import OpenAI

from .config import AppConfig, load_config


@lru_cache(maxsize=1)
def get_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    config: Optional[AppConfig] = None,
) -> OpenAI:
    """
    Return a shared OpenAI client.

    Args:
        api_key: Optional explicit API key; defaults to config/env OPENAI_API_KEY.
        base_url: Optional override for custom endpoints; defaults to config/env OPENAI_BASE_URL.
        config: Optional pre-loaded AppConfig to avoid reloading .env.
    """
    cfg = config or (load_config() if api_key is None and base_url is None else None)
    key = api_key or (cfg.openai_api_key if cfg else os.getenv("OPENAI_API_KEY"))
    if not key:
        raise EnvironmentError("Set OPENAI_API_KEY or pass api_key to get_openai_client.")
        
    return OpenAI(api_key=key)
