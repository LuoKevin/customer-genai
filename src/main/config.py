"""Centralized application configuration using Pydantic settings."""

from __future__ import annotations

from typing import Optional

from pydantic import  Field
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(None, env="OPENAI_BASE_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_config() -> AppConfig:
    """Load configuration from environment (and .env if present)."""
    return AppConfig()

