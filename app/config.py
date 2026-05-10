"""Runtime configuration for the Maseru Health AI application."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args, **_kwargs):
        return False

from src.paths import ROOT_DIR


def _streamlit_secret(name: str) -> str | None:
    """Read a root-level Streamlit secret when running on Streamlit Cloud."""
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        return None

    return str(value) if value else None


def _streamlit_nested_secret(section: str, name: str) -> str | None:
    """Read a nested Streamlit secret such as [openai] api_key = '...'."""
    try:
        import streamlit as st

        section_values = st.secrets.get(section, {})
        value = section_values.get(name)
    except Exception:
        return None

    return str(value) if value else None


def _setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from environment variables, then Streamlit secrets."""
    return os.getenv(name) or _streamlit_secret(name) or default


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_name: str = "maseru_health_support"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None

    @property
    def llm_available(self) -> bool:
        """Return whether the configured LiteLLM/OpenAI model can be called."""
        return bool(self.openai_api_key)


def get_settings() -> Settings:
    """Load settings from `.env`, environment variables, and Streamlit secrets."""
    load_dotenv(ROOT_DIR / ".env")
    openai_api_key = (
        _setting("OPENAI_API_KEY")
        or _streamlit_nested_secret("openai", "api_key")
    )

    if openai_api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openai_api_key

    return Settings(
        app_name=_setting("MASERU_APP_NAME", "maseru_health_support") or "maseru_health_support",
        llm_model=_setting("MASERU_LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        openai_api_key=openai_api_key,
    )
