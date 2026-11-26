"""
config.py — one place for all settings.

Every setting comes from an environment variable, so the same code runs
locally, in Docker, or in CI just by changing the environment — never the
code. No API key is ever written in this file.

Values are read when they are ACCESSED (properties), not at import time.
That matters: main.py loads the .env file after importing the modules,
so settings read at import time would miss the .env values. Properties
read the environment fresh on every access, so the order does not matter.

Settings:
  LLM_API_KEY   — API key for the LLM provider. None means: use the mock LLM.
  LLM_BASE_URL  — which API server to talk to (default: OpenAI).
  LLM_MODEL     — which model to use (default: gpt-4o-mini).
  MOCK_LLM      — set to 1/true/yes to force the mock LLM even with a key.
  LOG_LEVEL     — DEBUG / INFO / WARNING / ERROR (default: INFO).
  API_HOST      — host the API binds to (default: 127.0.0.1).
  API_PORT      — port the API listens on (default: 8000).
"""

import logging
import os


class Settings:
    """All configuration, read from the environment with safe defaults."""

    @property
    def llm_api_key(self):
        """API key, or None when no key is set (mock mode)."""
        return os.environ.get("LLM_API_KEY") or None

    @property
    def llm_base_url(self):
        return os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")

    @property
    def llm_model(self):
        return os.environ.get("LLM_MODEL", "gpt-4o-mini")

    @property
    def mock_llm(self):
        """Force the mock LLM even if a key is set (1 / true / yes)."""
        return os.environ.get("MOCK_LLM", "").lower() in ("1", "true", "yes")

    @property
    def log_level(self):
        return os.environ.get("LOG_LEVEL", "INFO").upper()

    @property
    def api_host(self):
        return os.environ.get("API_HOST", "127.0.0.1")

    @property
    def api_port(self):
        return int(os.environ.get("API_PORT", "8000"))


# One shared settings object for the whole project.
settings = Settings()


def setup_logging(level=None):
    """Configure one-line, structured (key=value) logging.

    Every pipeline step logs exactly one line, for example:
        step=classify ticket_id=T-1001 category=billing priority=high
    That makes the logs greppable: `grep ticket_id=T-1001 app.log`
    shows everything that happened to one ticket. The CLI's pretty
    THINK / ACT / OBSERVE printing stays as it is — logging is the
    machine-readable record, the prints are the human-readable demo.
    """
    chosen = level or settings.log_level
    logging.basicConfig(
        level=getattr(logging, chosen, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("support_desk")
