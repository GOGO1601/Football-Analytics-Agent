"""
API settings, read from environment variables (or agent-backend/.env; see .env.example).

    APP_ENV        development | test | production                     (default development)
    USE_MOCK_LLM   true = offline deterministic LLM (app/offline.py)    (default true)
                   false is rejected until the real LLM client exists (a later stage)
    CORS_ORIGINS   comma-separated browser origins allowed to call the API.
                   Default: WebStorm's built-in server, which serves the dashboard when you
                   press Debug on home.html (port 63342). "*" is not accepted.
    PRELOAD_DATA   load the match data at start-up so the first question is not slow (default true)

Other keys in .env (GROQ_API_KEY, MODEL_ROUTER, ...) are ignored here until their stage.
No secret is read by this stage.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CORS_ORIGINS = "http://localhost:63342,http://127.0.0.1:63342"
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.\-]+(:\d{1,5})?$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    use_mock_llm: bool = True
    cors_origins: str = DEFAULT_CORS_ORIGINS
    preload_data: bool = True

    @field_validator("cors_origins")
    @classmethod
    def _explicit_origins(cls, v: str) -> str:
        origins = [o.strip() for o in v.split(",") if o.strip()]
        bad = [o for o in origins if not _ORIGIN.match(o)]
        if bad:
            raise ValueError(f"CORS_ORIGINS must be explicit origins like http://localhost:63342 "
                             f"(scheme://host[:port], no path, no '*'); invalid: {bad}")
        return ",".join(origins)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o for o in self.cors_origins.split(",") if o]


@lru_cache
def get_settings() -> Settings:
    return Settings()