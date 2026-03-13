from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8010)
    api_debug: bool = Field(default=False)

    ollama_host: str = Field(default="localhost")
    ollama_port: int = Field(default=11434)
    ollama_timeout: float = Field(default=60.0)

    fast_model: str = Field(default="qwen3.5:9b")
    thinking_model: str = Field(default="qwen3.5:9b")
    research_model: str = Field(default="qwen3.5:9b")

    data_dir: Path = Field(default=Path("./mellow_chat_runtime_data"))
    domain_data_file: Path = Field(default=Path("./mellow_chat_runtime_data/domain_data.json"))
    domain_lookup_backend: str = Field(default="json")
    vectordb_lore_search_url: Optional[str] = Field(default=None)
    vectordb_timeout_sec: float = Field(default=2.0)
    memory_promotion_enabled: bool = Field(default=True)
    memory_promotion_max_items: int = Field(default=20)

    @property
    def ollama_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"


def get_settings() -> Settings:
    return _get_settings()


@lru_cache()
def _get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.domain_data_file.parent.mkdir(parents=True, exist_ok=True)
    return s

