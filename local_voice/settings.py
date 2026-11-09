"""Local-only settings. Paths are inside persistent Docker volumes."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LocalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://ollama:11434"
    llm_model: str = "qwen3:4b"
    whisper_path: Path = Path("/models/whisper/small")
    tts_base_url: str = "http://tts:8011"
    data_dir: Path = Path("/data/local_voice")
    knowledge_dir: Path = Path("/srv/knowledge")
    vad_threshold: float = Field(default=0.50, ge=0, le=1)
    silence_frames: int = Field(default=22, ge=5, le=60)
    scoring_timeout_seconds: float = Field(default=150, ge=10, le=600)
