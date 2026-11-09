import re
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: SecretStr = SecretStr("")
    api_token: SecretStr = SecretStr("")
    live_enabled: bool = False
    live_model: str = "gpt-live-1"
    backend_model: str = "gpt-6-luna"
    scoring_model: str = "gpt-6-luna"
    live_voice: str = "marin"
    sip_provider_url: str = ""
    sip_username: SecretStr = SecretStr("")
    sip_password: SecretStr = SecretStr("")
    sip_caller_number: str = ""
    allowed_test_number: str = ""
    require_consent: bool = True
    data_dir: Path = ROOT / "data"
    knowledge_dir: Path = ROOT / "knowledge"
    rubric_path: Path = ROOT / "scoring_rubric.json"
    call_timeout_seconds: float = Field(default=900, ge=10, le=7200)
    ring_timeout_seconds: float = Field(default=180, ge=1, le=180)
    close_timeout_seconds: float = Field(default=10, ge=1, le=60)
    farewell_seconds: float = Field(default=5, ge=0, le=15)
    http_timeout_seconds: float = Field(default=30, ge=1, le=120)
    scoring_timeout_seconds: float = Field(default=90, ge=1, le=300)

    def validate_live(self, destination: str) -> None:
        if not self.live_enabled:
            raise ValueError("Set LIVE_ENABLED=true to explicitly enable real calls")
        required = {
            "OPENAI_API_KEY": self.openai_api_key.get_secret_value(),
            "API_TOKEN": self.api_token.get_secret_value(),
            "SIP_PROVIDER_URL": self.sip_provider_url,
            "SIP_USERNAME": self.sip_username.get_secret_value(),
            "SIP_PASSWORD": self.sip_password.get_secret_value(),
            "SIP_CALLER_NUMBER": self.sip_caller_number,
            "ALLOWED_TEST_NUMBER": self.allowed_test_number,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError("Missing configuration: " + ", ".join(missing))
        if destination != self.allowed_test_number:
            raise ValueError("Destination must equal ALLOWED_TEST_NUMBER")
        if not re.fullmatch(r"\+[1-9]\d{7,14}", self.sip_caller_number):
            raise ValueError("SIP_CALLER_NUMBER must use E.164")
        if not re.fullmatch(
            r"sips:[a-zA-Z0-9.-]+(?::\d{1,5})?(?:;transport=tcp)?", self.sip_provider_url
        ):
            raise ValueError("SIP_PROVIDER_URL must be a sips: provider host with optional port")
