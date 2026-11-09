from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from app.config import Settings
from app.prompts import BACKEND_PROMPT, system_prompt, tool_definitions


class SessionIdentity(BaseModel):
    id: str = Field(min_length=1)


class SessionCreated(BaseModel):
    session: SessionIdentity


class LiveTelephony:
    """Verified Live API contract. No automatic retries for call creation."""

    def __init__(self, config: Settings, client: httpx.AsyncClient):
        self.config, self.client = config, client

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.openai_api_key.get_secret_value()}"}

    async def create(self, destination: str) -> str:
        config = self.config
        response = await self.client.post(
            "https://api.openai.com/v1/live/sessions",
            headers=self.headers(),
            json={
                "session": {
                    "model": config.live_model,
                    "store": False,
                    "instructions": system_prompt(config.require_consent),
                    "audio": {"output": {"voice": config.live_voice}},
                    "delegation": {
                        "type": "responses",
                        "responses": {
                            "model": config.backend_model,
                            "instructions": BACKEND_PROMPT,
                            "tools": tool_definitions(),
                            "tool_choice": "auto",
                            "parallel_tool_calls": False,
                        },
                    },
                },
                "transport": {
                    "type": "sip",
                    "destination": destination,
                    "trunk": {
                        "provider_url": config.sip_provider_url,
                        "auth": {
                            "type": "digest",
                            "username": config.sip_username.get_secret_value(),
                            "password": config.sip_password.get_secret_value(),
                        },
                        "caller_number": config.sip_caller_number,
                    },
                },
            },
            timeout=config.http_timeout_seconds,
        )
        response.raise_for_status()  # Both 200 (guide) and 201 (reference) are accepted.
        return SessionCreated.model_validate(response.json()).session.id

    def sideband_url(self, session_id: str) -> str:
        return f"wss://api.openai.com/v1/live/sessions/{quote(session_id, safe='')}/attach"

    async def hangup(self, session_id: str) -> None:
        response = await self.client.post(
            f"https://api.openai.com/v1/live/sessions/{quote(session_id, safe='')}/hangup",
            headers=self.headers(),
            timeout=self.config.close_timeout_seconds,
        )
        response.raise_for_status()
