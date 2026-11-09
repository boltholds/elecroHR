"""Behavior tests for server integration, without loading model weights or running Docker."""
import json

import httpx

from local_voice.core import empty_assessment, make_report
from local_voice.dialogue import GREETING, Phase
from local_voice.server import Interview
from local_voice.settings import LocalSettings


class FakeAudio:
    def synthesize(self, text: str) -> bytes:
        assert text
        return b"\x00\x00" * 100

    def reset_vad(self) -> None:
        pass


class FakeWS:
    def __init__(self):
        self.events = []
        self.audio_chunks = []

    async def send_json(self, data):
        self.events.append(data)

    async def send_bytes(self, data):
        self.audio_chunks.append(data)


async def test_hr_greets_automatically_without_candidate_prompt():
    async with httpx.AsyncClient() as client:
        interview = Interview(
            "demo", LocalSettings(_env_file=None), FakeAudio(), object(), client
        )
        websocket = FakeWS()
        interview.ws = websocket
        await interview.greet()
        assert interview.state == "speaking"
        assert interview.dialogue.phase == Phase.availability
        assert websocket.audio_chunks
        assert any(e.get("type") == "agent_text" and e["text"] == GREETING
                   for e in websocket.events)
        await interview.on_playback_done(interview.generation)
        assert interview.state == "listening"
        assert interview.transcript == [{"role": "agent", "text": GREETING}]


async def test_scoring_uses_only_answers_from_screening(tmp_path, monkeypatch):
    captured = []

    async def fake_score(session_id, transcript, settings, client):
        captured.extend(transcript)
        return make_report(
            session_id, empty_assessment("Нет технических ответов."),
            "insufficient_data",
        )

    monkeypatch.setattr("local_voice.server.score_transcript", fake_score)
    settings = LocalSettings(_env_file=None, data_dir=tmp_path)
    async with httpx.AsyncClient() as client:
        interview = Interview("scored", settings, FakeAudio(), object(), client)
        interview.transcript = [
            {"role": "agent", "text": "Как вас зовут?"},
            {"role": "candidate", "text": "Меня зовут Александр."},
            {"role": "agent", "text": "Хорошо, Александр."},
        ]
        report = await interview.finish()
    assert captured == []
    assert not report.assessment_complete
    saved = json.loads((tmp_path / "scored.json").read_text(encoding="utf-8"))
    assert saved["profile"]["confirmed"] is False
    assert saved["report"]["score"] == 0
