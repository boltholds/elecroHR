"""Latency metrics for voice turns, model inference and storage."""
import json

import httpx
import pytest

from app.schemas import RetrievedChunk
from local_voice.core import empty_assessment, make_report
from local_voice.dialogue import HRDialogue, Phase
from local_voice.metrics import TurnTiming, summarize_timings
from local_voice.server import Interview
from local_voice.settings import LocalSettings


def test_timing_summary_omits_missing_stages_and_greetings():
    greeting = TurnTiming(turn_id=1, phase="greeting", kind="greeting")
    greeting.add("tts_http_ms", 950)
    greeting.finish()
    a = TurnTiming(turn_id=2, phase="screening")
    a.add("stt_ms", 180)
    a.add("llm_http_ms", 200)
    a.finish()
    b = TurnTiming(turn_id=3, phase="screening")
    b.add("stt_ms", 420)
    b.add("tts_http_ms", 900)
    b.finish()
    third = TurnTiming(turn_id=4, phase="screening")
    third.add("stt_ms", 3000)
    third.finish("failed")
    summary = summarize_timings([greeting, a, b, third])
    assert summary["completed_turns"] == 2
    assert summary["average_ms"]["stt_ms"] == 300
    assert summary["p95_ms"]["stt_ms"] == 420
    assert summary["average_ms"]["tts_http_ms"] == 900
    assert summary["average_ms"]["llm_http_ms"] == 200
    assert "dialogue_ms" not in summary["average_ms"]


def test_timing_safe_and_no_text_leaks():
    turn = TurnTiming(turn_id=9, phase="screening")
    turn.add("stt_ms", 12.1234)
    turn.add("stt_ms", 7.0)
    turn.add("llm_http_ms", float("inf"))
    turn.vad_frames = 14
    result = turn.public()
    assert result["stages_ms"]["stt_ms"] == 19.12
    assert "llm_http_ms" not in result["stages_ms"]
    assert result["vad_frames"] == 14
    assert "transcript" not in result
    assert "reasoning" not in json.dumps(result)


class FakeAudio:
    def __init__(self):
        self.spoken = []

    def transcribe(self, _pcm):
        return "Какая зарплата?"

    def synthesize_timed(self, text):
        self.spoken.append(text)
        return b"\x00\x00", {"tts_model_ms": 9.0}

    def reset_vad(self):
        pass


class FakeSocket:
    def __init__(self):
        self.events = []

    async def send_json(self, item):
        self.events.append(item)

    async def send_bytes(self, item):
        self.events.append(item)


class FakeRAG:
    def search_vacancy(self, query, top_k=5):
        return [RetrievedChunk(
            chunk_id="demo-salary", source="vacancy.md",
            text="Зарплата: 200 000–280 000 рублей на руки. Демо-вакансия.",
            score=3.0,
        )]


def ollama_handler(request):
    payload = json.loads(request.content)
    assert payload["stream"] is False
    assert payload["think"] is False
    return httpx.Response(200, json={
        "message": {"content": json.dumps({
            "answer": "В демонстрационной вакансии указано от 200 до 280 тысяч рублей."
        }, ensure_ascii=False)},
        "total_duration": 185_000_000,
        "prompt_eval_duration": 25_000_000,
        "eval_duration": 150_000_000,
        "load_duration": 0,
    })


async def test_full_voice_turn_reports_stages_and_playback_metrics():
    audio = FakeAudio()
    sock = FakeSocket()
    dialogue = HRDialogue(
        phase=Phase.screening, confirmed=True, preferred_name="Антон", form="ты",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(ollama_handler)) as client:
        interview = Interview(
            session_id="test", settings=LocalSettings(_env_file=None),
            audio=audio, rag=FakeRAG(), http=client,
            dialogue=dialogue, state="thinking",
        )
        interview.ws = sock
        await interview.handle_turn(b"\x00\x00")
        assert interview.state == "speaking"
        assert interview.active_timing is not None
        await interview.on_playback_done(interview.generation, playback_ms=400.0)
    assert interview.state == "listening"
    assert len(interview.timings) == 1
    last = interview.timings[0]
    assert last.status == "completed"
    for key in (
        "stt_ms", "vacancy_rag_ms", "dialogue_ms",
        "llm_http_ms", "tts_http_ms", "tts_model_ms",
        "websocket_send_ms", "first_audio_ms", "backend_ms",
        "client_playback_ms", "playback_ack_ms", "end_to_end_ms",
        "ollama_prompt_ms", "ollama_decode_ms",
    ):
        assert key in last.stages_ms, key
    assert last.stages_ms["client_playback_ms"] == 400
    assert last.stages_ms["tts_model_ms"] == 9
    assert last.stages_ms["ollama_decode_ms"] == 150
    assert last.stages_ms["first_audio_ms"] <= last.stages_ms["backend_ms"]
    assert last.stages_ms["backend_ms"] <= last.stages_ms["end_to_end_ms"]
    assert any(x.get("type") == "timing" for x in sock.events if isinstance(x, dict))
    assert interview.metrics_public()["summary"]["completed_turns"] == 1


async def test_finish_persists_timing_without_private_model_prompts(tmp_path, monkeypatch):
    async def fake_score(*args, **kwargs):
        return make_report(
            "sample", empty_assessment("Insufficient evidence"), "insufficient_data"
        )

    import local_voice.server as local_server

    monkeypatch.setattr(local_server, "score_transcript", fake_score)
    async with httpx.AsyncClient() as client:
        interview = Interview(
            session_id="sample",
            settings=LocalSettings(_env_file=None, data_dir=tmp_path),
            audio=FakeAudio(), rag=FakeRAG(), http=client,
            state="listening",
        )
        completed = TurnTiming(turn_id=1, phase="screening")
        completed.add("stt_ms", 123)
        completed.finish()
        interview.timings.append(completed)
        report = await interview.finish()
    saved = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))
    assert report.status == "completed"
    assert saved["timings"]["summary"]["completed_turns"] == 1
    assert saved["timings"]["turns"][0]["stages_ms"]["stt_ms"] == 123
    assert saved["timings"]["scoring_ms"] >= 0
