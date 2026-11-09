"""Ollama timing spans: server durations, token counts, per-stage aggregation."""
import json
import math

import httpx
import pytest

from local_voice.dialogue import HRDialogue, Phase
from local_voice.metrics import TurnTiming, summarize_timings
from local_voice.ollama_metrics import ollama_call_stats, record_ollama_call
from local_voice.replies import generate_candidate_reply
from local_voice.scoring import score_transcript
from local_voice.settings import LocalSettings


OLLAMA_TIMING = {
    "load_duration": 10_000_000,
    "prompt_eval_duration": 200_000_000,
    "eval_duration": 400_000_000,
    "total_duration": 640_000_000,
    "prompt_eval_count": 120,
    "eval_count": 20,
}


def test_durations_tokens_and_rates_are_in_separate_units():
    stats = ollama_call_stats(
        OLLAMA_TIMING, operation="react", wall_ms=670.1
    )
    assert stats == {
        "operation": "react", "status": "ok", "wall_ms": 670.1,
        "load_ms": 10, "prompt_ms": 200, "decode_ms": 400,
        "total_ms": 640, "input_tokens": 120, "output_tokens": 20,
        "decode_tokens_per_s": 50.0, "prompt_tokens_per_s": 600.0,
    }
    timings = {}
    calls = []
    record_ollama_call(
        OLLAMA_TIMING, operation="react", wall_ms=670.1,
        timings_ms=timings, calls=calls,
    )
    assert timings["react_ollama_prompt_ms"] == 200
    assert timings["react_ollama_decode_ms"] == 400
    assert timings["react_ollama_wall_ms"] == 670.1
    assert "react_ollama_output_tokens" not in timings
    assert calls[0]["output_tokens"] == 20


def test_missing_or_invalid_provider_fields_do_not_invent_zeros():
    response = {
        "load_duration": -1,
        "eval_duration": float("nan"),
        "prompt_eval_duration": True,
        "total_duration": float("inf"),
        "eval_count": True,
        "prompt_eval_count": -5,
        "message": {"thinking": "secret", "content": "secret answer"},
    }
    result = ollama_call_stats(response, operation="react", wall_ms=99)
    assert result == {"operation": "react", "status": "ok", "wall_ms": 99}
    assert "secret" not in json.dumps(result)
    with pytest.raises(ValueError):
        ollama_call_stats({}, operation="malicious", wall_ms=20)


def test_errors_are_counted_but_do_not_fake_provider_data():
    calls = []
    record_ollama_call({}, operation="react", wall_ms=12, calls=calls, success=False)
    assert calls == [{"operation": "react", "status": "error", "wall_ms": 12}]


def test_timing_summary_groups_ollama_operations_without_double_counting():
    turn = TurnTiming(turn_id=1, phase="screening")
    record_ollama_call(
        OLLAMA_TIMING, operation="react", wall_ms=670,
        timings_ms=turn.stages_ms, calls=turn.ollama_calls,
    )
    record_ollama_call(
        OLLAMA_TIMING, operation="answer", wall_ms=780,
        timings_ms=turn.stages_ms, calls=turn.ollama_calls,
    )
    assert turn.stages_ms["ollama_decode_ms"] == 400
    turn.finish()
    summary = summarize_timings([turn])
    assert summary["completed_turns"] == 1
    assert summary["ollama_summary"]["react"]["avg_decode_tokens_per_s"] == 50
    assert summary["ollama_summary"]["react"]["avg_input_tokens"] == 120
    assert summary["ollama_summary"]["answer"]["avg_wall_ms"] == 780
    assert summary["ollama_summary"]["react"]["p95_wall_ms"] == 670
    assert len(turn.public()["ollama_calls"]) == 2
    assert "secret" not in json.dumps(turn.public())


async def test_intro_and_react_llm_calls_have_separate_stage_names():
    def handler(request):
        payload = json.loads(request.content)
        fmt = payload["format"]
        if "answer" in fmt["properties"]:
            answer = {"answer": "yes"}
        else:
            answer = {
                "action": "end_interview", "focus": "",
                "query": "", "instruction": "",
            }
        return httpx.Response(200, json={
            "message": {"content": json.dumps(answer)},
            **OLLAMA_TIMING,
        })

    settings = LocalSettings(_env_file=None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        dialogue = HRDialogue()
        intro_ms, intro_calls = {}, []
        answer = await dialogue.respond(
            "Да", [], client, settings, object(),
            timings_ms=intro_ms, ollama_calls=intro_calls,
        )
        assert dialogue.phase == Phase.name
        assert "имя" in answer.text.lower()
        assert intro_calls[0]["operation"] == "availability"
        assert intro_ms["availability_ollama_prompt_ms"] == 200
        assert "react_ollama_prompt_ms" not in intro_ms

        dialogue.phase = Phase.screening
        dialogue.confirmed = True
        dialogue.preferred_name = "Тест"
        dialogue.form = "вы"
        react_ms, react_calls = {}, []
        ending = await dialogue.respond(
            "Давайте закончим.", [], client, settings, object(),
            timings_ms=react_ms, ollama_calls=react_calls,
        )
        assert dialogue.phase == Phase.finished
        assert ending.text
        assert react_calls[0]["operation"] == "react"
        assert react_ms["react_ollama_prompt_ms"] == 200
        assert react_ms["react_ollama_decode_ms"] == 400
        assert react_ms["react_llm_ms"] >= 0


async def test_candidate_answer_profiles_ollama_without_leaking_thinking():
    def handler(request):
        return httpx.Response(200, json={
            "message": {
                "thinking": "private hidden reasoning",
                "content": '{"answer":"Работа удалённая."}',
            },
            **OLLAMA_TIMING,
        })
    timings, calls = {}, []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        text = await generate_candidate_reply(
            client, "http://ollama:11434", "qwen3:4b", "Скажи условия",
            [{"role": "candidate", "text": "Можно удалённо?"}], vacancy=True,
            timings_ms=timings, ollama_calls=calls,
        )
    assert text == "Работа удалённая."
    assert calls[0]["operation"] == "answer"
    assert calls[0]["output_tokens"] == 20
    assert timings["ollama_prompt_ms"] == 200
    assert timings["answer_ollama_decode_ms"] == 400
    assert "private" not in json.dumps(calls)


async def test_scoring_profiles_ollama_separately():
    from local_voice.core import LOCAL_WEIGHTS

    assessment = {
        "criteria": [
            {
                "criterion": criterion, "score": 0,
                "max_score": weight, "evidence": [],
                "assessed": False, "explanation": "Не проверено",
            }
            for criterion, weight in LOCAL_WEIGHTS.items()
        ],
        "strengths": [], "weaknesses": [],
        "summary": "Недостаточно данных.",
    }

    def handler(request):
        return httpx.Response(200, json={
            "message": {"content": json.dumps(assessment, ensure_ascii=False)},
            **OLLAMA_TIMING,
        })

    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await score_transcript(
            "session-id", [{"role": "candidate", "text": "Я писал API."}],
            LocalSettings(_env_file=None), client, ollama_calls=calls,
        )
    assert report.scoring_method == "ollama"
    assert calls[0]["operation"] == "scoring"
    assert calls[0]["input_tokens"] == 120
    assert "Я писал API" not in json.dumps(calls, ensure_ascii=False)


async def test_failed_ollama_http_call_records_operation():
    def handler(request):
        return httpx.Response(503)
    settings = LocalSettings(_env_file=None)
    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await HRDialogue().structured(
                client, settings,
                __import__("local_voice.dialogue", fromlist=["Intent"]).Intent,
                "Определи доступность", "Да",
                operation="availability", ollama_calls=calls,
            )
    assert calls[0]["operation"] == "availability"
    assert calls[0]["status"] == "error"
    assert "output_tokens" not in calls[0]
