import json

import httpx
import pytest

from local_voice.core import LOCAL_WEIGHTS
from local_voice.scoring import score_transcript
from local_voice.settings import LocalSettings


def example_assessment(quote: str) -> dict:
    return {
        "criteria": [
            {
                "criterion": name, "score": (weight // 2 if name == "rag_retrieval" else 0),
                "max_score": weight, "assessed": name == "docker",
                "evidence": [quote] if name == "docker" else [],
                "explanation": "Есть пример." if name == "docker" else "Не обсуждалось.",
            }
            for name, weight in LOCAL_WEIGHTS.items()
        ],
        "strengths": [], "weaknesses": [], "summary": "Частичная проверка.",
    }


async def test_ollama_scoring_validates_evidence():
    response_body = example_assessment("Я собирал Docker образы.")
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is False
        assert payload["think"] is False
        return httpx.Response(200, json={"message": {"content": json.dumps(response_body)}})

    transcript = [{"role": "candidate", "text": "Я собирал Docker образы."}]
    settings = LocalSettings(_env_file=None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await score_transcript("s1", transcript, settings, client)
        assert report.score == 17 and not report.assessment_complete
        response_body["criteria"][1]["evidence"] = ["такого не было"]
        with pytest.raises(ValueError, match="Unverifiable"):
            await score_transcript("s1", transcript, settings, client)


async def test_empty_transcript_does_not_call_ollama():
    def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Unexpected Ollama call")
    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        report = await score_transcript(
            "s2", [], LocalSettings(_env_file=None), client,
        )
        assert report.scoring_method == "insufficient_data"
