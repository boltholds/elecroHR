import json
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.schemas import CallRecord, CallStatus, TranscriptFragment


async def test_empty_transcript_does_not_call_model():
    from app.scoring import ScoringService

    async with httpx.AsyncClient() as client:
        service = await ScoringService.create(Settings(_env_file=None), client)
        call = CallRecord(call_id="empty", mode="live", status=CallStatus.completed)
        report = await service.score(call)
    assert not report.assessment_complete
    assert report.scoring_method == "insufficient_data"


async def test_invalid_model_json_and_invented_evidence_rejected():
    from app.scoring import ScoringService, empty_report

    call = CallRecord(
        call_id="x",
        mode="live",
        status=CallStatus.completed,
        consent="granted",
        transcript=[
            TranscriptFragment(
                event_id="1", role="candidate", text="I use Python", start_ms=0, end_ms=10
            )
        ],
    )
    response_text = "not json"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["store"] is False
        assert body["text"]["format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": response_text}]}
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = await ScoringService.create(Settings(_env_file=None), client)
        with pytest.raises(ValueError):
            await service.score(call)
        report = empty_report("x", "completed", "live", "test")
        data = report.model_dump(include={"criteria", "strengths", "weaknesses", "summary"})
        data["criteria"][0].update(assessed=True, score=20, evidence=["invented quote"])
        response_text = json.dumps(data)
        with pytest.raises(ValueError, match="evidence"):
            await service.score(call)


async def test_no_consent_transcript_not_saved(tmp_path: Path):
    from app.call_store import CallStore

    call = CallRecord(
        call_id="private",
        mode="live",
        transcript=[
            TranscriptFragment(event_id="1", role="candidate", text="PRIVATE", start_ms=0, end_ms=1)
        ],
    )
    store = CallStore(tmp_path)
    await store.save(call)
    assert "PRIVATE" not in (tmp_path / "private.call.json").read_text()
