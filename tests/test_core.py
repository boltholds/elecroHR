from pathlib import Path

import pytest
from pydantic import ValidationError


def test_state_machine_rejects_skip_and_terminal_transition():
    from app.schemas import CallRecord, CallStatus

    call = CallRecord(call_id="test", mode="demo")
    with pytest.raises(ValueError):
        call.transition(CallStatus.completed)
    for status in [CallStatus.dialing, CallStatus.ringing, CallStatus.active, CallStatus.completed]:
        call.transition(status)
    with pytest.raises(ValueError):
        call.transition(CallStatus.active)


async def test_rag_ranks_sources_and_unknown(tmp_path: Path):
    from app.rag import RAGService

    (tmp_path / "vacancy.md").write_text(
        "Python FastAPI REST API\n\nPostgreSQL SQL indexes", encoding="utf-8"
    )
    (tmp_path / "faq.md").write_text("Тестовые условия: удаленная работа", encoding="utf-8")
    rag = await RAGService.load(tmp_path)
    found = rag.search_vacancy("PostgreSQL SQL")
    assert found[0].source == "vacancy.md"
    assert "indexes" in found[0].text
    assert found[0].score > 0
    assert rag.search_vacancy("xyznotfound") == []
    assert rag.search_vacancy("") == []


async def test_rag_empty_directory(tmp_path: Path):
    from app.rag import RAGService

    with pytest.raises(ValueError, match="knowledge"):
        await RAGService.load(tmp_path)


def test_report_rejects_incorrect_sum_and_unassessed_score():
    from app.schemas import Report
    from app.scoring import empty_report

    report = empty_report("x", "completed", "demo", "No evidence")
    assert report.score == 0
    assert len(report.unassessed_criteria) == 4
    data = report.model_dump()
    data["score"] = 1
    with pytest.raises(ValidationError):
        Report.model_validate(data)
    data["score"] = 0
    data["criteria"][0]["score"] = 1
    with pytest.raises(ValidationError):
        Report.model_validate(data)


async def test_store_restart_marks_interrupted(tmp_path: Path):
    from app.call_store import CallStore
    from app.schemas import CallRecord, CallStatus

    store = CallStore(tmp_path)
    call = CallRecord(call_id="test", mode="demo")
    call.transition(CallStatus.dialing)
    await store.save(call)
    restored = CallStore(tmp_path)
    await restored.load()
    result = restored.get("test")
    assert result.status == CallStatus.failed
    assert result.error_code == "process_interrupted"


def test_live_config_is_explicit():
    from app.config import Settings

    config = Settings(_env_file=None)
    with pytest.raises(ValueError, match="LIVE_ENABLED"):
        config.validate_live("+12025550123")
