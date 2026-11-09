import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings


def wait_report(client, call_id):
    for _ in range(200):
        response = client.get(f"/calls/{call_id}/report")
        if response.status_code == 200:
            return response.json()
        time.sleep(0.01)
    raise AssertionError("Report did not finish")


def test_demo_end_to_end_and_restart(tmp_path: Path):
    from app.main import create_app

    settings = Settings(_env_file=None, data_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/calls", json={"mode": "demo"})
        assert response.status_code == 202
        call_id = response.json()["call_id"]
        report = wait_report(client, call_id)
        assert report["mode"] == "demo"
        assert report["scoring_method"] == "demo_rules"
        assert report["score"] == sum(c["score"] for c in report["criteria"])
        assert report["recommendation"] == "manual_review"
        assert report["assessment_complete"]
        state = client.get(f"/calls/{call_id}").json()
        assert state["history"] == ["created", "dialing", "ringing", "active", "completed"]
        assert (tmp_path / f"{call_id}.report.json").exists()
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/calls/{call_id}/report").json() == report


def test_live_missing_configuration_and_unknown_call(tmp_path: Path):
    from app.main import create_app

    with TestClient(create_app(Settings(_env_file=None, data_dir=tmp_path))) as client:
        response = client.post(
            "/calls",
            json={"mode": "live", "destination": "+12025550123", "confirm_test_call": True},
        )
        assert response.status_code == 503
        assert "LIVE_ENABLED" in response.json()["detail"]
        assert client.get("/calls/not-found").status_code == 404
        assert client.get("/calls/not-found/report").status_code == 404
        assert client.post("/calls", json={"mode": "live", "destination": "123"}).status_code == 422


def test_api_token_protects_calls_and_reports(tmp_path: Path):
    from app.main import create_app

    with TestClient(
        create_app(Settings(_env_file=None, data_dir=tmp_path, api_token="test-token"))
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/calls", json={"mode": "demo"}).status_code == 401
        assert client.get("/calls/unknown/report").status_code == 401
        assert (
            client.post(
                "/calls", json={"mode": "demo"}, headers={"Authorization": "Bearer test-token"}
            ).status_code
            == 202
        )


async def test_single_call_reservation(tmp_path: Path):
    import httpx
    import pytest

    from app.call_store import CallStore
    from app.controller import BusyError, CallController
    from app.rag import RAGService
    from app.schemas import DemoRequest
    from app.scoring import ScoringService

    config = Settings(_env_file=None, data_dir=tmp_path)
    async with httpx.AsyncClient() as client:
        controller = CallController(
            config,
            CallStore(tmp_path),
            await RAGService.load(config.knowledge_dir),
            await ScoringService.create(config, client),
            client,
        )
        await controller.start(DemoRequest())
        with pytest.raises(BusyError):
            await controller.start(DemoRequest())
        await controller.shutdown()
