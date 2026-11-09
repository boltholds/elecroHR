import asyncio
import json
from pathlib import Path

import httpx
import pytest

from app.call_store import CallStore
from app.config import Settings
from app.controller import BusyError, CallController
from app.rag import RAGService
from app.schemas import CallRecord, CallStatus, LiveRequest
from app.scoring import ScoringService


def live_config(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        live_enabled=True,
        openai_api_key="fake",
        api_token="fake",
        sip_provider_url="sips:example.com",
        sip_username="fake",
        sip_password="fake",
        sip_caller_number="+12025550100",
        allowed_test_number="+12025550123",
    )


async def controller_for(config, client):
    return CallController(
        config,
        CallStore(config.data_dir),
        await RAGService.load(config.knowledge_dir),
        await ScoringService.create(config, client),
        client,
    )


async def test_ambiguous_creation_blocks_redial_and_survives_restart(tmp_path: Path):
    config = live_config(tmp_path)
    count = 0

    def timeout(request):
        nonlocal count
        count += 1
        raise httpx.ReadTimeout("secret-provider-error", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
        controller = await controller_for(config, client)
        request = LiveRequest(
            mode="live", destination=config.allowed_test_number, confirm_test_call=True
        )
        call = await controller.start(request)
        await controller.tasks[call.call_id]
        assert count == 1
        assert call.error_code == "dialing_outcome_unknown"
        with pytest.raises(BusyError):
            await controller.start(request)
        interrupted = CallRecord(
            call_id="crash",
            mode="live",
            status=CallStatus.dialing,
            error_code="dialing_outcome_unknown",
        )
        await controller.store.save(interrupted)
        restored = CallStore(tmp_path)
        await restored.load()
        assert restored.get("crash").error_code == "dialing_outcome_unknown"


async def test_sideband_disconnect_is_failed_and_hangs_up(tmp_path: Path, monkeypatch):
    from app import voice_agent

    class BrokenSocket:
        async def __aenter__(self):
            raise OSError("socket unavailable")

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(voice_agent, "connect", lambda *args, **kwargs: BrokenSocket())
    paths = []

    def provider(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"session": {"id": "live_1"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        config = live_config(tmp_path)
        controller = await controller_for(config, client)
        call = await controller.start(
            LiveRequest(mode="live", destination=config.allowed_test_number, confirm_test_call=True)
        )
        await controller.tasks[call.call_id]
        assert call.status == CallStatus.failed
        assert not call.finalization_confirmed
        assert paths == ["/v1/live/sessions", "/v1/live/sessions/live_1/hangup"]
        assert controller.store.reports[call.call_id].scoring_method == "insufficient_data"


async def test_live_sideband_flow_and_graceful_cancel(tmp_path: Path, monkeypatch):
    from app import voice_agent

    events = asyncio.Queue()
    sent = []
    for event in [
        {"type": "transport.ringing", "event_id": "ring"},
        {"type": "transport.answered", "event_id": "answer"},
        {
            "type": "session.input_transcript.delta",
            "event_id": "input",
            "delta": "Нет, спасибо.",
            "start_ms": 10,
            "end_ms": 100,
        },
    ]:
        events.put_nowait(json.dumps(event))

    class Socket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def recv(self):
            return await events.get()

        async def send(self, value):
            sent.append(json.loads(value))

    monkeypatch.setattr(voice_agent, "connect", lambda *args, **kwargs: Socket())

    def provider(request):
        if request.url.path.endswith("/hangup"):
            events.put_nowait(
                json.dumps(
                    {
                        "type": "session.closed",
                        "event_id": "closed",
                        "reason": "close_requested",
                        "usage": {"seconds": 1},
                    }
                )
            )
        return httpx.Response(200, json={"session": {"id": "live_1"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        config = live_config(tmp_path)
        controller = await controller_for(config, client)
        call = await controller.start(
            LiveRequest(mode="live", destination=config.allowed_test_number, confirm_test_call=True)
        )
        for _ in range(100):
            if call.status == CallStatus.active:
                break
            await asyncio.sleep(0.01)
        assert call.status == CallStatus.active
        controller.cancel(call.call_id)
        await asyncio.wait_for(controller.tasks[call.call_id], timeout=3)
        assert call.status == CallStatus.canceled
        assert call.finalization_confirmed
        assert call.transcript == []
        assert not any(e["type"] == "session.start" for e in sent)


def test_destination_allowlist(tmp_path: Path):
    config = live_config(tmp_path)
    config.validate_live(config.allowed_test_number)
    with pytest.raises(ValueError, match="ALLOWED_TEST_NUMBER"):
        config.validate_live("+12025550999")


async def test_final_snapshot_failure_exposes_report_error(tmp_path: Path, monkeypatch):
    from app.schemas import DemoRequest

    async with httpx.AsyncClient() as client:
        controller = await controller_for(Settings(_env_file=None, data_dir=tmp_path), client)
        original = controller.store.save

        async def fail_terminal(call):
            if call.status == CallStatus.completed:
                raise OSError("disk full")
            await original(call)

        monkeypatch.setattr(controller.store, "save", fail_terminal)
        call = await controller.start(DemoRequest())
        await asyncio.gather(controller.tasks[call.call_id], return_exceptions=True)
        assert call.report_state == "failed"
        assert call.report_error == "storage_unavailable"


async def test_successful_recovery_clears_cleanup_block(tmp_path: Path):
    def provider(request):
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        controller = await controller_for(live_config(tmp_path), client)
        call = CallRecord(
            call_id="recovery",
            mode="live",
            status=CallStatus.failed,
            provider_session_id="live_old",
            error_code="cleanup_unconfirmed",
        )
        await controller.store.save(call)
        await controller.recover()
        assert call.error_code != "cleanup_unconfirmed"
        assert call.close_reason == "restart_hangup_requested"
