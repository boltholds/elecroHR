import json

import httpx

from app.config import Settings
from app.rag import RAGService
from app.schemas import CallRecord, CallStatus


class Socket:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(json.loads(message))


async def test_native_contract_and_no_audio_format():
    from app.telephony import LiveTelephony

    config = Settings(
        _env_file=None,
        sip_provider_url="sips:example.com:5061",
        sip_username="user",
        sip_password="secret",
        sip_caller_number="+12025550100",
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            201, json={"session": {"id": "live_test"}, "transport": {"type": "sip"}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = LiveTelephony(config, client)
        assert await adapter.create("+12025550123") == "live_test"
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://api.openai.com/v1/live/sessions"
    assert body["transport"]["trunk"]["auth"]["type"] == "digest"
    assert body["session"]["model"] == "gpt-live-1"
    assert "format" not in body["session"]["audio"]
    assert body["session"]["delegation"]["type"] == "responses"


async def test_transcript_dedup_and_tool_loop():
    from app.voice_agent import VoiceAgent

    config = Settings(_env_file=None)
    rag = await RAGService.load(config.knowledge_dir)
    agent = VoiceAgent(config, rag)
    call = CallRecord(call_id="test", mode="live", status=CallStatus.dialing)
    socket = Socket()
    event = {
        "type": "session.input_transcript.delta",
        "event_id": "e1",
        "delta": "Да, согласен.",
        "start_ms": 0,
        "end_ms": 100,
    }
    await agent.handle(event, call, socket)
    await agent.handle(event, call, socket)
    assert len(call.transcript) == 1
    await agent.handle({"type": "transport.answered", "event_id": "e2"}, call, socket)
    assert call.status == CallStatus.active
    tool = {
        "type": "response.event",
        "event_id": "tool1",
        "delegation_id": "d1",
        "event": {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "f1",
                "name": "search_vacancy",
                "arguments": '{"query":"PostgreSQL"}',
            },
        },
    }
    await agent.handle(tool, call, socket)
    await agent.handle({**tool, "event_id": "tool2"}, call, socket)
    await agent.handle(
        {
            "type": "response.event",
            "event_id": "done",
            "delegation_id": "d1",
            "event": {"type": "response.completed", "response": {"id": "r1", "output": []}},
        },
        call,
        socket,
    )
    outputs = [x for x in socket.sent if x["type"] == "response.item.create"]
    assert len(outputs) == 1
    assert "vacancy.md" in outputs[0]["item"]["output"]
    assert socket.sent[-1]["type"] == "response.create"


async def test_close_reason_and_transport_failure():
    from app.voice_agent import VoiceAgent

    config = Settings(_env_file=None)
    agent = VoiceAgent(config, await RAGService.load(config.knowledge_dir))
    call = CallRecord(call_id="test", mode="live", status=CallStatus.active)
    assert await agent.handle(
        {
            "type": "session.closed",
            "event_id": "closed",
            "reason": "connection_lost",
            "usage": {"seconds": 3},
        },
        call,
        Socket(),
    )
    assert call.status == CallStatus.failed
    assert call.finalization_confirmed


async def test_unknown_tool_and_bad_arguments_safe():
    from app.voice_agent import VoiceAgent

    config = Settings(_env_file=None)
    agent = VoiceAgent(config, await RAGService.load(config.knowledge_dir))
    call = CallRecord(call_id="test", mode="live")
    assert await agent.execute_tool("shell", "{}", call) == {"error": "unknown_tool"}
    assert await agent.execute_tool("search_vacancy", "not json", call) == {
        "error": "invalid_arguments"
    }
    assert await agent.execute_tool(
        "record_consent", '{"granted":true,"quote":"invented"}', call
    ) == {"error": "consent_evidence_missing"}


async def test_consent_refusal_discards_transcript_and_cannot_be_reversed():
    from app.schemas import TranscriptFragment
    from app.voice_agent import VoiceAgent

    config = Settings(_env_file=None)
    agent = VoiceAgent(config, await RAGService.load(config.knowledge_dir))
    call = CallRecord(
        call_id="test",
        mode="live",
        transcript=[
            TranscriptFragment(
                event_id="1", role="candidate", text="Нет, не согласен.", start_ms=1, end_ms=2
            )
        ],
    )
    result = await agent.execute_tool(
        "record_consent", json.dumps({"granted": False, "quote": "Нет, не согласен."}), call
    )
    assert result == {"consent": "declined"}
    assert call.transcript == []
    await agent.handle(
        {
            "type": "session.input_transcript.delta",
            "event_id": "late",
            "delta": "private later speech",
            "start_ms": 3,
            "end_ms": 4,
        },
        call,
        Socket(),
    )
    assert call.transcript == []


async def test_rag_failure_returns_safe_tool_error(monkeypatch):
    from app.voice_agent import VoiceAgent

    config = Settings(_env_file=None)
    rag = await RAGService.load(config.knowledge_dir)

    def broken(query):
        raise RuntimeError("internal secret")

    monkeypatch.setattr(rag, "search_vacancy", broken)
    agent = VoiceAgent(config, rag)
    result = await agent.execute_tool(
        "search_vacancy", '{"query":"Python"}', CallRecord(call_id="test", mode="live")
    )
    assert result["error"] == "retrieval_unavailable"
    assert "internal secret" not in str(result)
