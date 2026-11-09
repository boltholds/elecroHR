import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Protocol

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError
from websockets.asyncio.client import connect

from app.config import Settings
from app.prompts import ConsentArgs, EndArgs, SearchArgs
from app.rag import RAGService
from app.schemas import TERMINAL, CallRecord, CallStatus, TranscriptFragment
from app.telephony import LiveTelephony

WIRE = TypeAdapter(dict[str, JsonValue])


class Sender(Protocol):
    async def send(self, message: str) -> None: ...


class TranscriptEvent(BaseModel):
    event_id: str
    delta: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class FunctionItem(BaseModel):
    call_id: str
    name: str
    arguments: str


def obj(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("Expected event object")
    return value


class VoiceAgent:
    """One per call. SIP owns native audio and barge-in; this owns sideband tools."""

    def __init__(self, config: Settings, rag: RAGService):
        self.config, self.rag = config, rag
        self.seen_events: set[str] = set()
        self.seen_tools: set[str] = set()
        self.pending_continuations: set[str] = set()
        self.finish_at: float = float("inf")

    async def execute_tool(self, name: str, arguments: str, call: CallRecord) -> JsonValue:
        try:
            if name == "search_vacancy":
                args = SearchArgs.model_validate_json(arguments)
                try:
                    chunks = self.rag.search_vacancy(args.query)
                except Exception:
                    return {
                        "error": "retrieval_unavailable",
                        "instruction": "No verified vacancy facts",
                    }
                call.retrievals.extend(chunks)
                return [c.model_dump(mode="json") for c in chunks]
            if name == "record_consent":
                consent = ConsentArgs.model_validate_json(arguments)
                if consent.quote not in call.candidate_text():
                    return {"error": "consent_evidence_missing"}
                if call.consent == "declined":
                    return {"error": "consent_already_declined"}
                call.consent = "granted" if consent.granted else "declined"
                if not consent.granted:
                    call.transcript.clear()
                    self.finish_at = time.monotonic() + self.config.farewell_seconds
                return {"consent": call.consent}
            if name == "end_interview":
                EndArgs.model_validate_json(arguments)
                self.finish_at = time.monotonic() + self.config.farewell_seconds
                return {
                    "status": "ending",
                    "instruction": "Give a brief farewell now if not already said.",
                }
            return {"error": "unknown_tool"}
        except ValidationError:
            return {"error": "invalid_arguments"}

    async def handle(self, event: dict[str, JsonValue], call: CallRecord, socket: Sender) -> bool:
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            if event_id in self.seen_events:
                return False
            self.seen_events.add(event_id)
        kind = event.get("type")
        if kind in {"session.input_transcript.delta", "session.output_transcript.delta"}:
            fragment = TranscriptEvent.model_validate(event)
            if call.consent != "declined":
                call.transcript.append(
                    TranscriptFragment(
                        event_id=fragment.event_id,
                        role="candidate" if kind == "session.input_transcript.delta" else "agent",
                        text=fragment.delta,
                        start_ms=fragment.start_ms,
                        end_ms=fragment.end_ms,
                    )
                )
        elif kind == "transport.ringing" and call.status == CallStatus.dialing:
            call.transition(CallStatus.ringing)
        elif kind == "transport.answered" and call.status in {
            CallStatus.dialing,
            CallStatus.ringing,
        }:
            call.transition(CallStatus.active)
        elif kind == "transport.failed":
            call.error_code = "transport_failed"
            if call.status not in TERMINAL:
                call.transition(CallStatus.failed)
            # Continue receiving final usage; run() bounds the drain time.
            self.finish_at = time.monotonic()
        elif kind == "session.usage.updated":
            call.usage["voice"] = event.get("usage", {})
        elif kind == "session.closed":
            call.finalization_confirmed = True
            reason = event.get("reason", "unknown")
            call.close_reason = reason if isinstance(reason, str) else "unknown"
            call.usage["voice"] = event.get("usage", {})
            if call.status not in TERMINAL:
                success = (
                    call.status == CallStatus.active
                    and call.close_reason in {"remote_hangup", "close_requested"}
                    and not call.error_code
                )
                call.transition(CallStatus.completed if success else CallStatus.failed)
                if not success and not call.error_code:
                    call.error_code = "session_ended_abnormally"
            return True
        elif kind == "response.event":
            nested = obj(event.get("event"))
            delegation = event.get("delegation_id")
            key = delegation if isinstance(delegation, str) else "uncorrelated"
            if nested.get("type") == "response.output_item.done":
                item = obj(nested.get("item"))
                if item.get("type") == "function_call":
                    function = FunctionItem.model_validate(item)
                    if function.call_id not in self.seen_tools:
                        self.seen_tools.add(function.call_id)
                        result = await self.execute_tool(function.name, function.arguments, call)
                        await socket.send(
                            json.dumps(
                                {
                                    "type": "response.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": function.call_id,
                                        "output": json.dumps(result, ensure_ascii=False),
                                    },
                                }
                            )
                        )
                        self.pending_continuations.add(key)
            elif nested.get("type") == "response.completed":
                response = obj(nested.get("response"))
                call.usage["backend"] = response.get("usage", {})
                if key in self.pending_continuations:
                    self.pending_continuations.remove(key)
                    await socket.send(json.dumps({"type": "response.create"}))
            elif nested.get("type") in {"response.failed", "response.incomplete"}:
                raise ValueError("Delegated response failed")
        elif kind == "error":
            # Fail closed on protocol/configuration errors; never expose upstream text.
            raise ValueError("Live provider rejected a command")
        return False

    async def run(
        self,
        call: CallRecord,
        telephony: LiveTelephony,
        cancel: asyncio.Event,
        persist: Callable[[], Awaitable[None]],
    ) -> None:
        started = time.monotonic()
        closing_at = float("inf")
        last_save = 0.0
        async with connect(
            telephony.sideband_url(call.provider_session_id),
            additional_headers=telephony.headers(),
            open_timeout=10,
            close_timeout=5,
            max_size=2**22,
            ping_interval=20,
        ) as socket:
            await persist()
            while True:
                now = time.monotonic()
                ringing = call.status in {CallStatus.dialing, CallStatus.ringing}
                timed_out = now - started > (
                    self.config.ring_timeout_seconds
                    if ringing
                    else self.config.call_timeout_seconds
                )
                if closing_at == float("inf") and (
                    cancel.is_set() or timed_out or now >= self.finish_at
                ):
                    if cancel.is_set() and call.status not in TERMINAL:
                        call.transition(CallStatus.canceled)
                    if timed_out:
                        call.error_code = "ring_timeout" if ringing else "call_timeout"
                    # Keep receiving after hangup until final session.closed.
                    closing_at = now + self.config.close_timeout_seconds
                    await telephony.hangup(call.provider_session_id)
                    call.hangup_confirmed = True
                if now >= closing_at:
                    raise TimeoutError("Finalization timeout")
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=0.5)
                except TimeoutError:
                    continue
                event = WIRE.validate_json(raw)
                # Reflected audio is deliberately not stored; SIP transports it natively.
                if event.get("type") in {
                    "session.input_audio.append",
                    "session.output_audio.delta",
                }:
                    continue
                previous_status = call.status
                previous_consent = call.consent
                done = await self.handle(event, call, socket)
                if (
                    done
                    or call.status != previous_status
                    or call.consent != previous_consent
                    or now - last_save >= 1
                ):
                    await persist()
                    last_save = now
                if done:
                    return
