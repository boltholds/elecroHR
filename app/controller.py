import asyncio
import json
import logging
from uuid import uuid4

import httpx

from app.call_store import CallStore
from app.config import Settings
from app.demo import run_demo
from app.rag import RAGService
from app.schemas import TERMINAL, CallRecord, CallRequest, CallStatus, LiveRequest
from app.scoring import ScoringService
from app.telephony import LiveTelephony
from app.voice_agent import VoiceAgent

logger = logging.getLogger("hr_caller")


class BusyError(Exception):
    pass


class CallController:
    def __init__(
        self,
        config: Settings,
        store: CallStore,
        rag: RAGService,
        scoring: ScoringService,
        client: httpx.AsyncClient,
    ):
        self.config, self.store, self.rag, self.scoring = config, store, rag, scoring
        self.telephony = LiveTelephony(config, client)
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.cancellations: dict[str, asyncio.Event] = {}
        self.lock = asyncio.Lock()
        self.stopping = False

    async def recover(self) -> None:
        for call in self.store.calls.values():
            if (
                call.mode == "live"
                and call.provider_session_id
                and not call.finalization_confirmed
                and not call.hangup_confirmed
            ):
                try:
                    await self.telephony.hangup(call.provider_session_id)
                    call.close_reason = "restart_hangup_requested"
                    call.hangup_confirmed = True
                    if call.error_code == "cleanup_unconfirmed":
                        call.error_code = "recovered_after_cleanup_failure"
                except Exception:
                    call.error_code = "cleanup_unconfirmed"
                await self.store.save(call)

    async def start(self, request: CallRequest) -> CallRecord:
        if isinstance(request, LiveRequest):
            self.config.validate_live(request.destination)
            if any(
                c.error_code in {"dialing_outcome_unknown", "cleanup_unconfirmed"}
                for c in self.store.calls.values()
            ):
                raise BusyError(
                    "Previous live outcome unknown; reconcile the call with SIP provider first"
                )
        async with self.lock:
            if self.stopping or any(not task.done() for task in self.tasks.values()):
                raise BusyError("One call or report is already in progress")
            self.tasks.clear()
            self.cancellations.clear()
            call = CallRecord(
                call_id=str(uuid4()),
                mode=request.mode,
                consent="pending" if self.config.require_consent else "not_required",
            )
            await self.store.save(call)
            cancel = asyncio.Event()
            self.cancellations[call.call_id] = cancel
            destination = request.destination if isinstance(request, LiveRequest) else ""
            self.tasks[call.call_id] = asyncio.create_task(
                self._supervise(call, destination, cancel)
            )
            return call

    async def _supervise(self, call: CallRecord, destination: str, cancel: asyncio.Event) -> None:
        try:
            await self._run(call, destination, cancel)
        except (Exception, asyncio.CancelledError) as error:
            if call.status not in TERMINAL:
                call.transition(CallStatus.failed)
            call.report_state = "failed"
            call.report_error = (
                "storage_unavailable" if isinstance(error, OSError) else "worker_interrupted"
            )
            logger.error(
                json.dumps(
                    {
                        "event": "worker_failed",
                        "call_id": call.call_id,
                        "error_code": call.report_error,
                        "error_type": type(error).__name__,
                    }
                )
            )
            # In-memory API status remains truthful even when durable storage is unavailable.
            try:
                await self.store.save(call)
            except OSError:
                pass

    async def _run(self, call: CallRecord, destination: str, cancel: asyncio.Event) -> None:
        try:
            if cancel.is_set():
                call.transition(CallStatus.canceled)
            else:
                call.transition(CallStatus.dialing)
                await self.store.save(call)
                if call.mode == "demo":
                    call.transition(CallStatus.ringing)
                    call.transition(CallStatus.active)
                    run_demo(call, self.rag)
                    call.transition(CallStatus.completed)
                else:
                    # Mark before the request: if the process dies, creation may have succeeded.
                    call.error_code = "dialing_outcome_unknown"
                    await self.store.save(call)
                    call.provider_session_id = await self.telephony.create(destination)
                    call.error_code = ""
                    agent = VoiceAgent(self.config, self.rag)
                    await agent.run(call, self.telephony, cancel, lambda: self.store.save(call))
        except asyncio.CancelledError:
            if call.status not in TERMINAL:
                call.transition(CallStatus.canceled)
            call.error_code = call.error_code or "shutdown_interrupted"
            raise
        except Exception as error:
            if call.status not in TERMINAL:
                call.transition(CallStatus.failed)
            if isinstance(error, httpx.HTTPStatusError) and not call.provider_session_id:
                # 4xx is a definite rejection. 5xx may have an ambiguous side effect.
                if error.response.status_code < 500:
                    call.error_code = "provider_rejected_call"
            call.error_code = call.error_code or "call_failed"
            logger.warning(
                json.dumps(
                    {
                        "event": "call_failed",
                        "call_id": call.call_id,
                        "error_code": call.error_code,
                        "error_type": type(error).__name__,
                    }
                )
            )
        finally:
            if call.mode == "live" and call.provider_session_id and not call.finalization_confirmed:
                try:
                    await self.telephony.hangup(call.provider_session_id)
                    call.hangup_confirmed = True
                except Exception:
                    call.error_code = "cleanup_unconfirmed"
            if call.consent not in {"granted", "not_required"}:
                call.transcript.clear()
            await self.store.save(call)
        try:
            report = await self.scoring.score(call)
            await self.store.save_report(report)
            call.report_state = "ready"
        except Exception as error:
            call.report_state = "failed"
            call.report_error = (
                "storage_unavailable" if isinstance(error, OSError) else "scoring_failed"
            )
            logger.warning(
                json.dumps(
                    {
                        "event": "scoring_failed",
                        "call_id": call.call_id,
                        "error_type": type(error).__name__,
                    }
                )
            )
        await self.store.save(call)

    def cancel(self, call_id: str) -> None:
        call = self.store.get(call_id)
        if call.status in TERMINAL:
            raise BusyError("Call has already ended")
        self.cancellations[call_id].set()

    async def shutdown(self) -> None:
        self.stopping = True
        for cancel in self.cancellations.values():
            cancel.set()
        pending = [task for task in self.tasks.values() if not task.done()]
        if not pending:
            return
        timeout = (
            self.config.http_timeout_seconds
            + 2 * self.config.close_timeout_seconds
            + self.config.scoring_timeout_seconds
            + 5
        )
        _, unfinished = await asyncio.wait(pending, timeout=timeout)
        for task in unfinished:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
