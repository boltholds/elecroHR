"""Standalone local voice FastAPI app. Existing cloud/SIP endpoints are untouched."""
import asyncio
import json
import logging
import re
from time import perf_counter
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag import RAGService
from local_voice.audio import AudioModels
from local_voice.core import EndpointDetector, LocalReport, empty_assessment, make_report
from local_voice.dialogue import GREETING, HRDialogue, Phase
from local_voice.question_bank import QuestionBank
from local_voice.vacancy import VacancyRAG
from local_voice.scoring import score_transcript
from local_voice.replies import (SPEECH_CONTRACT_VERSION, candidate_facing_text,
                                 generate_candidate_reply, safe_reply_fallback)
from local_voice.settings import LocalSettings
from local_voice.metrics import TurnTiming, elapsed_ms, summarize_timings, METRICS_CONTRACT_VERSION

LOG = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
def sentences(text: str) -> tuple[list[str], str]:
    """Cut only full sentences; keep the unfinished tail for the next stream chunk."""
    ready: list[str] = []
    while True:
        match = re.search(r"[.!?](?=\s|$)", text)
        if not match:
            break
        end = match.end()
        sentence = text[:end].strip()
        if sentence:
            ready.append(sentence)
        text = text[end:].lstrip()
    return ready, text


@dataclass
class Interview:
    session_id: str
    settings: LocalSettings
    audio: AudioModels
    rag: RAGService
    http: httpx.AsyncClient
    questions: QuestionBank | None = None
    transcript: list[dict[str, str]] = field(default_factory=list)
    screening_transcript: list[dict[str, str]] = field(default_factory=list)
    dialogue: HRDialogue = field(default_factory=HRDialogue)
    state: str = "created"
    generation: int = 0
    pending_text: str = ""
    reply_task: asyncio.Task[None] | None = None
    ws: WebSocket | None = None
    report: LocalReport | None = None
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _finish_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    detector: EndpointDetector = field(default_factory=EndpointDetector)
    timings: list[TurnTiming] = field(default_factory=list)
    active_timing: TurnTiming | None = None
    scoring_ms: float = 0.0
    scoring_ollama_calls: list[dict[str, str | float | int]] = field(default_factory=list)
    _vad_cpu_ms: float = 0.0
    _vad_frames: int = 0
    _speech_started_at: float = 0.0
    _audio_end_at: float = 0.0

    def begin_timing(self, kind: str = "candidate") -> TurnTiming:
        if self.active_timing is not None and self.active_timing.status == "running":
            self.active_timing.finish("interrupted")
        timing = TurnTiming(
            turn_id=len(self.timings) + 1,
            phase=self.dialogue.phase.value if kind == "candidate" else "greeting",
            kind="greeting" if kind == "greeting" else "candidate",
        )
        self.timings.append(timing)
        self.active_timing = timing
        return timing

    def metrics_public(self) -> dict[str, Any]:
        return {
            "contract": METRICS_CONTRACT_VERSION,
            "summary": summarize_timings(self.timings),
            "turns": [item.public() for item in self.timings],
            "scoring_ms": round(self.scoring_ms, 2),
            "scoring_ollama_calls": [dict(call) for call in self.scoring_ollama_calls],
        }

    async def emit_timing(self, timing: TurnTiming | None = None) -> None:
        target = timing or self.active_timing
        if target is not None:
            await self.send({
                "type": "timing",
                "timing": target.public(),
                "summary": summarize_timings(self.timings),
            })

    async def send(self, payload: dict[str, Any]) -> None:
        if self.ws:
            async with self._send_lock:
                await self.ws.send_json(payload)

    async def send_pcm(self, pcm: bytes) -> None:
        if self.ws:
            async with self._send_lock:
                await self.ws.send_bytes(pcm)

    async def set_state(self, state: str) -> None:
        self.state = state
        await self.send({"type": "state", "state": state})

    async def greet(self) -> None:
        self.begin_timing("greeting")
        if self.questions is not None:
            await self.send({"type": "question_bank", "total": len(self.questions.questions)})
        await self.send({"type": "dialogue", "profile": self.dialogue.public_view()})
        await self.speak_text(GREETING)

    async def speak_text(self, text: str) -> None:
        # Final safety boundary for ALL speech paths, including direct stage
        # turns. No transcript text or audio is emitted until validated.
        try:
            text = candidate_facing_text(text)
        except ValueError:
            LOG.warning("Refused non-candidate-facing text at TTS boundary")
            text = safe_reply_fallback(False)
        self.generation += 1
        ident = self.generation
        self.pending_text = ""
        await self.set_state("speaking")
        await self.send({"type": "audio_start", "generation": ident, "sample_rate": 24000})
        timing = self.active_timing
        try:
            for sentence in [text]:
                tts_started = perf_counter()
                if hasattr(self.audio, "synthesize_timed"):
                    pcm, details = await asyncio.to_thread(
                        self.audio.synthesize_timed, sentence
                    )
                else:
                    pcm = await asyncio.to_thread(self.audio.synthesize, sentence)
                    details = {}
                if timing is not None:
                    timing.add("tts_http_ms", elapsed_ms(tts_started))
                    for name, value in details.items():
                        timing.add(name, value)
                    await self.emit_timing(timing)
                if self.state != "speaking" or ident != self.generation:
                    return
                await self.send({"type": "agent_text", "text": sentence})
                self.pending_text += sentence + " "
                send_started = perf_counter()
                await self.send_pcm(pcm)
                if timing is not None:
                    timing.add("websocket_send_ms", elapsed_ms(send_started))
                    if "first_audio_ms" not in timing.stages_ms:
                        timing.mark("first_audio_ms")
            await self.send({"type": "audio_end", "generation": ident})
            self._audio_end_at = perf_counter()
            if timing is not None:
                timing.mark("backend_ms")
                await self.emit_timing(timing)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("TTS failure")
            await self.send({"type": "error", "code": "tts_unavailable"})
            await self.set_state("listening")

    async def on_audio(self, frame: bytes) -> None:
        if self.state not in {"listening", "speaking"}:
            return
        if len(frame) != 1024:
            await self.send({"type": "error", "code": "invalid_audio_frame"})
            return
        vad_started = perf_counter()
        confidence = self.audio.vad_probability(frame)
        started, completed = self.detector.feed(frame, confidence)
        self._vad_cpu_ms += elapsed_ms(vad_started)
        self._vad_frames += 1
        if started:
            self._speech_started_at = perf_counter()
        if started and self.state == "speaking":
            self.generation += 1
            self.dialogue.playback_interrupted()
            if self.reply_task and not self.reply_task.done():
                self.reply_task.cancel()
            self.pending_text = ""
            if self.active_timing is not None:
                self.active_timing.finish("interrupted")
                await self.emit_timing(self.active_timing)
                self.active_timing = None
            await self.send({"type": "interrupt"})
            await self.set_state("listening")
        if completed and self.state == "listening":
            timing = self.begin_timing()
            timing.add("vad_cpu_ms", self._vad_cpu_ms)
            timing.vad_frames = self._vad_frames
            if self._speech_started_at:
                timing.add("speech_capture_ms", elapsed_ms(self._speech_started_at))
            self._vad_cpu_ms = 0.0
            self._vad_frames = 0
            self._speech_started_at = 0.0
            await self.set_state("thinking")
            await self.emit_timing(timing)
            self.reply_task = asyncio.create_task(self.handle_turn(completed))

    async def on_playback_done(self, generation: int, playback_ms: float = 0.0) -> None:
        if self.state != "speaking" or generation != self.generation:
            return
        utterance = self.pending_text.strip()
        if utterance:
            spoken = {"role": "agent", "text": utterance}
            self.transcript.append(spoken)
            if self.dialogue.phase == Phase.screening:
                self.screening_transcript.append(spoken)
        self.pending_text = ""
        self.dialogue.playback_confirmed()
        if self.active_timing is not None:
            if playback_ms > 0:
                self.active_timing.add("client_playback_ms", playback_ms)
            if self._audio_end_at:
                self.active_timing.add("playback_ack_ms", elapsed_ms(self._audio_end_at))
            self.active_timing.finish()
            await self.emit_timing(self.active_timing)
            self.active_timing = None
        self._audio_end_at = 0.0
        self._vad_cpu_ms = 0.0
        self._vad_frames = 0
        self._speech_started_at = 0.0
        self.detector.reset()
        self.audio.reset_vad()
        await self.set_state("listening")
        if self.dialogue.phase == Phase.finished:
            await self.send({"type": "session_finished"})

    async def handle_turn(self, pcm: bytes) -> None:
        timing = self.active_timing or self.begin_timing()
        try:
            stt_started = perf_counter()
            text = await asyncio.to_thread(self.audio.transcribe, pcm)
            timing.add("stt_ms", elapsed_ms(stt_started))
            await self.emit_timing(timing)
            if not text or self.state != "thinking":
                timing.finish("no_speech")
                await self.emit_timing(timing)
                if self.active_timing is timing:
                    self.active_timing = None
                await self.set_state("listening")
                return
            was_screening = self.dialogue.phase == Phase.screening
            self.transcript.append({"role": "candidate", "text": text})
            if was_screening:
                self.screening_transcript.append({"role": "candidate", "text": text})
            await self.send({"type": "candidate_text", "text": text})
            dialogue_started = perf_counter()
            try:
                plan = await self.dialogue.respond(
                    text, self.transcript, self.http, self.settings, self.rag,
                    self.questions, timings_ms=timing.stages_ms,
                    ollama_calls=timing.ollama_calls,
                )
            finally:
                timing.add("dialogue_ms", elapsed_ms(dialogue_started))
            await self.emit_timing(timing)
            if self.state != "thinking":
                return
            await self.send({"type": "dialogue", "profile": self.dialogue.public_view()})
            if plan.sources:
                await self.send({"type": "rag", "sources": plan.sources})
            if plan.question_id:
                await self.send({"type": "question", "question_id": plan.question_id})
            if plan.text:
                await self.speak_text(plan.text)
                return

            # Never stream raw model text directly into TTS or the transcript:
            # malformed content and model self-narration could become audible.
            llm_started = perf_counter()
            try:
                reply = await generate_candidate_reply(
                    client=self.http,
                    base_url=self.settings.ollama_base_url,
                    model=self.settings.llm_model,
                    prompt=self.dialogue.reply_prompt(plan),
                    transcript=self.transcript,
                    vacancy=plan.response_type == "vacancy",
                    timings_ms=timing.stages_ms,
                    ollama_calls=timing.ollama_calls,
                )
            except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
                LOG.exception("Failed to produce a validated candidate-facing reply")
                reply = safe_reply_fallback(plan.response_type == "vacancy")
            finally:
                timing.add("llm_http_ms", elapsed_ms(llm_started))
            await self.emit_timing(timing)
            if self.state != "thinking":
                return
            await self.speak_text(reply)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("Voice turn failed")
            timing.finish("failed")
            await self.emit_timing(timing)
            if self.active_timing is timing:
                self.active_timing = None
            await self.send({"type": "error", "code": "agent_unavailable"})
            await self.set_state("listening")

    async def synthesize_sentence(self, sentence: str, ident: int) -> None:
        # Legacy helper retained for compatibility, still subject to speech gate.
        if not sentence.strip():
            return
        try:
            sentence = candidate_facing_text(sentence)
        except ValueError:
            LOG.warning("Refused non-candidate-facing legacy TTS sentence")
            sentence = safe_reply_fallback(False)
        pcm = await asyncio.to_thread(self.audio.synthesize, sentence)
        if ident != self.generation or self.state != "speaking":
            return
        self.pending_text += sentence + " "
        await self.send({"type": "agent_text", "text": sentence})
        await self.send_pcm(pcm)

    async def finish(self) -> LocalReport:
        async with self._finish_lock:
            if self.report:
                return self.report
            self.generation += 1
            self.state = "completed"
            if self.active_timing is not None:
                self.active_timing.finish("interrupted")
                self.active_timing = None
            if self.reply_task and not self.reply_task.done():
                self.reply_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.reply_task
            score_started = perf_counter()
            try:
                self.report = await score_transcript(
                    self.session_id, self.screening_transcript, self.settings, self.http,
                    ollama_calls=self.scoring_ollama_calls,
                )
            except Exception:
                LOG.exception("Local scoring failed")
                self.report = make_report(
                    self.session_id,
                    empty_assessment("Автоматическая проверка не удалась; требуется HR."),
                    "insufficient_data",
                )
            finally:
                self.scoring_ms = elapsed_ms(score_started)
            path = self.settings.data_dir / (self.session_id + ".json")
            data = json.dumps(
                {"transcript": self.transcript, "profile": self.dialogue.public_view(),
                 "actions": self.dialogue.actions,
                 "questions_asked": self.dialogue.asked_question_ids,
                 "report": self.report.model_dump(mode="json"),
                 "timings": self.metrics_public()},
                ensure_ascii=False, indent=2,
            )
            def save() -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".tmp")
                temp.write_text(data, encoding="utf-8")
                temp.replace(path)
            await asyncio.to_thread(save)
            return self.report


class Consent(BaseModel):
    consent: bool


def create_local_app(settings: LocalSettings | None = None) -> FastAPI:
    config = settings or LocalSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        audio = await asyncio.to_thread(AudioModels, config)
        rag = await VacancyRAG.load(config.knowledge_dir / "vacancy.md")
        questions = await QuestionBank.load(config.knowledge_dir / "faq.md")
        async with httpx.AsyncClient() as client:
            app.state.audio, app.state.rag, app.state.client = audio, rag, client
            app.state.questions = questions
            app.state.sessions = {}
            app.state.active_id = ""
            app.state.registry_lock = asyncio.Lock()
            try:
                yield
            finally:
                await asyncio.to_thread(audio.close)

    app = FastAPI(title="elecroHR Local Voice", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    async def home() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str | int]:
        return {"status": "ok", "mode": "local",
                "questions": len(app.state.questions.questions),
                "speech_contract": SPEECH_CONTRACT_VERSION,
                "metrics_contract": METRICS_CONTRACT_VERSION}

    @app.post("/sessions", status_code=201)
    async def create_session(payload: Consent) -> dict[str, str]:
        if not payload.consent:
            raise HTTPException(422, "Explicit consent is required")
        async with app.state.registry_lock:
            active_id = app.state.active_id
            if active_id and app.state.sessions[active_id].state != "completed":
                raise HTTPException(409, "One active interview at a time")
            sid = uuid4().hex
            session = Interview(
                sid, config, app.state.audio, app.state.rag, app.state.client,
                questions=app.state.questions,
                detector=EndpointDetector(
                    threshold=config.vad_threshold, silence_frames=config.silence_frames,
                ),
            )
            app.state.sessions[sid] = session
            app.state.active_id = sid
            return {"session_id": sid}

    def get_session(sid: str) -> Interview:
        if sid not in app.state.sessions:
            raise HTTPException(404, "Session not found")
        return app.state.sessions[sid]

    @app.get("/sessions/{sid}")
    async def get_status(sid: str) -> dict[str, Any]:
        session = get_session(sid)
        return {"session_id": sid, "state": session.state,
                "profile": session.dialogue.public_view()}

    @app.post("/sessions/{sid}/finish", response_model=LocalReport)
    async def finish_session(sid: str) -> LocalReport:
        session = get_session(sid)
        report = await session.finish()
        if app.state.active_id == sid:
            app.state.active_id = ""
        return report

    @app.get("/sessions/{sid}/report", response_model=LocalReport)
    async def get_report(sid: str) -> LocalReport:
        session = get_session(sid)
        if not session.report:
            raise HTTPException(409, "Report pending")
        return session.report

    @app.get("/sessions/{sid}/transcript")
    async def get_transcript(sid: str) -> list[dict[str, str]]:
        return get_session(sid).transcript

    @app.get("/sessions/{sid}/metrics")
    async def get_metrics(sid: str) -> dict[str, Any]:
        return get_session(sid).metrics_public()

    @app.websocket("/ws/{sid}")
    async def voice_ws(websocket: WebSocket, sid: str) -> None:
        origin = websocket.headers.get("origin", "")
        if origin not in {"http://localhost:8000", "http://127.0.0.1:8000"}:
            await websocket.close(code=1008)
            return
        session = app.state.sessions.get(sid)
        if not session or session.ws or session.state != "created":
            await websocket.close(code=1008)
            return
        await websocket.accept()
        session.ws = websocket
        session.audio.reset_vad()
        session.reply_task = asyncio.create_task(session.greet())
        try:
            while session.state != "completed":
                incoming = await websocket.receive()
                if incoming["type"] == "websocket.disconnect":
                    break
                raw = incoming.get("bytes")
                if raw is not None:
                    await session.on_audio(raw)
                    continue
                content = incoming.get("text")
                if content:
                    event = json.loads(content)
                    if event.get("type") == "playback_done":
                        await session.on_playback_done(
                            int(event.get("generation", -1)),
                            float(event.get("playback_ms", 0)),
                        )
        except (WebSocketDisconnect, ValueError, json.JSONDecodeError):
            pass
        finally:
            session.ws = None
            if session.state != "completed":
                await session.finish()
            if app.state.active_id == sid:
                app.state.active_id = ""

    return app


app = create_local_app()
