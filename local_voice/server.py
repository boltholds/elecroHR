"""Standalone local voice FastAPI app. Existing cloud/SIP endpoints are untouched."""
import asyncio
import json
import logging
import re
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
from local_voice.scoring import score_transcript
from local_voice.settings import LocalSettings

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
        await self.send({"type": "dialogue", "profile": self.dialogue.public_view()})
        await self.speak_text(GREETING)

    async def speak_text(self, text: str) -> None:
        self.generation += 1
        ident = self.generation
        self.pending_text = ""
        await self.set_state("speaking")
        await self.send({"type": "audio_start", "generation": ident, "sample_rate": 24000})
        try:
            for sentence in [text]:
                pcm = await asyncio.to_thread(self.audio.synthesize, sentence)
                if self.state != "speaking" or ident != self.generation:
                    return
                await self.send({"type": "agent_text", "text": sentence})
                self.pending_text += sentence + " "
                await self.send_pcm(pcm)
            await self.send({"type": "audio_end", "generation": ident})
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
        confidence = self.audio.vad_probability(frame)
        started, completed = self.detector.feed(frame, confidence)
        if started and self.state == "speaking":
            self.generation += 1
            self.dialogue.playback_interrupted()
            if self.reply_task and not self.reply_task.done():
                self.reply_task.cancel()
            self.pending_text = ""
            await self.send({"type": "interrupt"})
            await self.set_state("listening")
        if completed and self.state == "listening":
            await self.set_state("thinking")
            self.reply_task = asyncio.create_task(self.handle_turn(completed))

    async def on_playback_done(self, generation: int) -> None:
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
        self.detector.reset()
        self.audio.reset_vad()
        await self.set_state("listening")
        if self.dialogue.phase == Phase.finished:
            await self.send({"type": "session_finished"})

    async def handle_turn(self, pcm: bytes) -> None:
        try:
            text = await asyncio.to_thread(self.audio.transcribe, pcm)
            if not text or self.state != "thinking":
                await self.set_state("listening")
                return
            was_screening = self.dialogue.phase == Phase.screening
            self.transcript.append({"role": "candidate", "text": text})
            if was_screening:
                self.screening_transcript.append({"role": "candidate", "text": text})
            await self.send({"type": "candidate_text", "text": text})
            plan = await self.dialogue.respond(
                text, self.transcript, self.http, self.settings, self.rag,
                self.questions,
            )
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

            messages: list[dict[str, str]] = [
                {"role": "system", "content": self.dialogue.reply_prompt(plan)}
            ]
            messages.extend(
                {
                    "role": "user" if turn["role"] == "candidate" else "assistant",
                    "content": turn["text"],
                }
                for turn in self.transcript[-16:]
            )
            self.generation += 1
            ident = self.generation
            self.pending_text = ""
            await self.set_state("speaking")
            await self.send({"type": "audio_start", "generation": ident, "sample_rate": 24000})
            pending = ""
            async with self.http.stream(
                "POST", self.settings.ollama_base_url.rstrip("/") + "/api/chat",
                json={
                    "model": self.settings.llm_model, "messages": messages, "stream": True,
                    "think": False, "options": {"temperature": 0.45, "num_predict": 170},
                },
                timeout=120.0,
            ) as stream:
                stream.raise_for_status()
                async for line in stream.aiter_lines():
                    if ident != self.generation:
                        return
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    pending += event.get("message", {}).get("content", "")
                    ready, pending = sentences(pending)
                    for sentence in ready:
                        await self.synthesize_sentence(sentence, ident)
                    if event.get("done"):
                        break
            if pending.strip():
                await self.synthesize_sentence(pending.strip(), ident)
            if ident == self.generation:
                await self.send({"type": "audio_end", "generation": ident})
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("Voice turn failed")
            await self.send({"type": "error", "code": "agent_unavailable"})
            await self.set_state("listening")

    async def synthesize_sentence(self, sentence: str, ident: int) -> None:
        if not sentence.strip():
            return
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
            if self.reply_task and not self.reply_task.done():
                self.reply_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.reply_task
            try:
                self.report = await score_transcript(
                    self.session_id, self.screening_transcript, self.settings, self.http
                )
            except Exception:
                LOG.exception("Local scoring failed")
                self.report = make_report(
                    self.session_id,
                    empty_assessment("Автоматическая проверка не удалась; требуется HR."),
                    "insufficient_data",
                )
            path = self.settings.data_dir / (self.session_id + ".json")
            data = json.dumps(
                {"transcript": self.transcript, "profile": self.dialogue.public_view(),
                 "actions": self.dialogue.actions,
                 "questions_asked": self.dialogue.asked_question_ids,
                 "report": self.report.model_dump(mode="json")},
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
        rag = await RAGService.load(
            config.knowledge_dir, only_documents=frozenset({"vacancy.md"})
        )
        questions = await QuestionBank.load(config.knowledge_dir / "faq.md")
        async with httpx.AsyncClient() as client:
            app.state.audio, app.state.rag, app.state.client = audio, rag, client
            app.state.questions = questions
            app.state.sessions = {}
            app.state.active_id = ""
            app.state.registry_lock = asyncio.Lock()
            yield

    app = FastAPI(title="elecroHR Local Voice", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    async def home() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "mode": "local"}

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
                        await session.on_playback_done(int(event.get("generation", -1)))
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
