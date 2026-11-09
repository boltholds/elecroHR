"""Isolated Chatterbox Multilingual v3 HTTP synthesis service.

Loads one ~500M-parameter multilingual model for Russian and English.
Audio wire format is 24kHz mono signed little-endian PCM16, not a WAV file.
"""
import asyncio
import os
import re
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Literal

SAMPLE_RATE = 24000
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-z]")


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=350)
    language: Literal["auto", "ru", "en"] = "auto"


def detect_language(text: str) -> Literal["ru", "en"]:
    """Use Russian for mixed Russian sentences with Latin technology names."""
    russian = len(CYRILLIC.findall(text))
    english = len(LATIN.findall(text))
    return "en" if english > russian * 1.6 and english > 0 else "ru"


def render_pcm(model: object, text: str, language: str) -> bytes:
    """Call Chatterbox with its language tag; convert float audio to PCM16."""
    wave = model.generate(text, language_id=language)
    output_rate = int(model.sr)
    if output_rate != SAMPLE_RATE:
        raise ValueError(f"Unexpected model sample rate: {output_rate}")
    if hasattr(wave, "detach"):
        wave = wave.detach().cpu().numpy()
    samples = np.asarray(wave, dtype=np.float32).reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        raise RuntimeError("TTS produced empty or nonfinite audio")
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def create_app(model_override: object = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if model_override is None:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = await asyncio.to_thread(
                ChatterboxMultilingualTTS.from_pretrained,
                device=device,
                t3_model="v3",
            )
        else:
            model = model_override
        app.state.model = model
        app.state.inference_lock = asyncio.Lock()
        yield

    application = FastAPI(title="elecroHR multilingual TTS", lifespan=lifespan)

    @application.get("/health/ready")
    async def readiness() -> dict[str, str]:
        return {"status": "ready", "model": "chatterbox-multilingual-v3"}

    @application.post("/synthesize")
    async def synthesize(body: SpeechRequest) -> Response:
        language = detect_language(body.text) if body.language == "auto" else body.language
        async with application.state.inference_lock:
            try:
                pcm = await asyncio.to_thread(
                    render_pcm, application.state.model, body.text, language
                )
            except Exception:
                # Detailed exceptions stay in server logs; clients get normalized codes.
                raise HTTPException(status_code=503, detail="tts_inference_failed") from None
        return Response(
            content=pcm,
            media_type="application/octet-stream",
            headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-TTS-Language": language},
        )

    return application


app = create_app()
