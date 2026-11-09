"""Local XTTS-v2 synthesizer with 24kHz PCM16 wire compatibility.

Russian and English are supported by a single Coqui XTTS-v2 instance.
A preset voice is the default; optionally mount a consented reference WAV
under /voices and set XTTS_SPEAKER_WAV to that specific file.
"""
import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

LOG = logging.getLogger(__name__)
SAMPLE_RATE = 24000
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-z]")


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=350)
    language: Literal["auto", "ru", "en"] = "auto"


def detect_language(text: str) -> Literal["ru", "en"]:
    """Prefer Russian for Russian sentences containing Latin technology terms."""
    ru = len(CYRILLIC.findall(text))
    en = len(LATIN.findall(text))
    return "en" if en > ru * 1.6 and en > 0 else "ru"


def select_voice(
    model: Any, speaker: str = "Asya Anara", reference_wav: str = "",
) -> dict[str, str]:
    """Use exactly one voice selector; never silently choose a different voice."""
    if reference_wav:
        path = Path(reference_wav)
        if not path.is_file() or path.suffix.casefold() != ".wav":
            raise ValueError("XTTS reference audio is missing or is not a WAV file")
        return {"speaker_wav": str(path)}
    if not speaker.strip():
        raise ValueError("XTTS_SPEAKER must name an installed voice")
    speakers = getattr(model, "speakers", None)
    if speakers is not None and speaker not in speakers:
        raise ValueError("Configured XTTS speaker is not available")
    return {"speaker": speaker}


def render_pcm(model: Any, text: str, language: str, voice: dict[str, str]) -> bytes:
    """Call coqui-tts once, then convert samples to 24kHz signed PCM16."""
    wav = model.tts(
        text=text, language=language, split_sentences=True, **voice,
    )
    synth = getattr(model, "synthesizer", None)
    rate = getattr(synth, "output_sample_rate", getattr(model, "sr", SAMPLE_RATE))
    if int(rate) != SAMPLE_RATE:
        raise ValueError(f"XTTS output sample rate must be {SAMPLE_RATE}, got {rate}")
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    samples = np.asarray(wav, dtype=np.float32).reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        raise RuntimeError("XTTS returned empty or invalid audio")
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def load_xtts_checkpoint_readonly(model_path: Path) -> dict[str, Any]:
    """Read a trusted legacy XTTS checkpoint without Coqui's rewrite attempt.

    coqui-tts 0.28.0's Xtts._load_checkpoint catches UnpicklingError,
    safely loads the legacy pickle, and then tries torch.save(model.pth)
    to convert its config. The model volume is intentionally read-only.
    We need only the model state dict: avoid the write altogether.
    """
    import torch
    from TTS.tts.models.xtts import _legacy_safe_globals

    with _legacy_safe_globals():
        checkpoint = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=True
        )
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("Invalid XTTS-v2 checkpoint structure")
    return checkpoint["model"]


def load_xtts(model_dir: Path) -> Any:
    """Load the downloaded, pinned checkpoint without a runtime network request."""
    import torch
    from TTS.api import TTS

    for name in ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth"):
        if not (model_dir / name).is_file():
            raise RuntimeError("XTTS weights unavailable: " + name)
    # Only intercept the legacy XTTS checkpoint loader during construction:
    # Coqui's built-in conversion writes model.pth, which is read-only.
    from TTS.tts.models.xtts import Xtts

    original_loader = Xtts._load_checkpoint
    Xtts._load_checkpoint = load_xtts_checkpoint_readonly
    try:
        model = TTS(
            model_path=str(model_dir),
            config_path=str(model_dir / "config.json"),
            progress_bar=False,
        )
    finally:
        Xtts._load_checkpoint = original_loader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    LOG.info("XTTS-v2 model loaded on %s", device)
    return model


def create_app(model_override: Any = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if model_override is None:
            model = await asyncio.to_thread(
                load_xtts, Path(os.environ.get("XTTS_MODEL_DIR", "/models/huggingface/xtts-v2"))
            )
        else:
            model = model_override
        voice = select_voice(
            model,
            os.environ.get("XTTS_SPEAKER", "Asya Anara"),
            os.environ.get("XTTS_SPEAKER_WAV", "").strip(),
        )
        app.state.model = model
        app.state.voice = voice
        app.state.inference_lock = asyncio.Lock()
        LOG.info("XTTS voice mode: %s", "reference" if "speaker_wav" in voice else "preset")
        yield

    application = FastAPI(title="elecroHR XTTS-v2", lifespan=lifespan)

    @application.get("/health/ready")
    async def readiness() -> dict[str, str]:
        return {"status": "ready", "model": "xtts-v2"}

    @application.post("/synthesize")
    async def synthesize(body: SpeechRequest) -> Response:
        language = detect_language(body.text) if body.language == "auto" else body.language
        async with application.state.inference_lock:
            try:
                pcm = await asyncio.to_thread(
                    render_pcm, application.state.model, body.text,
                    language, application.state.voice,
                )
            except Exception:
                LOG.exception("XTTS inference failed")
                raise HTTPException(status_code=503, detail="tts_inference_failed") from None
        return Response(
            content=pcm,
            media_type="application/octet-stream",
            headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-TTS-Language": language},
        )

    return application


app = create_app()
