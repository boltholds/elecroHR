"""Audio inference adapters for Whisper and Silero VAD.

TTS uses an isolated XTTS-v2 RU/EN container accessed over HTTP.
Mic input: PCM16/16kHz. Speaker output: PCM16/24kHz.
"""
import numpy as np
import httpx


def validate_tts_response(response: httpx.Response) -> bytes:
    response.raise_for_status()
    if response.headers.get("x-sample-rate") != "24000":
        raise ValueError("TTS returned an unexpected sample rate")
    pcm = response.content
    if not pcm or len(pcm) % 2:
        raise ValueError("TTS returned empty or malformed PCM16")
    return pcm


def split_tts_text(text: str, max_chars: int = 300) -> list[str]:
    """Respect the local TTS request limit while keeping words together."""
    if max_chars < 30:
        raise ValueError("max_chars must allow at least one short sentence")
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))
            continue
        if current and len(current) + 1 + len(word) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        chunks.append(current)
    return chunks


class AudioModels:
    def __init__(self, settings: object) -> None:
        import torch
        from faster_whisper import WhisperModel
        from silero_vad import load_silero_vad

        torch.set_num_threads(2)
        self._torch = torch
        self.vad = load_silero_vad(onnx=True)
        self.whisper = WhisperModel(
            str(settings.whisper_path), device="cpu", compute_type="int8"
        )
        self.tts_url = settings.tts_base_url.rstrip("/") + "/synthesize"
        self._http = httpx.Client(timeout=150)

    def close(self) -> None:
        self._http.close()

    def reset_vad(self) -> None:
        self.vad.reset_states()

    def vad_probability(self, pcm: bytes) -> float:
        frame = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return float(self.vad(self._torch.from_numpy(frame), 16000).item())

    def transcribe(self, pcm: bytes) -> str:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments, _info = self.whisper.transcribe(
            samples, language="ru", beam_size=3, vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def synthesize_timed(self, text: str) -> tuple[bytes, dict[str, float]]:
        """Return PCM and server-side XTTS inference time when available."""
        segments = split_tts_text(text)
        if not segments:
            raise ValueError("Cannot synthesize empty text")
        outputs: list[bytes] = []
        model_ms = 0.0
        saw_inference = False
        for segment in segments:
            response = self._http.post(
                self.tts_url,
                json={"text": segment, "language": "auto"},
            )
            outputs.append(validate_tts_response(response))
            header = response.headers.get("x-tts-model-ms", "")
            if header:
                try:
                    value = float(header)
                except ValueError:
                    continue
                if 0 <= value < 1_000_000 and np.isfinite(value):
                    model_ms += value
                    saw_inference = True
        metrics = {"tts_model_ms": model_ms} if saw_inference else {}
        return b"".join(outputs), metrics

    def synthesize(self, text: str) -> bytes:
        return self.synthesize_timed(text)[0]
