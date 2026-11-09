"""Audio inference adapters for Whisper and Silero VAD.

TTS is now a separate Chatterbox multilingual container accessed over HTTP.
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

    def synthesize(self, text: str) -> bytes:
        response = self._http.post(
            self.tts_url,
            json={"text": text, "language": "auto"},
        )
        return validate_tts_response(response)
