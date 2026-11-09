"""Inference adapters: Whisper, Silero ONNX VAD, and Silero package TTS.

Audio transport: signed little-endian PCM16; 16 kHz input, 24 kHz output.
Only one interview is active, so the stateful Silero VAD is never shared.
"""
import numpy as np


class AudioModels:
    def __init__(self, settings: object) -> None:
        import torch
        from faster_whisper import WhisperModel
        from silero_vad import load_silero_vad

        torch.set_num_threads(2)
        self._torch = torch
        self.vad = load_silero_vad(onnx=True)
        self.whisper = WhisperModel(str(settings.whisper_path), device="cpu", compute_type="int8")
        self.tts = torch.package.PackageImporter(str(settings.tts_path)).load_pickle(
            "tts_models", "model"
        ).to(torch.device("cpu"))
        self.speaker = settings.tts_speaker

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
        with self._torch.inference_mode():
            audio = self.tts.apply_tts(text=text, speaker=self.speaker, sample_rate=24000)
        samples = audio.detach().cpu().numpy()
        return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
