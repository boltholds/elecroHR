"""XTTS-v2 routes English and Russian while retaining the 24kHz PCM protocol."""
from pathlib import Path

import numpy as np
import pytest

from tts_service.server import detect_language, render_pcm, select_voice
from tts_service.prepare_models import FILES, required_files_present


class FakeXTTS:
    sr = 24000
    speakers = ["Asya Anara", "Ana Florence"]

    def __init__(self):
        self.calls = []

    def tts(self, **kwargs):
        self.calls.append(kwargs)
        return np.array([[0.0, 0.5, -0.5]], dtype=np.float32)


def test_detects_ru_en_and_russian_with_technical_terms():
    assert detect_language("Hello! How are you today?") == "en"
    assert detect_language("Здравствуйте! Давайте начнём интервью.") == "ru"
    assert detect_language("Расскажи про FastAPI и Docker.") == "ru"


def test_same_model_handles_ru_en_and_preset_speaker():
    model = FakeXTTS()
    voice = select_voice(model, "Asya Anara")
    russian = render_pcm(model, "Привет!", "ru", voice)
    english = render_pcm(model, "Hello!", "en", voice)
    assert len(russian) == len(english) == 6
    assert [call["language"] for call in model.calls] == ["ru", "en"]
    assert all(call["speaker"] == "Asya Anara" for call in model.calls)
    assert all(call["split_sentences"] is True for call in model.calls)


def test_missing_preset_fails_fast():
    with pytest.raises(ValueError, match="not available"):
        select_voice(FakeXTTS(), "Unknown Person")


def test_reference_wav_overrides_preset(tmp_path: Path):
    sample = tmp_path / "reference.wav"
    sample.write_bytes(b"RIFFtest")
    model = FakeXTTS()
    voice = select_voice(model, "Asya Anara", str(sample))
    assert voice == {"speaker_wav": str(sample)}
    render_pcm(model, "Привет!", "ru", voice)
    assert "speaker" not in model.calls[-1]
    assert model.calls[-1]["speaker_wav"] == str(sample)


def test_missing_reference_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        select_voice(FakeXTTS(), "Asya Anara", "/voices/nonexistent.wav")


def test_wrong_tts_sample_rate_rejected():
    model = FakeXTTS()
    model.sr = 22050
    with pytest.raises(ValueError, match="sample rate"):
        render_pcm(model, "Hello!", "en", {"speaker": "Asya Anara"})


def test_invalid_waveform_is_rejected():
    model = FakeXTTS()
    model.tts = lambda **kwargs: [float("nan")]
    with pytest.raises(RuntimeError, match="invalid audio"):
        render_pcm(model, "Hello!", "en", {"speaker": "Asya Anara"})


def test_download_file_check(tmp_path: Path):
    assert not required_files_present(tmp_path)
    for name in FILES:
        (tmp_path / name).write_bytes(b"x")
    assert required_files_present(tmp_path)
