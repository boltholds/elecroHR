"""Multilingual TTS routes Russian and English to the same model."""
import numpy as np
import pytest

from tts_service.server import detect_language, render_pcm


class FakeMultilingualModel:
    sr = 24000

    def __init__(self):
        self.calls = []

    def generate(self, text, language_id):
        self.calls.append((text, language_id))
        return np.array([[0.0, 0.5, -0.5]], dtype=np.float32)


def test_detects_en_and_ru_with_technical_terms():
    assert detect_language("Hello! How are you today?") == "en"
    assert detect_language("Здравствуйте! Давайте начнём интервью.") == "ru"
    assert detect_language("Расскажи про FastAPI и Docker.") == "ru"


def test_same_tts_model_handles_both_languages():
    model = FakeMultilingualModel()
    russian = render_pcm(model, "Привет!", "ru")
    english = render_pcm(model, "Hello!", "en")
    assert len(russian) == 6
    assert len(english) == 6
    assert model.calls == [("Привет!", "ru"), ("Hello!", "en")]


def test_wrong_tts_sample_rate_rejected():
    model = FakeMultilingualModel()
    model.sr = 22050
    with pytest.raises(ValueError, match="sample rate"):
        render_pcm(model, "Hello!", "en")
