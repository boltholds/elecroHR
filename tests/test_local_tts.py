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


async def test_xtts_http_api_remains_compatible(monkeypatch):
    import httpx

    from tts_service.server import create_app

    monkeypatch.setenv("XTTS_SPEAKER", "Asya Anara")
    monkeypatch.delenv("XTTS_SPEAKER_WAV", raising=False)
    model = FakeXTTS()
    app = create_app(model_override=model)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://tts:8011",
        ) as client:
            ready = await client.get("/health/ready")
            assert ready.status_code == 200
            assert ready.json()["model"] == "xtts-v2"

            response = await client.post(
                "/synthesize", json={"text": "Здравствуйте!", "language": "auto"}
            )
            assert response.status_code == 200
            assert response.content
            assert response.headers["x-sample-rate"] == "24000"
            assert response.headers["x-tts-language"] == "ru"
            assert model.calls[-1]["speaker"] == "Asya Anara"



def test_xtts_checkpoint_load_has_scoped_safe_globals(monkeypatch, tmp_path: Path):
    """Exercise the loader without GPU or unpickling a third-party checkpoint."""
    import sys
    from contextlib import contextmanager
    from types import ModuleType, SimpleNamespace

    from tts_service.server import load_xtts

    classes = {}
    for label in ("XttsConfig", "XttsArgs", "XttsAudioConfig", "BaseDatasetConfig"):
        classes[label] = type(label, (), {})

    packages = (
        "TTS", "TTS.api", "TTS.tts", "TTS.tts.models",
        "TTS.tts.models.xtts", "TTS.tts.configs",
        "TTS.tts.configs.xtts_config", "TTS.config",
        "TTS.config.shared_configs",
    )
    modules = {}
    for package in packages:
        mod = ModuleType(package)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, package, mod)
        modules[package] = mod
    modules["TTS.tts.configs.xtts_config"].XttsConfig = classes["XttsConfig"]
    modules["TTS.tts.models.xtts"].XttsArgs = classes["XttsArgs"]
    modules["TTS.tts.models.xtts"].XttsAudioConfig = classes["XttsAudioConfig"]
    modules["TTS.config.shared_configs"].BaseDatasetConfig = classes["BaseDatasetConfig"]

    state = {"inside": False, "closed": False}

    @contextmanager
    def safe_globals(whitelist):
        assert set(whitelist) == set(classes.values())
        state["inside"] = True
        try:
            yield
        finally:
            state["inside"] = False
            state["closed"] = True

    class MockModel:
        def to(self, device):
            assert device == "cpu"
            return self

    def make_tts(*, model_path, config_path, progress_bar):
        assert state["inside"]
        assert model_path == str(tmp_path)
        assert config_path == str(tmp_path / "config.json")
        assert progress_bar is False
        return MockModel()

    modules["TTS.api"].TTS = make_tts
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.serialization = SimpleNamespace(safe_globals=safe_globals)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    for name in ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth"):
        (tmp_path / name).write_bytes(b"mock")

    loaded = load_xtts(tmp_path)
    assert isinstance(loaded, MockModel)
    assert state == {"inside": False, "closed": True}
