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



def test_xtts_legacy_loader_is_scoped_and_restored(monkeypatch, tmp_path: Path):
    """The unsafe flag is active only during trusted XTTS construction."""
    import sys
    from types import ModuleType, SimpleNamespace
    from tts_service.server import load_xtts

    module_tts = ModuleType("TTS")
    module_tts.__path__ = []
    module_api = ModuleType("TTS.api")
    module_trainer = ModuleType("trainer")
    module_trainer.__path__ = []
    module_io = ModuleType("trainer.io")
    module_io._WEIGHTS_ONLY = True

    state = {"observed": None, "restored": None, "moved": False}
    class FakeModel:
        def to(self, device):
            assert device == "cpu"
            state["moved"] = True
            return self

    def construct(*, model_path, config_path, progress_bar):
        assert model_path == str(tmp_path)
        assert config_path == str(tmp_path / "config.json")
        state["observed"] = module_io._WEIGHTS_ONLY
        return FakeModel()

    module_api.TTS = construct
    module_trainer.io = module_io
    for name, mod in (
        ("TTS", module_tts), ("TTS.api", module_api),
        ("trainer", module_trainer), ("trainer.io", module_io)
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    for filename in ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth"):
        (tmp_path / filename).write_bytes(b"dummy")

    assert isinstance(load_xtts(tmp_path), FakeModel)
    assert state["observed"] is False
    assert module_io._WEIGHTS_ONLY is True
    assert state["moved"]


def test_xtts_legacy_loader_flag_restored_on_failure(monkeypatch, tmp_path: Path):
    import sys
    from types import ModuleType, SimpleNamespace
    from tts_service.server import load_xtts

    package = ModuleType("TTS")
    package.__path__ = []
    api = ModuleType("TTS.api")
    trainer = ModuleType("trainer")
    trainer.__path__ = []
    trainer_io = ModuleType("trainer.io")
    trainer_io._WEIGHTS_ONLY = True

    def fail(**kwargs):
        assert trainer_io._WEIGHTS_ONLY is False
        raise RuntimeError("simulated load error")

    api.TTS = fail
    trainer.io = trainer_io
    for name, value in (
        ("TTS", package), ("TTS.api", api),
        ("trainer", trainer), ("trainer.io", trainer_io),
    ):
        monkeypatch.setitem(sys.modules, name, value)
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    for name in ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth"):
        (tmp_path / name).write_bytes(b"dummy")
    with pytest.raises(RuntimeError, match="simulated"):
        load_xtts(tmp_path)
    assert trainer_io._WEIGHTS_ONLY is True
