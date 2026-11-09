"""Silero TTS loader must retain the model regardless of the result of to()."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_voice.audio import load_tts_model


class DummySilero:
    def __init__(self):
        self.moved_to = ""

    def to(self, device):
        self.moved_to = device
        return None  # The real method can mutate and return None.

    def apply_tts(self, text, speaker, sample_rate):
        return text, speaker, sample_rate


def fake_torch(model):
    class Importer:
        def __init__(self, path):
            assert path == "/models/tts/v5_6_ru.pt"

        def load_pickle(self, package, name):
            assert (package, name) == ("tts_models", "model")
            return model

    return SimpleNamespace(
        package=SimpleNamespace(PackageImporter=Importer),
        device=lambda label: label,
    )


def test_silero_model_is_kept_even_if_to_returns_none():
    expected = DummySilero()
    loaded = load_tts_model(fake_torch(expected), Path("/models/tts/v5_6_ru.pt"))
    assert loaded is expected
    assert loaded.moved_to == "cpu"
    assert loaded.apply_tts("Привет!", "xenia", 24000) == (
        "Привет!", "xenia", 24000
    )


@pytest.mark.parametrize("broken", [None, object()])
def test_silero_loader_rejects_missing_or_invalid_model(broken):
    with pytest.raises(RuntimeError, match="usable model"):
        load_tts_model(fake_torch(broken), Path("/models/tts/v5_6_ru.pt"))
