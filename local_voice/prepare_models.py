"""Download all audio weights once, before running the application.

This script runs in model-init with internet access; the app runs offline.
"""
from pathlib import Path
from urllib.request import urlretrieve

from faster_whisper.utils import download_model

from local_voice.settings import LocalSettings


def main() -> None:
    settings = LocalSettings()
    settings.whisper_path.mkdir(parents=True, exist_ok=True)
    if not (settings.whisper_path / "model.bin").is_file():
        download_model("small", output_dir=str(settings.whisper_path))
    settings.tts_path.parent.mkdir(parents=True, exist_ok=True)
    if not settings.tts_path.is_file():
        temp = settings.tts_path.with_suffix(".download")
        urlretrieve("https://models.silero.ai/models/tts/ru/v5_6_ru.pt", temp)
        if temp.stat().st_size < 1024 * 1024:
            raise RuntimeError("Downloaded TTS model appears incomplete")
        temp.replace(settings.tts_path)
    print("STT and TTS weights ready")


if __name__ == "__main__":
    main()
