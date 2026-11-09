"""Download all audio weights once, before running the application.

This script runs in model-init with internet access; the app runs offline.
"""
from pathlib import Path

from faster_whisper.utils import download_model

from local_voice.settings import LocalSettings


def main() -> None:
    settings = LocalSettings()
    settings.whisper_path.mkdir(parents=True, exist_ok=True)
    if not (settings.whisper_path / "model.bin").is_file():
        download_model("small", output_dir=str(settings.whisper_path))
    print("Whisper STT weights ready")


if __name__ == "__main__":
    main()
