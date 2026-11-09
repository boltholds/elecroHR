"""Cache XTTS-v2 once inside the existing named TTS Docker volume.

The checkpoint is pinned to an inspected Hugging Face revision. Since model.pth
contains pickle objects, only use this trusted source; do not load arbitrary files.
"""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "coqui/XTTS-v2"
REVISION = "143a5149406459c326c1a41934f52a1395012409"
FILES = (
    "config.json",
    "model.pth",
    "vocab.json",
    "speakers_xtts.pth",
    "dvae.pth",
    "mel_stats.pth",
    "LICENSE.txt",
)


def required_files_present(path: Path) -> bool:
    return all((path / filename).is_file() and (path / filename).stat().st_size > 0
               for filename in FILES)


def main() -> None:
    model_dir = Path(os.environ.get("XTTS_MODEL_DIR", "/models/huggingface/xtts-v2"))
    model_dir.mkdir(parents=True, exist_ok=True)
    if not required_files_present(model_dir):
        snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            allow_patterns=list(FILES),
            local_dir=str(model_dir),
        )
    if not required_files_present(model_dir):
        raise RuntimeError("XTTS-v2 model files are missing or incomplete")
    print("XTTS-v2 weights ready; model license: Coqui Public Model License (non-commercial)")


if __name__ == "__main__":
    main()
