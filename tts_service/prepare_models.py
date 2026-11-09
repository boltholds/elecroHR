"""Download only the Chatterbox multilingual v3 weights, then allow offline inference."""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "ResembleAI/chatterbox"
FILES = (
    "ve.pt",
    "s3gen.pt",
    "t3_mtl23ls_v3.safetensors",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)


def main() -> None:
    root = Path(os.environ.get("HF_HOME", "/models/huggingface"))
    root.mkdir(parents=True, exist_ok=True)
    flag = root / "chatterbox-multilingual-v3.ready"
    if flag.exists():
        print("Chatterbox multilingual weights already cached")
        return
    folder = snapshot_download(
        repo_id=REPO_ID, revision="main", allow_patterns=list(FILES),
    )
    missing = [filename for filename in FILES if not (Path(folder) / filename).exists()]
    if missing:
        raise RuntimeError("Missing multilingual weights: " + ", ".join(missing))
    flag.write_text("ok", encoding="utf-8")
    print("Chatterbox multilingual v3 weights ready")


if __name__ == "__main__":
    main()
