#!/usr/bin/env python3
"""Apply a narrow PyTorch safe-globals fix to the local XTTS service.

Run from the elecroHR repository root:
    python scripts/fix_xtts.py

Optionally specify the service file:
    python scripts/fix_xtts.py path/to/tts_service/server.py

Creates a one-time .py.bak backup before modifying the file. Safe to rerun.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import shutil
import sys

BLOCK = (
    "import torch",
    "from TTS.config.shared_configs import BaseDatasetConfig",
    "from TTS.tts.configs.xtts_config import XttsConfig",
    "from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig",
    "torch.serialization.add_safe_globals([",
    "    XttsConfig, XttsArgs, XttsAudioConfig, BaseDatasetConfig,",
    "])",
)


def patch(source_path: Path) -> str:
    if not source_path.is_file():
        raise FileNotFoundError(f"TTS service file not found: {source_path}")

    original = source_path.read_bytes()
    source = original.decode("utf-8-sig")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source, filename=str(source_path))
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "load_xtts"
    ]
    if len(candidates) != 1:
        raise RuntimeError("Expected one load_xtts() function; file left unchanged")
    func = candidates[0]
    region = "".join(lines[func.lineno - 1 : func.end_lineno])
    if "torch.serialization.add_safe_globals(" in region:
        return "already_patched"

    loads = [
        node
        for node in ast.walk(func)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and (
            (isinstance(node.value.func, ast.Name) and node.value.func.id == "TTS")
            or (isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "TTS")
        )
    ]
    if len(loads) != 1:
        raise RuntimeError("Expected one TTS(...) assignment in load_xtts(); file left unchanged")

    target = loads[0]
    line = lines[target.lineno - 1]
    match = re.match(r"([ \t]*)\S", line)
    if not match:
        raise RuntimeError("Cannot determine load_xtts() indentation")
    indent = match.group(1)
    newline = "\r\n" if b"\r\n" in original else "\n"
    insert = "".join(f"{indent}{item}{newline}" for item in BLOCK)

    patched = (
        "".join(lines[: target.lineno - 1])
        + insert
        + indent + newline
        + "".join(lines[target.lineno - 1 :])
    )
    ast.parse(patched, filename=str(source_path))
    backup_path = source_path.with_name(source_path.name + ".bak")
    if backup_path.exists():
        raise FileExistsError(
            f"Backup already exists: {backup_path}. Move it aside before applying another patch."
        )
    shutil.copy2(source_path, backup_path)
    source_path.write_bytes(
        (b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b"")
        + patched.encode("utf-8")
    )
    return "patched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    repo = Path.cwd()
    candidates = (
        [args.path]
        if args.path is not None
        else [
            repo / "tts_service" / "server.py",
            repo / "services" / "tts_service" / "server.py",
            repo / "app" / "tts_service" / "server.py",
        ]
    )
    path = next((p for p in candidates if p.is_file()), candidates[0])
    try:
        status = patch(path)
    except (FileNotFoundError, RuntimeError, FileExistsError, UnicodeDecodeError, SyntaxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if status == "already_patched":
        print(f"Already patched: {path}")
    else:
        print(f"Patched: {path}")
        print(f"Backup: {path}.bak")
        print("Rebuild: docker compose up -d --build --force-recreate tts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
