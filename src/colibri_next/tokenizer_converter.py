from __future__ import annotations

import json
import shutil
from pathlib import Path


class TokenizerAssetsConverter:
    """Copies Hugging Face tokenizer and generation assets into a model directory."""

    REQUIRED_FILES = ("tokenizer.json", "tokenizer_config.json")
    OPTIONAL_FILES = ("generation_config.json", "chat_template.jinja")

    def __init__(self, source: Path | str):
        self.source = Path(source)

    def convert(
        self, output: Path | str, *, overwrite: bool = False
    ) -> dict[str, object]:
        output_root = Path(output)
        destination = output_root / "tokenizer"
        destination.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in self.REQUIRED_FILES + self.OPTIONAL_FILES:
            source_path = self.source / name
            if not source_path.exists():
                if name in self.REQUIRED_FILES:
                    raise FileNotFoundError(f"missing tokenizer asset: {source_path}")
                continue
            destination_path = destination / name
            if destination_path.exists() and not overwrite:
                raise FileExistsError(
                    f"tokenizer asset already exists: {destination_path}"
                )
            shutil.copy2(source_path, destination_path)
            copied.append(name)

        storage = {"mode": "huggingface-tokenizer-json", "path": "tokenizer"}
        manifest_path = output_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["tokenizer_storage"] = storage
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        return {**storage, "files": copied}
