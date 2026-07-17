from __future__ import annotations

import json
import random
import struct
from array import array
from pathlib import Path

from .expert import Expert, ExpertKey


_HEADER = struct.Struct("<4sIII")
_MAGIC = b"COLI"


class ExpertStore:
    """Disk-backed expert store with one independently loadable file per expert."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing expert manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.layers = int(self.manifest["layers"])
        self.experts_per_layer = int(self.manifest["experts_per_layer"])
        self.width = int(self.manifest["width"])

    @classmethod
    def create_demo(
        cls,
        root: Path | str,
        *,
        layers: int = 6,
        experts_per_layer: int = 12,
        width: int = 16,
        seed: int = 7,
    ) -> "ExpertStore":
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format": 1,
            "layers": layers,
            "experts_per_layer": experts_per_layer,
            "width": width,
            "dtype": "float32",
        }
        (root_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        for layer in range(layers):
            layer_path = root_path / f"layer-{layer:03d}"
            layer_path.mkdir(exist_ok=True)
            for expert_id in range(experts_per_layer):
                rng = random.Random(seed + layer * 100_003 + expert_id * 997)
                weights = array(
                    "f", (rng.uniform(-0.18, 0.18) for _ in range(width * width))
                )
                bias = array("f", (rng.uniform(-0.05, 0.05) for _ in range(width)))
                expert_path = layer_path / f"expert-{expert_id:04d}.coli"
                with expert_path.open("wb") as handle:
                    handle.write(_HEADER.pack(_MAGIC, layer, expert_id, width))
                    weights.tofile(handle)
                    bias.tofile(handle)
        return cls(root_path)

    def path_for(self, key: ExpertKey) -> Path:
        self._validate_key(key)
        return self.root / f"layer-{key.layer:03d}" / f"expert-{key.expert:04d}.coli"

    def load(self, key: ExpertKey) -> Expert:
        expert_path = self.path_for(key)
        with expert_path.open("rb") as handle:
            magic, layer, expert_id, width = _HEADER.unpack(handle.read(_HEADER.size))
            if magic != _MAGIC or layer != key.layer or expert_id != key.expert:
                raise ValueError(f"invalid expert file: {expert_path}")
            weights = array("f")
            weights.fromfile(handle, width * width)
            bias = array("f")
            bias.fromfile(handle, width)
        return Expert(key=key, width=width, weights=weights, bias=bias)

    def expert_byte_size(self) -> int:
        return (self.width * self.width + self.width) * array("f").itemsize

    def _validate_key(self, key: ExpertKey) -> None:
        if not 0 <= key.layer < self.layers:
            raise IndexError(f"layer {key.layer} is outside [0, {self.layers})")
        if not 0 <= key.expert < self.experts_per_layer:
            raise IndexError(
                f"expert {key.expert} is outside [0, {self.experts_per_layer})"
            )
