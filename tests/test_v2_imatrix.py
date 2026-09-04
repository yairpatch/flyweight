"""Importance-matrix weighted IQ packing, and native capture of the matrix."""

from __future__ import annotations

import math
import os
import struct
import tempfile
import unittest
from pathlib import Path

from flyweight.cli import main as cli_main
from flyweight.v2 import V2Error, V2Model
from tests import hf_safetensors_fixture as fixture
from tests.dense_gguf_fixture import build_dense_qwen35_gguf


def write_imatrix(path: Path, entries: dict[str, list[float]]) -> None:
    """The legacy llama.cpp imatrix.dat layout, without the optional trailer."""
    blob = struct.pack("<i", len(entries))
    for name, values in entries.items():
        encoded = name.encode()
        blob += struct.pack("<i", len(encoded)) + encoded
        blob += struct.pack("<ii", 1, len(values))
        blob += struct.pack(f"<{len(values)}f", *values)
    path.write_bytes(blob)


def arena(cache_file: Path) -> bytes:
    """The packed arena, past the header and descriptor table.

    Mirrors hf::cache::Header: magic[8], format u32, byte order u32, then five
    u64 fields of which arena_offset is the fourth -- byte 40.
    """
    raw = cache_file.read_bytes()
    (offset,) = struct.unpack_from("<Q", raw, 40)
    return raw[offset:]


class ImatrixPackingTests(unittest.TestCase):
    """Each test builds its own checkpoint: an imatrix beside the shards is
    picked up automatically, so a shared fixture would leak between tests."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        root = Path(self._directory.name)
        self.checkpoint = fixture.build(root / "imatrix-fixture")
        self.cache = root / "cache"
        self.cache.mkdir()
        self._saved = {
            name: os.environ.get(name)
            for name in ("FLYWEIGHT_HF_QUANT", "FLYWEIGHT_HF_CACHE",
                         "FLYWEIGHT_HF_IMATRIX")
        }
        self.addCleanup(self._restore)
        os.environ["FLYWEIGHT_HF_QUANT"] = "IQ3_XXS"
        os.environ["FLYWEIGHT_HF_CACHE"] = str(self.cache)
        os.environ.pop("FLYWEIGHT_HF_IMATRIX", None)

    def _restore(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def open_once(self) -> None:
        with V2Model(self.checkpoint):
            pass

    def cache_files(self) -> list[Path]:
        return sorted(self.cache.glob("flyweight-*.cache"))

    def spiky_matrix(self) -> dict[str, list[float]]:
        # Strongly nonuniform importance: one hot channel per eight flips the
        # codebook search away from the unweighted choice across most blocks.
        spiky = [1000.0 if i % 8 == 0 else 0.01 for i in range(fixture.HIDDEN)]
        return {
            "blk.0.ffn_gate.weight": spiky,
            "blk.0.ffn_up.weight": spiky,
            # A stacked expert tensor, weighted per expert: row * EXPERTS.
            "blk.1.ffn_gate_exps.weight": spiky * fixture.EXPERTS,
        }

    def test_an_imatrix_changes_what_the_pack_emits(self) -> None:
        self.open_once()
        baseline = self.cache_files()
        self.assertEqual(len(baseline), 1)

        write_imatrix(self.checkpoint / "imatrix.dat", self.spiky_matrix())
        self.open_once()
        weighted = [f for f in self.cache_files() if f not in baseline]
        # A different fingerprint: the matrix is part of the cache identity.
        self.assertEqual(len(weighted), 1)
        # And different bytes: the search actually weighed it.
        self.assertNotEqual(arena(baseline[0]), arena(weighted[0]))
        self.assertEqual(len(arena(baseline[0])), len(arena(weighted[0])))

    def test_iq4_xs_reads_the_matrix_too(self) -> None:
        # The second target whose search is importance-weighted; the same
        # spiky matrix must move its bytes the same way.
        os.environ["FLYWEIGHT_HF_QUANT"] = "IQ4_XS"
        self.open_once()
        baseline = self.cache_files()
        with V2Model(self.checkpoint) as model:
            types = {int(t["ggml_type"]) for t in model.tensors()}
        self.assertIn(23, types)
        write_imatrix(self.checkpoint / "imatrix.dat", self.spiky_matrix())
        self.open_once()
        weighted = [f for f in self.cache_files() if f not in baseline]
        self.assertEqual(len(weighted), 1)
        self.assertNotEqual(arena(baseline[0]), arena(weighted[0]))
        self.assertEqual(len(arena(baseline[0])), len(arena(weighted[0])))

    def test_iq2_xs_requires_the_matrix(self) -> None:
        # The 2.31-bit codebook has nothing to say about which channels can
        # afford to be wrong without calibration data; refusing beats packing
        # a worse-than-Q2_K checkpoint silently. The menu says so too.
        os.environ["FLYWEIGHT_HF_QUANT"] = "IQ2_XS"
        with self.assertRaisesRegex(V2Error, "importance matrix"):
            self.open_once()
        options = {
            str(option["name"]): str(option["unavailable"])
            for option in V2Model.hf_quant_options(self.checkpoint)
        }
        self.assertIn("importance matrix", options["IQ2_XS"])

        write_imatrix(self.checkpoint / "imatrix.dat", self.spiky_matrix())
        options = {
            str(option["name"]): str(option["unavailable"])
            for option in V2Model.hf_quant_options(self.checkpoint)
        }
        self.assertEqual(options["IQ2_XS"], "")
        self.open_once()
        with V2Model(self.checkpoint) as model:
            types = {int(t["ggml_type"]) for t in model.tensors()}
        self.assertIn(17, types)

    def test_off_restores_the_unweighted_fingerprint(self) -> None:
        self.open_once()
        write_imatrix(self.checkpoint / "imatrix.dat", self.spiky_matrix())
        os.environ["FLYWEIGHT_HF_IMATRIX"] = "off"
        self.open_once()
        # The second open hit the first cache rather than packing a new one.
        self.assertEqual(len(self.cache_files()), 1)

    def test_a_mismatched_geometry_packs_unweighted(self) -> None:
        self.open_once()
        baseline = self.cache_files()
        # Entry sizes that match no tensor: a matrix for another checkpoint.
        write_imatrix(self.checkpoint / "imatrix.dat",
                      {"blk.0.ffn_gate.weight": [1000.0] * 7})
        self.open_once()
        weighted = [f for f in self.cache_files() if f not in baseline]
        # New fingerprint (the file is present), same bytes (nothing matched).
        self.assertEqual(len(weighted), 1)
        self.assertEqual(arena(baseline[0]), arena(weighted[0]))

    def test_targets_that_read_no_matrix_keep_their_fingerprint(self) -> None:
        os.environ["FLYWEIGHT_HF_QUANT"] = "Q6_K"
        self.open_once()
        write_imatrix(self.checkpoint / "imatrix.dat", self.spiky_matrix())
        self.open_once()
        # Q6_K packing never reads the matrix, so its cache must not go stale
        # over a file that changes nothing.
        self.assertEqual(len(self.cache_files()), 1)

    def test_a_configured_missing_path_is_an_error(self) -> None:
        os.environ["FLYWEIGHT_HF_IMATRIX"] = str(
            Path(self._directory.name) / "absent.dat")
        with self.assertRaises(V2Error):
            self.open_once()

    def test_a_gguf_imatrix_is_refused_with_direction(self) -> None:
        (self.checkpoint / "imatrix.dat").write_bytes(
            b"GGUF" + b"\x00" * 64)
        with self.assertRaises(V2Error) as caught:
            self.open_once()
        self.assertIn("legacy", str(caught.exception))


def read_imatrix(path: Path) -> dict[str, tuple[int, list[float]]]:
    """Parse the legacy .dat layout back: name -> (ncall, values)."""
    raw = path.read_bytes()
    cursor = 0

    def read_i32() -> int:
        nonlocal cursor
        (value,) = struct.unpack_from("<i", raw, cursor)
        cursor += 4
        return value

    entries: dict[str, tuple[int, list[float]]] = {}
    for _ in range(read_i32()):
        length = read_i32()
        name = raw[cursor:cursor + length].decode()
        cursor += length
        calls = read_i32()
        count = read_i32()
        values = list(struct.unpack_from(f"<{count}f", raw, cursor))
        cursor += count * 4
        entries[name] = (calls, values)
    return entries


class ImatrixCollectionTests(unittest.TestCase):
    """The `imatrix` subcommand end to end, on the CPU backend."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self._saved = {
            name: os.environ.get(name)
            for name in ("FLYWEIGHT_IMATRIX", "FLYWEIGHT_PREFILL_EXPERT_STREAM_MIB")
        }
        self.addCleanup(self._restore)
        V2Model.select_backend("cpu")
        self.addCleanup(V2Model.select_backend, "auto")

    def _restore(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_the_cli_gathers_a_matrix_the_loader_reads_back(self) -> None:
        model = self.root / "dense.gguf"
        build_dense_qwen35_gguf(model, quantize="q8_0")
        calibration = self.root / "calibration.txt"
        calibration.write_text(
            "The importance of a channel is how hard the model drives it.\n"
            * 12
        )
        output = self.root / "imatrix.dat"

        code = cli_main([
            "imatrix", str(model),
            "--text", str(calibration),
            "--output", str(output),
            "--chunk", "48", "--context", "128",
        ])

        self.assertEqual(code, 0)
        entries = read_imatrix(output)
        self.assertTrue(entries)
        # Every projection the forward ran through is present under its GGUF
        # name, with one value per input channel. This fixture fuses QKV.
        self.assertIn("blk.0.attn_qkv.weight", entries)
        calls, values = entries["blk.0.attn_qkv.weight"]
        self.assertGreater(calls, 0)
        self.assertEqual(len(values), 256)
        for name, (_, sums) in entries.items():
            self.assertTrue(
                all(math.isfinite(value) and value >= 0.0 for value in sums),
                name,
            )
        # Activation energy is nonzero somewhere; an all-zero matrix would
        # mean the hooks ran but read the wrong buffers.
        self.assertTrue(any(any(sums) for _, (_, sums) in entries.items()))


if __name__ == "__main__":
    unittest.main()
