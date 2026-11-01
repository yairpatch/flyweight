"""Get a model: fitting, the catalog, Hugging Face lookups and the downloader,
against a local stand-in for the Hub (tests/hf_stub.py)."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from flyweight import models
from tests.hf_stub import HubStub, StubFile

GB = 2**30


def wait_for(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")


class FitTests(unittest.TestCase):
    def test_levels_follow_the_hardware(self) -> None:
        gpu = models.Machine(ram=32 * GB, vram=12 * GB, gpu="RTX", disk_free=500 * GB)
        cpu = models.Machine(ram=16 * GB, vram=0, gpu="", disk_free=500 * GB)
        self.assertEqual(models.fit(5 * GB, False, gpu)["level"], "great")
        self.assertEqual(models.fit(9 * GB, False, gpu)["level"], "good")     # mostly on the GPU
        self.assertEqual(models.fit(18.6 * GB, True, gpu)["level"], "good")   # MoE: experts in RAM
        self.assertEqual(models.fit(20 * GB, False, gpu)["level"], "slow")
        self.assertEqual(models.fit(40 * GB, True, gpu)["level"], "no")
        self.assertEqual(models.fit(2.5 * GB, False, cpu)["level"], "good")
        self.assertEqual(models.fit(5 * GB, False, cpu)["level"], "slow")
        self.assertEqual(models.fit(30 * GB, False, cpu)["level"], "no")
        full = models.Machine(ram=64 * GB, vram=24 * GB, gpu="RTX", disk_free=3 * GB)
        self.assertEqual(models.fit(5 * GB, False, full)["label"], "Not enough disk space")


class NamingTests(unittest.TestCase):
    def test_quantization_and_architecture(self) -> None:
        self.assertEqual(models.quant_of("Qwen3-8B-Q4_K_M.gguf"), "Q4_K_M")
        self.assertEqual(models.quant_of("Qwen3-30B-A3B-UD-Q4_K_XL.gguf"), "UD-Q4_K_XL")
        self.assertEqual(models.quant_of("m-IQ4_XS-00001-of-00002.gguf"), "IQ4_XS")
        for name, verdict in [("qwen3", "yes"), ("qwen3moe", "yes"), ("gemma4", "yes"),
                              ("lfm2", "yes"), ("qwen2", "no"), ("qwenimage21-dit", "no"),
                              ("llama", "no"), ("", "unknown")]:
            self.assertEqual(models.architecture_support(name), verdict, name)

    def test_variants_group_shards_and_drop_incomplete_sets(self) -> None:
        files = [models.HubFile(p, 10, "", models.quant_of(p)) for p in (
            "M-Q4_K_M-00001-of-00002.gguf", "M-Q4_K_M-00002-of-00002.gguf",
            "M-Q8_0-00001-of-00002.gguf",                     # its part 2 is missing
            "M-Q6_K.gguf", "mmproj-F16.gguf", "mmproj-Q8_0.gguf")]
        found, projectors = models.variants(files)
        self.assertEqual([(v.quant, len(v.files), v.size) for v in found],
                         [("Q6_K", 1, 10), ("Q4_K_M", 2, 20)])
        self.assertEqual(models.pick_projector(projectors).path, "mmproj-F16.gguf")
        self.assertEqual(models.pick_variant(found, ["Q5_K_M", "Q4_K_M"]).quant, "Q4_K_M")
        self.assertEqual(models.pick_variant(found, ["Q2_K"]).quant, "Q4_K_M")  # the Q4 fallback


class CatalogTests(unittest.TestCase):
    def test_the_bundled_catalog_is_complete(self) -> None:
        entries = models.load_catalog(fetch=False)
        self.assertGreater(len(entries), 3)
        for entry in entries:
            for key in ("id", "name", "repo", "quants", "description", "rank"):
                self.assertIn(key, entry, entry.get("id"))
            self.assertRegex(entry["repo"], r"^[\w.-]+/[\w.-]+$")
        self.assertEqual(len({e["id"] for e in entries}), len(entries))

    def test_a_cached_copy_wins_over_the_bundled_one_and_bad_files_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "catalog.json"
            cache.write_text(json.dumps({"models": [{"repo": "a/b", "name": "Cached"}]}))
            self.assertEqual(models.load_catalog(cache, fetch=False)[0]["name"], "Cached")
            cache.write_text('{"models": "nope"}')
            self.assertNotEqual(models.load_catalog(cache, fetch=False)[0]["name"], "Cached")


class HubTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.hub = HubStub()
        self.data = b"GGUF" + os.urandom(3 * (1 << 20) + 123)
        self.hub.add("org/Small-GGUF", [StubFile("Small-Q8_0.gguf", self.data),
                                         StubFile("Small-Q4_K_M.gguf", b"GGUF" + b"q" * 100),
                                         StubFile("mmproj-Small-F16.gguf", b"GGUF" + b"v" * 50)])
        self.hub.add("org/Gated-GGUF", [StubFile("G-Q4_K_M.gguf", b"GGUF")], gated=True)
        self.hub.add("org/Old-GGUF", [StubFile("O-Q4_K_M.gguf", b"GGUF")], architecture="qwen2")
        self.hub.start()
        self.env = patch.dict(os.environ, {"HF_ENDPOINT": self.hub.url})
        self.env.start()
        self.machine = models.Machine(ram=32 * GB, vram=12 * GB, gpu="RTX", disk_free=500 * GB)

    def tearDown(self) -> None:
        self.env.stop()
        self.hub.stop()
        self._dir.cleanup()

    def test_search_and_repository_details(self) -> None:
        results = {r["repo"]: r for r in models.search("gguf")}
        self.assertEqual(results["org/Small-GGUF"]["supported"], "yes")
        self.assertEqual(results["org/Old-GGUF"]["supported"], "no")
        self.assertTrue(results["org/Gated-GGUF"]["gated"])
        details = models.repo_details("org/Small-GGUF", self.machine)
        self.assertEqual([v["quant"] for v in details["variants"]], ["Q4_K_M", "Q8_0"])
        self.assertEqual(details["projector"]["path"], "mmproj-Small-F16.gguf")
        with self.assertRaises(models.HubError):
            models.repo_details("org/Missing", self.machine)

    def test_a_catalog_entry_resolves_its_real_file(self) -> None:
        entry = {"id": "s", "name": "Small", "repo": "org/Small-GGUF", "quants": ["Q8_0"],
                 "vision": True}
        resolved = models.resolve_entry(entry, self.machine)
        self.assertTrue(resolved["available"])
        self.assertEqual([f["path"] for f in resolved["files"]],
                         ["Small-Q8_0.gguf", "mmproj-Small-F16.gguf"])
        self.assertEqual(resolved["fit"]["level"], "great")
        missing = models.resolve_entry({"id": "m", "name": "M", "repo": "org/Missing"}, self.machine)
        self.assertFalse(missing["available"])
        self.assertIn("not found", missing["error"])

    def downloads(self) -> tuple[models.Downloads, list]:
        finished: list = []
        return models.Downloads(lambda: self.root, finished.append), finished

    def files(self, repo: str = "org/Small-GGUF") -> list[dict]:
        return models.resolve_entry({"id": "x", "name": "x", "repo": repo, "quants": ["Q8_0"]},
                                    self.machine)["files"]

    def test_download_verifies_and_lands_in_the_library(self) -> None:
        queue, finished = self.downloads()
        job = queue.add("org/Small-GGUF", "Small", self.files())
        wait_for(lambda: finished)
        target = self.root / "org__Small-GGUF" / "Small-Q8_0.gguf"
        self.assertEqual(target.read_bytes(), self.data)
        self.assertEqual(finished[0].result, str(target))
        self.assertEqual(job.done, job.total)
        library = models.library(self.root)
        self.assertEqual([(m["name"], m["repo"]) for m in library], [("Small", "org/Small-GGUF")])
        models.delete_model(self.root, library[0]["folder"])
        self.assertEqual(models.library(self.root), [])
        with self.assertRaises(ValueError):
            models.delete_model(self.root, str(self.root.parent))

    def test_pause_then_resume_continues_where_it_stopped(self) -> None:
        self.hub.delay = 0.3
        queue, finished = self.downloads()
        job = queue.add("org/Small-GGUF", "Small", self.files())
        wait_for(lambda: job.done > 0)
        queue.pause(job.id)
        wait_for(lambda: job.state == "paused")
        part = self.root / "org__Small-GGUF" / "Small-Q8_0.gguf.part"
        partial = part.stat().st_size
        self.assertGreater(partial, 0)
        self.hub.delay = 0
        self.hub.requests.clear()
        queue.resume(job.id)
        wait_for(lambda: finished)
        self.assertEqual((self.root / "org__Small-GGUF" / "Small-Q8_0.gguf").read_bytes(), self.data)
        self.assertTrue(any("resolve" in r for r in self.hub.requests))

    def test_a_corrupt_file_is_deleted_and_reported(self) -> None:
        queue, _ = self.downloads()
        files = self.files()
        files[0]["sha256"] = "0" * 64
        job = queue.add("org/Small-GGUF", "Small", files)
        wait_for(lambda: job.state == "failed")
        self.assertIn("checksum", job.error)
        self.assertFalse((self.root / "org__Small-GGUF" / "Small-Q8_0.gguf").exists())
        self.assertFalse((self.root / "org__Small-GGUF" / "Small-Q8_0.gguf.part").exists())

    def test_gated_models_and_unsafe_names_are_refused(self) -> None:
        queue, _ = self.downloads()
        job = queue.add("org/Gated-GGUF", "Gated", [{"path": "G-Q4_K_M.gguf", "size": 4}])
        wait_for(lambda: job.state == "failed")
        self.assertIn("login", job.error)
        for bad in ("../escape.gguf", "/etc/passwd"):
            with self.assertRaises(ValueError):
                queue.add("org/Small-GGUF", "x", [{"path": bad, "size": 1}])

    def test_cancel_removes_the_partial_file(self) -> None:
        self.hub.delay = 0.3
        queue, _ = self.downloads()
        job = queue.add("org/Small-GGUF", "Small", self.files())
        wait_for(lambda: job.done > 0)
        queue.remove(job.id)
        self.assertEqual(queue.snapshot(), [])
        self.assertFalse((self.root / "org__Small-GGUF" / "Small-Q8_0.gguf.part").exists())


if __name__ == "__main__":
    unittest.main()
