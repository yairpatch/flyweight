"""GPU setup without the CUDA toolkit: what is downloaded, where it lands, and
what the engine is told. Runs offline against a local stand-in for PyPI."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import threading
import unittest
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from flyweight import gpu_setup


class LayoutTests(unittest.TestCase):
    def test_wheel_members_land_where_the_runtime_looks(self) -> None:
        root = Path("/r")
        linux = {
            "nvidia/cuda_runtime/include/cuda_fp16.h": root / "include/cuda_fp16.h",
            "nvidia/cuda_cccl/include/cub/config.cuh": root / "include/cub/config.cuh",
            "nvidia/cu13/include/cccl/cub/config.cuh": root / "include/cccl/cub/config.cuh",
            "nvidia/cuda_nvrtc/lib/libnvrtc.so.12": root / "lib64/libnvrtc.so.12",
            "nvidia/cu13/lib/libnvrtc-builtins.so.13.4": root / "lib64/libnvrtc-builtins.so.13.4",
            "nvidia/cuda_nvrtc/lib/libnvrtc.alt.so.12": None,
            "nvidia/cuda_runtime/lib/libcudart.so.12": None,
            "nvidia/cuda_cccl/lib/cmake/cub/cub-config.cmake": None,
            "nvidia/__init__.py": None,
            "nvidia/cuda_runtime/include/../../evil.h": None,
        }
        with patch.object(gpu_setup.os, "name", "posix"):
            for member, expected in linux.items():
                self.assertEqual(gpu_setup.target_path(member, root), expected, member)
        windows = {
            "nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll": root / "bin/nvrtc64_120_0.dll",
            "nvidia/cu13/bin/x86_64/nvrtc-builtins64_134.dll": root / "bin/nvrtc-builtins64_134.dll",
            "nvidia/cuda_nvrtc/bin/nvrtc64_120_0.alt.dll": None,
            "nvidia/cuda_nvrtc/lib/libnvrtc.so.12": None,
        }
        with patch.object(gpu_setup.os, "name", "nt"):
            for member, expected in windows.items():
                self.assertEqual(gpu_setup.target_path(member, root), expected, member)


class StatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.patches = [
            patch.object(gpu_setup, "data_dir", return_value=Path(self._dir.name)),
            patch.object(gpu_setup, "_system_toolkit", return_value=False),
            patch.object(gpu_setup, "nvidia_gpu_names", return_value=[]),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        self._dir.cleanup()

    def state(self, driver: int | None) -> gpu_setup.GpuStatus:
        with patch.object(gpu_setup, "driver_cuda_version", return_value=driver):
            return gpu_setup.status()

    def test_each_situation_gets_its_own_next_step(self) -> None:
        self.assertEqual(self.state(None).state, "none")
        with patch.object(gpu_setup, "nvidia_gpu_names", return_value=["RTX 4070"]):
            self.assertEqual(self.state(None).state, "needs-driver")
        self.assertEqual(self.state(11080).state, "old-driver")
        twelve = self.state(12040)
        self.assertEqual((twelve.state, twelve.major, twelve.driver), ("needs-setup", 12, "12.4"))
        self.assertEqual(self.state(13020).major, 13)
        with patch.object(gpu_setup, "_system_toolkit", return_value=True):
            self.assertEqual(self.state(12040).state, "ready")

    def test_an_install_is_used_only_with_a_driver_that_can_run_it(self) -> None:
        root = gpu_setup.cuda_dir(12)
        root.mkdir(parents=True)
        (root / gpu_setup.MARKER).write_text("{}")
        self.assertEqual(self.state(12040).state, "ready")
        with patch.object(gpu_setup, "driver_cuda_version", return_value=12040):
            env = gpu_setup.engine_environment()
        self.assertEqual(env["CUDA_PATH"], str(root))
        with patch.object(gpu_setup, "driver_cuda_version", return_value=None):
            self.assertEqual(gpu_setup.engine_environment(), {})


def _wheel(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class InstallTests(unittest.TestCase):
    """The whole download-verify-unpack path, against a local package index."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        site = self.root / "site"
        site.mkdir()
        lib = "nvidia/cuda_nvrtc/lib/libnvrtc.so.12" if os.name != "nt" \
            else "nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll"
        self.wheels = {
            "nvidia-cuda-nvrtc-cu12": _wheel({lib: b"nvrtc"}),
            "nvidia-cuda-runtime-cu12": _wheel({"nvidia/cuda_runtime/include/cuda_fp16.h": b"//"}),
            "nvidia-cuda-cccl-cu12": _wheel({"nvidia/cuda_cccl/include/cub/config.cuh": b"//"}),
        }
        for name, data in self.wheels.items():
            (site / f"{name}.whl").write_bytes(data)
        handler = partial(SimpleHTTPRequestHandler, directory=str(site))
        handler.log_message = lambda *a: None  # type: ignore[method-assign]
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.digest = {n: hashlib.sha256(d).hexdigest() for n, d in self.wheels.items()}
        self.index = {n: (f"{base}/{n}.whl", self.digest[n], len(d)) for n, d in self.wheels.items()}

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._dir.cleanup()

    def test_install_unpacks_a_complete_marked_folder(self) -> None:
        messages: list[float] = []
        with patch.object(gpu_setup, "data_dir", return_value=self.root / "data"), \
                patch.object(gpu_setup, "_wheel_for", side_effect=self.index.__getitem__):
            folder = gpu_setup.install(12, lambda m, f: messages.append(f))
            self.assertEqual(gpu_setup.installed_major(), 12)
        self.assertTrue((folder / "include/cuda_fp16.h").is_file())
        self.assertTrue((folder / "include/cub/config.cuh").is_file())
        self.assertTrue((folder / ("bin" if os.name == "nt" else "lib64")).is_dir())
        self.assertEqual(json.loads((folder / gpu_setup.MARKER).read_text())["major"], 12)
        self.assertEqual(messages[-1], 1.0)
        self.assertEqual(messages, sorted(messages))

    def test_a_corrupt_download_leaves_nothing_behind(self) -> None:
        self.index["nvidia-cuda-cccl-cu12"] = (self.index["nvidia-cuda-cccl-cu12"][0], "0" * 64, 1)
        with patch.object(gpu_setup, "data_dir", return_value=self.root / "data"), \
                patch.object(gpu_setup, "_wheel_for", side_effect=self.index.__getitem__):
            with self.assertRaises(RuntimeError) as raised:
                gpu_setup.install(12)
            self.assertIn("checksum", str(raised.exception))
            self.assertIsNone(gpu_setup.installed_major())
        self.assertEqual(list((self.root / "data").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
