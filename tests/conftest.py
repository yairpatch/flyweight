"""Suite-wide options.

Four parity tests account for 81% of the suite's wall time (14 of 17
minutes): each builds a fixture, loads the native runtime and generates the
same tokens twice to compare them bit for bit, and the worst runs that on
the CPU backend, where every CUDA kernel is emulated. They are worth having
and not worth waiting for on every edit, so they are marked `slow` and
skipped unless asked for.

    pytest                 # ~3 minutes, everything else
    pytest --run-slow      # the full suite
    pytest -n auto         # in parallel; combines with --run-slow
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="also run the tests marked `slow` (minutes each)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: exhaustive parity test measured in minutes; needs --run-slow",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow: pass --run-slow to include it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
