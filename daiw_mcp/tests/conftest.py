"""Test configuration for DAiW MCP tools."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


# Ensure repository root is on sys.path so local packages import correctly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Register the asyncio marker used throughout the suite."""

    config.addinivalue_line("markers", "asyncio: mark test as coroutine-based")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run coroutine tests without requiring pytest-asyncio."""

    test_func = pyfuncitem.obj
    if asyncio.iscoroutinefunction(test_func):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_func(**pyfuncitem.funcargs))
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        return True

    return None




