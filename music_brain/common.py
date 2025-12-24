"""
Shared helpers for MCP-facing music_brain modules.

Provides a small set of utilities that appear in the harmony, groove, and
intent tool wrappers:
- Lazy API accessor (`api`) so imports stay lightweight until first use
- Async helper (`run_sync`) to offload blocking calls
- Helpers for handling MIDI payloads and intent JSON
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

JsonDict = Dict[str, Any]


class _LazyApi:
    """Lazily create the heavy DAiW API wrapper on first access.

    Importing ``music_brain.api`` pulls in audio/voice dependencies. Creating
    the instance lazily keeps imports cheap for tooling contexts (like tests)
    that only touch a subset of functionality.
    """

    _instance = None

    def _get(self):
        if self._instance is None:
            from music_brain.api import DAiWAPI

            self._instance = DAiWAPI()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)


# Exposed API handle used by MCP tools
api = _LazyApi()


async def run_sync(func, *args, **kwargs) -> Any:
    """Run a blocking function in a worker thread and return its result."""

    return await asyncio.to_thread(func, *args, **kwargs)


def file_to_base64(path: str) -> str:
    """Return base64-encoded contents of a file."""

    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


@contextmanager
def midi_file_context(
    midi_path: Optional[str] = None,
    midi_base64: Optional[str] = None,
    suffix: str = ".mid",
) -> Generator[str, None, None]:
    """Yield a path to a MIDI file regardless of source format.

    Accepts either an existing file path or a base64-encoded MIDI blob.
    When given base64 data, writes to a temporary file and cleans it up
    after use.
    """

    if midi_path:
        yield midi_path
        return

    if not midi_base64:
        raise ValueError("Provide either 'midi_path' or 'midi_base64'.")

    data = base64.b64decode(midi_base64)
    fd, temp_path = tempfile.mkstemp(prefix="daiw_mcp_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        yield temp_path
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


def make_midi_payload(path: str, filename: Optional[str] = None) -> JsonDict:
    """Bundle MIDI content as a response-friendly payload."""

    return {
        "filename": filename or os.path.basename(path),
        "midi_base64": file_to_base64(path),
    }


def parse_intent_json(intent_json: str):
    """Deserialize a CompleteSongIntent from JSON."""

    from music_brain.session.intent_schema import CompleteSongIntent

    data = json.loads(intent_json)
    return CompleteSongIntent.from_dict(data)


__all__ = [
    "api",
    "run_sync",
    "midi_file_context",
    "make_midi_payload",
    "parse_intent_json",
]




