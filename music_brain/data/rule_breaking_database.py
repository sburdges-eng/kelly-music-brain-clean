"""
Rule-breaking database loader.

Exports RULE_BREAKS loaded from the bundled JSON file so tests can import
`music_brain.data.rule_breaking_database`.
"""

import json
from pathlib import Path
from typing import Any, List

DATA_FILE = Path(__file__).parent / "rule_breaking_database.json"

try:
    if DATA_FILE.exists():
        with open(DATA_FILE, "r") as f:
            RULE_BREAKS: List[Any] = json.load(f)
    else:
        RULE_BREAKS = []
except Exception:
    RULE_BREAKS = []

__all__ = ["RULE_BREAKS", "DATA_FILE"]

