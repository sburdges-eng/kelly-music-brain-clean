"""
Pytest configuration for the repo.

We explicitly put the project root at the front of sys.path so that imports
like ``import music_brain`` resolve to the real package instead of the
similarly named ``tests/music_brain`` test package. We also purge any
preloaded modules from other checkouts (e.g., miDiKompanion).
"""

import importlib
import sys
from pathlib import Path

# Ensure this repository's root is first on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Drop any conflicting installs (e.g., other checkouts of music_brain)
sys.path = [p for p in sys.path if "miDiKompanion" not in p]

# Purge any preloaded music_brain modules that came from elsewhere
for name, module in list(sys.modules.items()):
    if name.startswith("music_brain"):
        mod_file = getattr(module, "__file__", "") or ""
        if mod_file and "kelly-clean" not in mod_file:
            sys.modules.pop(name, None)

importlib.invalidate_caches()

# Force-load local music_brain after cleanup so subsequent imports use it
import music_brain  # noqa: E402
import music_brain.groove.templates as _local_templates  # noqa: E402,F401

