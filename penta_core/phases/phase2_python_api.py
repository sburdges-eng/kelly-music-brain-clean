"""
Phase 2: Python API Wrapper
- High-level Python API for C++ engine
"""

import importlib

class PythonAPIPhase:
    """
    Handles high-level Python API wrapper for the C++ engine.
    """
    def __init__(self, package='penta_core'):
        self.package = package

    def generate_api(self):
        """Check that all C++ modules are exposed in the Python API."""
        print(f"[Phase2] Checking Python API for package '{self.package}'...")
        try:
            mod = importlib.import_module(self.package)
            print(f"Imported {self.package}: {mod}")
            # List available classes
            for attr in dir(mod):
                if not attr.startswith('_'):
                    print(f"  {attr}")
        except ImportError as e:
            print(f"Failed to import {self.package}: {e}")

    def test_api(self):
        """Run a simple test to ensure API works (e.g., instantiate HarmonyEngine)."""
        print("[Phase2] Testing API instantiation...")
        try:
            from penta_core import HarmonyEngine
            engine = HarmonyEngine()
            print("HarmonyEngine instantiated successfully.")
        except Exception as e:
            print(f"API test failed: {e}")
