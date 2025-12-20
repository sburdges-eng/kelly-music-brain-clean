"""
Phase 4: JUCE Plugin Integration
- VST3/AU/Standalone plugin
"""

import os
import subprocess

class PluginPhase:
    """
    Handles JUCE plugin integration: VST3, AU, Standalone.
    """
    def __init__(self, root_dir=None):
        self.root_dir = root_dir or os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    def build_plugin(self):
        """Build JUCE plugin targets using CMake."""
        build_dir = os.path.join(self.root_dir, 'build')
        print(f"[Phase4] Building JUCE plugin in {build_dir}...")
        result = subprocess.run(['cmake', '--build', '.', '--target', 'PluginEditor', '-j4'], cwd=build_dir, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("JUCE plugin build failed.")

    def test_plugin(self):
        """Stub: Run plugin tests and validation."""
        print("[Phase4] (Stub) Run plugin tests here.")
