"""
Phase 1: Project Infrastructure
- CMake build system
- Dependency management
- Cross-platform support
"""

import os
import subprocess

class InfrastructurePhase:
    """
    Handles project infrastructure setup: CMake, dependencies, cross-platform support.
    """
    def __init__(self, root_dir=None):
        # Use the workspace root (where CMakeLists.txt is), not the python package root
        self.root_dir = root_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))

    def setup_cmake(self):
        """Configure CMake build system (creates build directory and runs cmake)."""
        build_dir = os.path.join(self.root_dir, 'build')
        os.makedirs(build_dir, exist_ok=True)
        print(f"[Phase1] Running cmake in {build_dir}...")
        result = subprocess.run([
            'cmake', '..',
            '-DPENTA_BUILD_JUCE_PLUGIN=OFF',
            '-DPENTA_BUILD_TESTS=OFF'
        ], cwd=build_dir, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("CMake configuration failed.")

    def setup_dependencies(self):
        """Check for required dependencies (pybind11, JUCE, etc)."""
        print("[Phase1] Checking for pybind11...")
        try:
            import pybind11
            print(f"pybind11 found: {pybind11.__version__}")
        except ImportError:
            print("pybind11 not found. Please install with 'pip install pybind11'.")
        # Add checks for JUCE, oscpack, etc. as needed

    def verify_platforms(self):
        """Check cross-platform compatibility (Linux, macOS, Windows)."""
        import platform
        print(f"[Phase1] Platform: {platform.system()} {platform.release()}")
        # Could add more detailed checks here
