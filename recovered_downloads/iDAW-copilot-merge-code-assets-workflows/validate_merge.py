#!/usr/bin/env python3
"""
Validation script for iDAW merged repository.
Checks that all key components from both source repos are present.
"""

import os
import sys
from pathlib import Path

def check_exists(path, description):
    """Check if a path exists and print result."""
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def main():
    print("=" * 60)
    print("iDAW Repository Validation")
    print("=" * 60)
    
    checks_passed = 0
    checks_total = 0
    
    # Check documentation files
    print("\n📄 Documentation Files:")
    checks = [
        ("README.md", "Main README"),
        ("README_penta-core.md", "Penta Core README"),
        ("README_music-brain.md", "Music Brain README"),
        ("ROADMAP_penta-core.md", "Penta Core Roadmap"),
        ("DEVELOPMENT_ROADMAP_music-brain.md", "Music Brain Roadmap"),
        ("MERGE_SUMMARY.md", "Merge Summary"),
    ]
    for path, desc in checks:
        checks_total += 1
        if check_exists(path, desc):
            checks_passed += 1
    
    # Check configuration files
    print("\n⚙️  Configuration Files:")
    checks = [
        ("pyproject.toml", "Unified pyproject.toml"),
        ("requirements.txt", "Unified requirements.txt"),
        (".gitignore", "Git ignore file"),
        ("CMakeLists.txt", "CMake build file"),
    ]
    for path, desc in checks:
        checks_total += 1
        if check_exists(path, desc):
            checks_passed += 1
    
    # Check GitHub configuration
    print("\n🔧 GitHub Configuration:")
    checks = [
        (".github/workflows/ci.yml", "CI workflow"),
        (".github/agents/my-agent.agent.md", "Custom agent"),
        (".github/copilot-instructions.md", "Copilot instructions"),
    ]
    for path, desc in checks:
        checks_total += 1
        if check_exists(path, desc):
            checks_passed += 1
    
    # Check penta-core components
    print("\n🔬 Penta Core Components:")
    checks = [
        ("include/", "C++ headers"),
        ("src_penta-core/", "C++ source"),
        ("bindings/", "Python bindings"),
        ("python/", "Python package"),
        ("plugins/", "JUCE plugins"),
        ("external/", "External dependencies"),
        ("docs_penta-core/", "Penta Core docs"),
        ("examples_penta-core/", "Penta Core examples"),
        ("tests_penta-core/", "Penta Core tests"),
    ]
    for path, desc in checks:
        checks_total += 1
        if check_exists(path, desc):
            checks_passed += 1
    
    # Check DAiW-Music-Brain components
    print("\n🎵 DAiW-Music-Brain Components:")
    checks = [
        ("music_brain/", "Music Brain package"),
        ("mcp_todo/", "MCP TODO server"),
        ("mcp_workstation/", "MCP workstation"),
        ("vault/", "Knowledge vault"),
        ("tools/", "Tools directory"),
        ("data/", "Data directory"),
        ("docs_music-brain/", "Music Brain docs"),
        ("examples_music-brain/", "Music Brain examples"),
        ("tests_music-brain/", "Music Brain tests"),
    ]
    for path, desc in checks:
        checks_total += 1
        if check_exists(path, desc):
            checks_passed += 1
    
    # Check Python package structure
    print("\n🐍 Python Package Structure:")
    checks = [
        ("music_brain/__init__.py", "Music Brain init"),
        ("music_brain/groove/", "Groove module"),
        ("music_brain/structure/", "Structure module"),
        ("music_brain/session/", "Session module"),
        ("music_brain/audio/", "Audio module"),
        ("python/penta_core/__init__.py", "Penta Core init"),
    ]
    for path, desc in checks:
        checks_total += 1
        if check_exists(path, desc):
            checks_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Validation Results: {checks_passed}/{checks_total} checks passed")
    percentage = (checks_passed / checks_total) * 100 if checks_total > 0 else 0
    print(f"Success Rate: {percentage:.1f}%")
    print("=" * 60)
    
    if checks_passed == checks_total:
        print("\n✅ All validation checks passed!")
        return 0
    else:
        print(f"\n⚠️  {checks_total - checks_passed} checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
