#!/usr/bin/env python3
"""
Name Standardization Script

Standardizes naming across the monorepo:
- Python packages: kelly_core, kelly_cli, kelly_desktop
- C++ namespaces: kelly::core, kelly::audio
- Files: snake_case for Python, PascalCase for C++
- Configs: kebab-case
- Handles *_music-brain, *_penta-core variants
- Updates all import statements
- Updates CMake targets

Usage:
    python standardize_names.py --scan /path/to/repo
    python standardize_names.py --fix --dry-run
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


class NameStandardizer:
    """Standardize naming conventions across the repository."""
    
    def __init__(self, repo_path: str, dry_run: bool = False):
        self.repo_path = Path(repo_path)
        self.dry_run = dry_run
        self.changes = []
        
        # Naming patterns to fix
        self.package_renames = {
            'music_brain': 'kelly_core',
            'penta_core': 'kelly_core',
            'daiw': 'kelly_cli',
        }
        
        # File suffix patterns to clean
        self.suffix_patterns = ['_music-brain', '_penta-core', '_music_brain', '_penta_core']
    
    def scan_repository(self):
        """Scan repository for naming issues."""
        print(f"Scanning repository: {self.repo_path}")
        
        issues = {
            'package_names': [],
            'file_suffixes': [],
            'import_statements': [],
            'cmake_targets': [],
        }
        
        for py_file in self.repo_path.rglob('*.py'):
            # Skip certain directories
            if self._should_skip(py_file):
                continue
            
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            
            # Check for old package imports
            for old_pkg, new_pkg in self.package_renames.items():
                if old_pkg in content:
                    issues['import_statements'].append((str(py_file), old_pkg, new_pkg))
        
        # Check file names
        for file_path in self.repo_path.rglob('*'):
            if self._should_skip(file_path):
                continue
            
            name = file_path.name
            for suffix in self.suffix_patterns:
                if suffix in name:
                    new_name = name.replace(suffix, '')
                    issues['file_suffixes'].append((str(file_path), new_name))
        
        # Report findings
        self._report_issues(issues)
        return issues
    
    def _should_skip(self, path: Path) -> bool:
        """Check if path should be skipped."""
        skip_patterns = [
            '.git', 'node_modules', '__pycache__', '.venv', 'venv',
            'build', 'dist', '.cache', 'external', 'JUCE'
        ]
        
        parts = path.parts
        for pattern in skip_patterns:
            if pattern in parts:
                return True
        return False
    
    def _report_issues(self, issues: Dict[str, List]):
        """Report found issues."""
        print("\n" + "="*60)
        print("NAMING ISSUES FOUND")
        print("="*60)
        
        if issues['import_statements']:
            print(f"\n📦 Import Statements ({len(issues['import_statements'])} files):")
            for file, old, new in issues['import_statements'][:10]:
                rel_path = Path(file).relative_to(self.repo_path)
                print(f"  {rel_path}: {old} → {new}")
            if len(issues['import_statements']) > 10:
                print(f"  ... and {len(issues['import_statements']) - 10} more")
        
        if issues['file_suffixes']:
            print(f"\n📄 File Name Suffixes ({len(issues['file_suffixes'])} files):")
            for file, new_name in issues['file_suffixes'][:10]:
                rel_path = Path(file).relative_to(self.repo_path)
                print(f"  {rel_path} → {new_name}")
            if len(issues['file_suffixes']) > 10:
                print(f"  ... and {len(issues['file_suffixes']) - 10} more")
    
    def fix_import_statements(self):
        """Fix import statements in Python files."""
        print("\n🔧 Fixing import statements...")
        count = 0
        
        for py_file in self.repo_path.rglob('*.py'):
            if self._should_skip(py_file):
                continue
            
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            original_content = content
            
            # Replace package names in imports
            for old_pkg, new_pkg in self.package_renames.items():
                # from old_pkg import ...
                content = re.sub(
                    rf'\bfrom {old_pkg}\b',
                    f'from {new_pkg}',
                    content
                )
                # import old_pkg
                content = re.sub(
                    rf'\bimport {old_pkg}\b',
                    f'import {new_pkg}',
                    content
                )
                # old_pkg.something
                content = re.sub(
                    rf'\b{old_pkg}\.',
                    f'{new_pkg}.',
                    content
                )
            
            if content != original_content:
                if self.dry_run:
                    rel_path = py_file.relative_to(self.repo_path)
                    print(f"  [DRY RUN] Would fix: {rel_path}")
                else:
                    py_file.write_text(content, encoding='utf-8')
                    rel_path = py_file.relative_to(self.repo_path)
                    print(f"  ✓ Fixed: {rel_path}")
                count += 1
        
        print(f"✓ Fixed {count} files")
    
    def fix_file_names(self):
        """Fix file names with unwanted suffixes."""
        print("\n🔧 Fixing file names...")
        count = 0
        
        # Collect renames first to avoid conflicts
        renames = []
        
        for file_path in self.repo_path.rglob('*'):
            if self._should_skip(file_path):
                continue
            
            name = file_path.name
            new_name = name
            
            for suffix in self.suffix_patterns:
                if suffix in new_name:
                    new_name = new_name.replace(suffix, '')
            
            if new_name != name:
                new_path = file_path.parent / new_name
                if not new_path.exists():
                    renames.append((file_path, new_path))
        
        # Execute renames
        for old_path, new_path in renames:
            if self.dry_run:
                rel_old = old_path.relative_to(self.repo_path)
                rel_new = new_path.relative_to(self.repo_path)
                print(f"  [DRY RUN] Would rename: {rel_old} → {new_path.name}")
            else:
                old_path.rename(new_path)
                rel_new = new_path.relative_to(self.repo_path)
                print(f"  ✓ Renamed: {rel_new}")
            count += 1
        
        print(f"✓ Renamed {count} files")
    
    def fix_cmake_files(self):
        """Fix CMake target names."""
        print("\n🔧 Fixing CMake files...")
        count = 0
        
        for cmake_file in self.repo_path.rglob('CMakeLists.txt'):
            if self._should_skip(cmake_file):
                continue
            
            content = cmake_file.read_text(encoding='utf-8', errors='ignore')
            original_content = content
            
            # Standardize project names
            content = re.sub(
                r'project\(MusicBrain',
                'project(Kelly',
                content
            )
            content = re.sub(
                r'project\(PentaCore',
                'project(Kelly',
                content
            )
            
            # Standardize target names
            old_targets = ['MusicBrainCore', 'PentaCoreLib', 'DAiW']
            new_target = 'KellyCore'
            
            for old_target in old_targets:
                content = content.replace(old_target, new_target)
            
            if content != original_content:
                if self.dry_run:
                    rel_path = cmake_file.relative_to(self.repo_path)
                    print(f"  [DRY RUN] Would fix: {rel_path}")
                else:
                    cmake_file.write_text(content, encoding='utf-8')
                    rel_path = cmake_file.relative_to(self.repo_path)
                    print(f"  ✓ Fixed: {rel_path}")
                count += 1
        
        print(f"✓ Fixed {count} CMake files")
    
    def generate_report(self) -> str:
        """Generate standardization report."""
        lines = [
            "# Name Standardization Report",
            "",
            "## Changes Made",
            ""
        ]
        
        if self.dry_run:
            lines.append("**Note:** Dry run - no actual changes made")
            lines.append("")
        
        for change in self.changes:
            lines.append(f"- {change}")
        
        return '\n'.join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Standardize naming conventions across the repository"
    )
    parser.add_argument(
        '--scan',
        type=str,
        default='.',
        help='Repository path to scan (default: current directory)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Apply fixes (default: scan only)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fixed without making changes'
    )
    
    args = parser.parse_args()
    
    standardizer = NameStandardizer(args.scan, dry_run=args.dry_run)
    
    if args.fix or args.dry_run:
        standardizer.fix_import_statements()
        standardizer.fix_file_names()
        standardizer.fix_cmake_files()
        
        if args.dry_run:
            print("\n[DRY RUN] No changes were made")
    else:
        standardizer.scan_repository()


if __name__ == '__main__':
    main()
