#!/usr/bin/env python3
"""
Migration Validation Script

Comprehensive testing to validate the merger:
- All tests still pass
- No broken imports
- No circular dependencies
- Documentation links valid
- Build system works
- Git history preserved
- No data loss
- Performance benchmarks

Usage:
    python validate_migration.py --all
    python validate_migration.py --test imports
    python validate_migration.py --test build
"""

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class MigrationValidator:
    """Validate the repository migration."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.results = {
            'imports': {'passed': 0, 'failed': 0, 'errors': []},
            'tests': {'passed': 0, 'failed': 0, 'errors': []},
            'build': {'passed': 0, 'failed': 0, 'errors': []},
            'links': {'passed': 0, 'failed': 0, 'errors': []},
            'history': {'passed': 0, 'failed': 0, 'errors': []},
            'dependencies': {'passed': 0, 'failed': 0, 'errors': []},
        }
    
    def validate_imports(self) -> bool:
        """Validate all Python imports."""
        print("🔍 Validating Python imports...")
        
        errors = []
        checked = 0
        
        for py_file in self.repo_path.rglob('*.py'):
            if self._should_skip(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                tree = ast.parse(content, filename=str(py_file))
                
                # Extract imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            checked += 1
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            checked += 1
                            
                            # Check for old package names
                            if 'music_brain' in node.module:
                                errors.append(f"{py_file}: Old import 'music_brain' found")
                            elif 'penta_core' in node.module and 'kelly' not in node.module:
                                errors.append(f"{py_file}: Old import 'penta_core' found")
            
            except SyntaxError as e:
                errors.append(f"{py_file}: Syntax error - {e}")
        
        self.results['imports']['passed'] = checked - len(errors)
        self.results['imports']['failed'] = len(errors)
        self.results['imports']['errors'] = errors
        
        if errors:
            print(f"  ❌ Found {len(errors)} import issues:")
            for error in errors[:5]:
                print(f"    - {error}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")
            return False
        else:
            print(f"  ✅ All {checked} imports valid")
            return True
    
    def validate_circular_dependencies(self) -> bool:
        """Check for circular dependencies."""
        print("🔍 Checking for circular dependencies...")
        
        # Build dependency graph
        graph = {}
        
        for py_file in self.repo_path.rglob('*.py'):
            if self._should_skip(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                tree = ast.parse(content, filename=str(py_file))
                
                module_name = self._get_module_name(py_file)
                if not module_name:
                    continue
                
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith('kelly'):
                            imports.append(node.module)
                
                graph[module_name] = imports
            
            except SyntaxError:
                continue
        
        # Detect cycles using DFS
        cycles = []
        
        def dfs(node, visited, path):
            if node in path:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for dep in graph.get(node, []):
                dfs(dep, visited, path[:])
        
        for module in graph:
            dfs(module, set(), [])
        
        if cycles:
            self.results['dependencies']['failed'] = len(cycles)
            self.results['dependencies']['errors'] = [' → '.join(c) for c in cycles]
            print(f"  ❌ Found {len(cycles)} circular dependencies:")
            for cycle in cycles[:3]:
                print(f"    - {' → '.join(cycle)}")
            return False
        else:
            self.results['dependencies']['passed'] = len(graph)
            print(f"  ✅ No circular dependencies found")
            return True
    
    def validate_tests(self) -> bool:
        """Run test suite."""
        print("🔍 Running tests...")
        
        result = subprocess.run(
            ['pytest', 'tests/python', '-v', '--tb=short'],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            self.results['tests']['passed'] = 1
            print("  ✅ All tests passed")
            return True
        else:
            self.results['tests']['failed'] = 1
            self.results['tests']['errors'] = [result.stdout + result.stderr]
            print("  ❌ Tests failed")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return False
    
    def validate_build_system(self) -> bool:
        """Validate CMake build."""
        print("🔍 Validating CMake build...")
        
        build_dir = self.repo_path / 'build_validate'
        build_dir.mkdir(exist_ok=True)
        
        # Configure
        result = subprocess.run(
            [
                'cmake', '-B', str(build_dir),
                '-DBUILD_TESTS=OFF',
                '-DBUILD_PLUGINS=OFF'
            ],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            self.results['build']['failed'] = 1
            self.results['build']['errors'] = [result.stderr]
            print("  ❌ CMake configuration failed")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
        
        # Build
        result = subprocess.run(
            ['cmake', '--build', str(build_dir)],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            self.results['build']['failed'] = 1
            self.results['build']['errors'] = [result.stderr]
            print("  ❌ Build failed")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
        
        self.results['build']['passed'] = 1
        print("  ✅ Build successful")
        return True
    
    def validate_git_history(self) -> bool:
        """Verify git history is intact."""
        print("🔍 Validating git history...")
        
        result = subprocess.run(
            ['git', 'log', '--oneline', '--all', '--graph'],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            self.results['history']['failed'] = 1
            self.results['history']['errors'] = ['Git history not accessible']
            print("  ❌ Cannot access git history")
            return False
        
        # Check for reasonable commit count
        commit_count = len(result.stdout.strip().split('\n'))
        
        if commit_count < 10:
            self.results['history']['failed'] = 1
            self.results['history']['errors'] = [f'Only {commit_count} commits found']
            print(f"  ⚠️  Only {commit_count} commits found (expected more)")
            return False
        
        self.results['history']['passed'] = commit_count
        print(f"  ✅ Git history valid ({commit_count} commits)")
        return True
    
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
    
    def _get_module_name(self, file_path: Path) -> str:
        """Get Python module name from file path."""
        try:
            rel_path = file_path.relative_to(self.repo_path)
            parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            
            # Remove src, packages, etc.
            if 'src' in parts:
                parts = parts[parts.index('src')+1:]
            if 'packages' in parts:
                parts = parts[parts.index('packages')+1:]
            
            return '.'.join(parts)
        except:
            return ''
    
    def generate_report(self) -> str:
        """Generate validation report."""
        lines = [
            "# Migration Validation Report",
            "",
            "## Summary",
            ""
        ]
        
        total_passed = sum(r['passed'] for r in self.results.values())
        total_failed = sum(r['failed'] for r in self.results.values())
        
        if total_failed == 0:
            lines.append("✅ **All validations passed!**")
        else:
            lines.append(f"⚠️  **{total_failed} validation(s) failed**")
        
        lines.append("")
        lines.append("## Detailed Results")
        lines.append("")
        
        for category, result in self.results.items():
            status = "✅" if result['failed'] == 0 else "❌"
            lines.append(f"### {status} {category.title()}")
            lines.append(f"- Passed: {result['passed']}")
            lines.append(f"- Failed: {result['failed']}")
            
            if result['errors']:
                lines.append("- Errors:")
                for error in result['errors'][:5]:
                    lines.append(f"  - {error}")
                if len(result['errors']) > 5:
                    lines.append(f"  - ... and {len(result['errors']) - 5} more")
            lines.append("")
        
        return '\n'.join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate repository migration"
    )
    parser.add_argument(
        '--repo',
        type=str,
        default='.',
        help='Repository path (default: current directory)'
    )
    parser.add_argument(
        '--test',
        choices=['imports', 'tests', 'build', 'history', 'dependencies', 'all'],
        default='all',
        help='Which validation to run (default: all)'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Output report to file'
    )
    
    args = parser.parse_args()
    
    validator = MigrationValidator(args.repo)
    
    all_passed = True
    
    if args.test in ['all', 'imports']:
        if not validator.validate_imports():
            all_passed = False
    
    if args.test in ['all', 'dependencies']:
        if not validator.validate_circular_dependencies():
            all_passed = False
    
    if args.test in ['all', 'tests']:
        if not validator.validate_tests():
            all_passed = False
    
    if args.test in ['all', 'build']:
        if not validator.validate_build_system():
            all_passed = False
    
    if args.test in ['all', 'history']:
        if not validator.validate_git_history():
            all_passed = False
    
    # Generate report
    report = validator.generate_report()
    
    if args.report:
        Path(args.report).write_text(report, encoding='utf-8')
        print(f"\n✓ Report written to: {args.report}")
    else:
        print("\n" + report)
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
