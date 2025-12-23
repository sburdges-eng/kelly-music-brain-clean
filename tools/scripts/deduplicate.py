#!/usr/bin/env python3
"""
Duplicate File Detection and Analysis Tool

Finds duplicate files by:
- Content hash (MD5/SHA256)
- File size
- Name patterns (*_music-brain, *_penta-core variants)
- Semantic comparison for JSON/YAML files

Generates actionable reports with merge recommendations.

Usage:
    python deduplicate.py --scan /path/to/repo
    python deduplicate.py --scan . --output report.md --format markdown
    python deduplicate.py --scan . --output report.json --format json
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import difflib


@dataclass
class FileInfo:
    """Information about a file."""
    path: str
    size: int
    md5: str
    sha256: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DuplicateGroup:
    """Group of duplicate files."""
    files: List[FileInfo]
    reason: str  # 'exact_match', 'name_pattern', 'semantic_match'
    recommendation: str
    
    def to_dict(self):
        return {
            'files': [f.to_dict() for f in self.files],
            'reason': self.reason,
            'recommendation': self.recommendation
        }


class DuplicateFinder:
    """Find duplicate files in a directory tree."""
    
    def __init__(self, root_path: str, exclude_patterns: List[str] = None):
        self.root_path = Path(root_path)
        self.exclude_patterns = exclude_patterns or [
            '.git', 'node_modules', '__pycache__', '.venv', 'venv',
            'build', 'dist', '.cache', '.mypy_cache', '.pytest_cache',
            'external', 'JUCE'  # Exclude external dependencies
        ]
        self.files_by_hash: Dict[str, List[FileInfo]] = defaultdict(list)
        self.files_by_name: Dict[str, List[FileInfo]] = defaultdict(list)
        self.duplicate_groups: List[DuplicateGroup] = []
        
    def should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded."""
        parts = path.parts
        for pattern in self.exclude_patterns:
            if pattern in parts:
                return True
        return False
    
    def compute_file_hash(self, filepath: Path) -> Tuple[str, str]:
        """Compute MD5 and SHA256 hashes of a file."""
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        try:
            with open(filepath, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b''):
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)
            return md5_hash.hexdigest(), sha256_hash.hexdigest()
        except (IOError, OSError) as e:
            print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
            return "", ""
    
    def scan_directory(self):
        """Scan directory tree and index all files."""
        print(f"Scanning directory: {self.root_path}")
        file_count = 0
        
        for filepath in self.root_path.rglob('*'):
            if not filepath.is_file():
                continue
            
            if self.should_exclude(filepath):
                continue
            
            # Skip very large files (> 100MB)
            try:
                size = filepath.stat().st_size
                if size > 100 * 1024 * 1024:
                    continue
            except OSError:
                continue
            
            md5, sha256 = self.compute_file_hash(filepath)
            if not md5:
                continue
            
            rel_path = filepath.relative_to(self.root_path)
            file_info = FileInfo(
                path=str(rel_path),
                size=size,
                md5=md5,
                sha256=sha256
            )
            
            # Index by hash
            self.files_by_hash[sha256].append(file_info)
            
            # Index by name
            name = filepath.name
            self.files_by_name[name].append(file_info)
            
            file_count += 1
            if file_count % 100 == 0:
                print(f"  Scanned {file_count} files...", end='\r')
        
        print(f"\n✓ Scanned {file_count} files")
    
    def find_exact_duplicates(self):
        """Find files with identical content (hash)."""
        print("Finding exact duplicates...")
        count = 0
        
        for sha256, files in self.files_by_hash.items():
            if len(files) < 2:
                continue
            
            # Sort by path length (prefer shorter paths as canonical)
            files.sort(key=lambda f: len(f.path))
            
            recommendation = f"Keep: {files[0].path}\nDelete: " + ", ".join(f.path for f in files[1:])
            
            self.duplicate_groups.append(DuplicateGroup(
                files=files,
                reason='exact_match',
                recommendation=recommendation
            ))
            count += 1
        
        print(f"✓ Found {count} groups of exact duplicates")
    
    def find_name_pattern_duplicates(self):
        """Find files with similar names (e.g., *_music-brain, *_penta-core)."""
        print("Finding name pattern duplicates...")
        count = 0
        
        # Patterns to look for
        patterns = [
            ('_music-brain', '_penta-core'),
            ('_music-brain', ''),
            ('_penta-core', ''),
        ]
        
        checked_groups = set()
        
        for name, files in self.files_by_name.items():
            if len(files) < 2:
                continue
            
            # Check if any files have pattern suffixes
            base_names = {}
            for f in files:
                fname = Path(f.path).stem
                
                # Try to extract base name
                for p1, p2 in patterns:
                    for pattern in [p1, p2]:
                        if pattern and fname.endswith(pattern):
                            base = fname[:-len(pattern)]
                            if base not in base_names:
                                base_names[base] = []
                            base_names[base].append(f)
                            break
                    else:
                        continue
                    break
            
            # Group files by base name
            for base, group_files in base_names.items():
                if len(group_files) < 2:
                    continue
                
                # Create unique key for this group
                group_key = tuple(sorted(f.path for f in group_files))
                if group_key in checked_groups:
                    continue
                checked_groups.add(group_key)
                
                # Check if they have similar content (size within 10%)
                sizes = [f.size for f in group_files]
                if sizes:
                    avg_size = sum(sizes) / len(sizes)
                    if all(abs(s - avg_size) / avg_size < 0.1 for s in sizes if avg_size > 0):
                        group_files.sort(key=lambda f: len(f.path))
                        recommendation = f"Review: {', '.join(f.path for f in group_files)}\nLikely variants of same file"
                        
                        self.duplicate_groups.append(DuplicateGroup(
                            files=group_files,
                            reason='name_pattern',
                            recommendation=recommendation
                        ))
                        count += 1
        
        print(f"✓ Found {count} groups of name pattern duplicates")
    
    def find_semantic_duplicates(self):
        """Find JSON/YAML files with same semantic content."""
        print("Finding semantic duplicates...")
        count = 0
        
        json_files = []
        for files in self.files_by_name.values():
            for f in files:
                if f.path.endswith(('.json', '.yaml', '.yml')):
                    json_files.append(f)
        
        # Group by normalized content
        content_groups = defaultdict(list)
        
        for f in json_files:
            try:
                full_path = self.root_path / f.path
                with open(full_path, 'r', encoding='utf-8') as file:
                    if f.path.endswith('.json'):
                        data = json.load(file)
                        # Normalize by dumping with sorted keys
                        normalized = json.dumps(data, sort_keys=True)
                        content_groups[normalized].append(f)
            except (json.JSONDecodeError, IOError, OSError):
                continue
        
        for normalized, files in content_groups.items():
            if len(files) < 2:
                continue
            
            files.sort(key=lambda f: len(f.path))
            recommendation = f"Keep: {files[0].path}\nDelete or merge: " + ", ".join(f.path for f in files[1:])
            
            self.duplicate_groups.append(DuplicateGroup(
                files=files,
                reason='semantic_match',
                recommendation=recommendation
            ))
            count += 1
        
        print(f"✓ Found {count} groups of semantic duplicates")
    
    def analyze(self):
        """Run all duplicate detection analyses."""
        self.scan_directory()
        self.find_exact_duplicates()
        self.find_name_pattern_duplicates()
        self.find_semantic_duplicates()
    
    def generate_markdown_report(self) -> str:
        """Generate a Markdown report."""
        lines = [
            "# Duplicate Files Analysis Report",
            "",
            f"**Repository:** {self.root_path}",
            f"**Total duplicate groups found:** {len(self.duplicate_groups)}",
            "",
            "---",
            ""
        ]
        
        # Group by reason
        groups_by_reason = defaultdict(list)
        for group in self.duplicate_groups:
            groups_by_reason[group.reason].append(group)
        
        for reason, groups in groups_by_reason.items():
            lines.append(f"## {reason.replace('_', ' ').title()} ({len(groups)} groups)")
            lines.append("")
            
            for i, group in enumerate(groups, 1):
                lines.append(f"### Group {i}")
                lines.append("")
                lines.append("**Files:**")
                for f in group.files:
                    lines.append(f"- `{f.path}` ({f.size} bytes, MD5: {f.md5[:8]}...)")
                lines.append("")
                lines.append("**Recommendation:**")
                for line in group.recommendation.split('\n'):
                    lines.append(f"> {line}")
                lines.append("")
        
        # Generate deletion script
        lines.append("---")
        lines.append("")
        lines.append("## Automated Deletion Script")
        lines.append("")
        lines.append("```bash")
        lines.append("#!/bin/bash")
        lines.append("# Generated deletion script - REVIEW BEFORE RUNNING")
        lines.append("")
        
        for group in self.duplicate_groups:
            if group.reason == 'exact_match' and len(group.files) > 1:
                # Keep first (shortest path), delete rest
                for f in group.files[1:]:
                    lines.append(f'rm "{f.path}"  # Duplicate of {group.files[0].path}')
        
        lines.append("```")
        lines.append("")
        
        return '\n'.join(lines)
    
    def generate_json_report(self) -> str:
        """Generate a JSON report."""
        report = {
            'repository': str(self.root_path),
            'total_groups': len(self.duplicate_groups),
            'groups': [g.to_dict() for g in self.duplicate_groups]
        }
        return json.dumps(report, indent=2)
    
    def generate_csv_report(self) -> str:
        """Generate a CSV report."""
        lines = ["Group,Reason,File Path,Size,MD5,SHA256"]
        
        for i, group in enumerate(self.duplicate_groups, 1):
            for f in group.files:
                lines.append(f'{i},{group.reason},"{f.path}",{f.size},{f.md5},{f.sha256}')
        
        return '\n'.join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Find and analyze duplicate files in a repository"
    )
    parser.add_argument(
        '--scan',
        type=str,
        default='.',
        help='Directory to scan (default: current directory)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: print to stdout)'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'json', 'csv'],
        default='markdown',
        help='Output format (default: markdown)'
    )
    parser.add_argument(
        '--exclude',
        nargs='+',
        help='Additional patterns to exclude'
    )
    
    args = parser.parse_args()
    
    # Create finder
    finder = DuplicateFinder(args.scan, exclude_patterns=args.exclude)
    
    # Run analysis
    finder.analyze()
    
    # Generate report
    if args.format == 'markdown':
        report = finder.generate_markdown_report()
    elif args.format == 'json':
        report = finder.generate_json_report()
    else:  # csv
        report = finder.generate_csv_report()
    
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding='utf-8')
        print(f"\n✓ Report written to: {args.output}")
    else:
        print("\n" + report)


if __name__ == '__main__':
    main()
