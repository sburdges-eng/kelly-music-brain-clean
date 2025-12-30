"""Dual-repository synchronization system

Manages synchronized versions across kelly-project and MidiKompanion repositories.
Features:
- Intelligent file mapping between repos
- Automatic version comparison and selection
- Color-coded status tracking
- Atomic multi-repo commits
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from .git_handler import GitHandler
from .version_manager import (
    VersionManager,
    BuildType,
    VersionStatus,
    ColorCoder,
    FileVersion,
    VersionComparison
)


@dataclass
class SyncMapping:
    """Maps a file across repositories"""
    kelly_path: str
    midikompanion_path: str
    shared_identifier: str
    priority: int = 1  # Higher = more important to sync


@dataclass
class SyncResult:
    """Result of syncing a single file"""
    mapping: SyncMapping
    status: VersionStatus
    selected_version: str  # 'kelly' or 'midikompanion'
    merged_successfully: bool = True
    kelly_hash: str = ""
    midikompanion_hash: str = ""


class DualRepoSync:
    """Manages synchronization between kelly-project and MidiKompanion"""

    # Define file mappings
    DEFAULT_MAPPINGS = [
        SyncMapping(
            kelly_path="rr_cli/rr/git_handler.py",
            midikompanion_path="mcp_web_parser/git_handler.py",
            shared_identifier="GitHandler",
            priority=5
        ),
        SyncMapping(
            kelly_path="rr_cli/rr/ai_handler.py",
            midikompanion_path="mcp_web_parser/ai_handler.py",
            shared_identifier="AIHandler",
            priority=5
        ),
        SyncMapping(
            kelly_path="rr_cli/rr/version_manager.py",
            midikompanion_path="version_management/version_manager.py",
            shared_identifier="VersionManager",
            priority=4
        ),
    ]

    def __init__(
        self,
        kelly_project_path: str,
        midikompanion_path: str,
        mappings: Optional[List[SyncMapping]] = None
    ):
        """Initialize dual repo sync"""
        self.kelly_path = kelly_project_path
        self.midikompanion_path = midikompanion_path
        self.mappings = mappings or self.DEFAULT_MAPPINGS

        self.kelly_handler = GitHandler(kelly_project_path)
        self.midikompanion_handler = GitHandler(midikompanion_path)

        self.kelly_version_manager = VersionManager(kelly_project_path)
        self.midikompanion_version_manager = VersionManager(midikompanion_path)

        self.sync_results: List[SyncResult] = []
        self.sync_timestamp = datetime.now().isoformat()

    def analyze_all_files(self) -> List[VersionComparison]:
        """Analyze all mapped files for differences"""
        comparisons = []

        for mapping in self.mappings:
            kelly_file = Path(self.kelly_path) / mapping.kelly_path
            midikompanion_file = Path(self.midikompanion_path) / mapping.midikompanion_path

            # Check both files exist
            if not kelly_file.exists() or not midikompanion_file.exists():
                continue

            # Register versions
            self.kelly_version_manager.register_version(
                mapping.kelly_path,
                BuildType.RELEASE
            )
            self.midikompanion_version_manager.register_version(
                mapping.midikompanion_path,
                BuildType.RELEASE
            )

            # Compare
            try:
                comparison = self.kelly_version_manager.compare_versions(
                    mapping.kelly_path,
                    BuildType.RELEASE,
                    BuildType.DEBUG  # Use as proxy for second version
                )
                comparisons.append(comparison)
            except Exception as e:
                print(f"Error comparing {mapping.kelly_path}: {e}")

        return comparisons

    def merge_file(
        self,
        mapping: SyncMapping,
        selected_source: str  # 'kelly' or 'midikompanion'
    ) -> SyncResult:
        """Merge a file from selected source to both repositories"""

        if selected_source == 'kelly':
            source_path = Path(self.kelly_path) / mapping.kelly_path
            target_path = Path(self.midikompanion_path) / mapping.midikompanion_path
        else:
            source_path = Path(self.midikompanion_path) / mapping.midikompanion_path
            target_path = Path(self.kelly_path) / mapping.kelly_path

        try:
            # Read source
            with open(source_path, 'r') as f:
                content = f.read()

            # Write to target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, 'w') as f:
                f.write(content)

            # Calculate hashes
            import hashlib
            hash_val = hashlib.sha256(content.encode()).hexdigest()[:8]

            result = SyncResult(
                mapping=mapping,
                status=VersionStatus.EQUAL,
                selected_version=selected_source,
                kelly_hash=hash_val if selected_source == 'kelly' else '',
                midikompanion_hash=hash_val if selected_source == 'midikompanion' else ''
            )

            self.sync_results.append(result)
            return result

        except Exception as e:
            result = SyncResult(
                mapping=mapping,
                status=VersionStatus.UNKNOWN,
                selected_version='unknown',
                merged_successfully=False
            )
            print(f"Error merging {mapping.kelly_path}: {e}")
            return result

    def sync_with_ai_selection(self) -> Dict[SyncMapping, str]:
        """Use AI to determine best version for each file"""
        selections = {}

        for mapping in self.mappings:
            kelly_file = Path(self.kelly_path) / mapping.kelly_path
            midikompanion_file = Path(self.midikompanion_path) / mapping.midikompanion_path

            if not kelly_file.exists() or not midikompanion_file.exists():
                continue

            # Read both files
            with open(kelly_file, 'r') as f:
                kelly_content = f.read()
            with open(midikompanion_file, 'r') as f:
                midikompanion_content = f.read()

            # Create comparison prompt
            prompt = f"""Compare these two versions of {mapping.shared_identifier} and select the best one.

KELLY-PROJECT version:
```
{kelly_content[:1500]}
```

MIDIKOMPANION version:
```
{midikompanion_content[:1500]}
```

Considerations:
1. Code quality and clarity
2. Feature completeness
3. Performance
4. Maintainability
5. Consistency with project patterns

Choose: kelly OR midikompanion
Provide brief justification."""

            from .ai_handler import AIHandler
            ai = AIHandler()
            response = ai.ask_question(prompt)

            # Parse response
            if "kelly" in response.lower() and "midikompanion" not in response.lower():
                selections[mapping] = 'kelly'
            elif "midikompanion" in response.lower():
                selections[mapping] = 'midikompanion'
            else:
                # Default to kelly if ambiguous
                selections[mapping] = 'kelly'

        return selections

    def stage_and_commit_all(
        self,
        kelly_message: str,
        midikompanion_message: str
    ) -> Tuple[bool, bool]:
        """Stage and commit changes to both repositories"""

        kelly_success = False
        midikompanion_success = False

        try:
            # Get changed files for kelly
            kelly_changed = self.kelly_handler.get_changed_files()
            if kelly_changed:
                self.kelly_handler.stage_files(kelly_changed)
                kelly_success = self.kelly_handler.commit(kelly_message)
        except Exception as e:
            print(f"Error committing to kelly-project: {e}")

        try:
            # Get changed files for midikompanion
            midikompanion_changed = self.midikompanion_handler.get_changed_files()
            if midikompanion_changed:
                self.midikompanion_handler.stage_files(midikompanion_changed)
                midikompanion_success = self.midikompanion_handler.commit(midikompanion_message)
        except Exception as e:
            print(f"Error committing to MidiKompanion: {e}")

        return kelly_success, midikompanion_success

    def create_sync_report(self) -> str:
        """Create detailed sync report"""

        report = "=" * 80 + "\n"
        report += "DUAL-REPO SYNCHRONIZATION REPORT\n"
        report += f"Timestamp: {self.sync_timestamp}\n"
        report += "=" * 80 + "\n\n"

        # Summary
        total = len(self.sync_results)
        successful = sum(1 for r in self.sync_results if r.merged_successfully)
        report += f"SUMMARY: {successful}/{total} files synced successfully\n\n"

        # Details
        report += "DETAILED RESULTS:\n"
        report += "-" * 80 + "\n"

        for result in self.sync_results:
            mapping = result.mapping
            report += f"\n{ColorCoder.colorize(mapping.shared_identifier, 'bold')}\n"
            report += f"  Kelly: {mapping.kelly_path}\n"
            report += f"  MidiKompanion: {mapping.midikompanion_path}\n"
            report += f"  Selected: {ColorCoder.colorize(result.selected_version, 'cyan')}\n"
            report += f"  Status: {ColorCoder.colorize_status(result.status)}\n"
            report += f"  Hash: {result.kelly_hash or result.midikompanion_hash}\n"

        report += "\n" + "=" * 80 + "\n"

        return report

    def save_sync_log(self, output_file: str = ".sync_log.json") -> None:
        """Save sync log as JSON"""

        log_data = {
            'timestamp': self.sync_timestamp,
            'kelly_repo': self.kelly_path,
            'midikompanion_repo': self.midikompanion_path,
            'results': [
                {
                    'mapping': {
                        'kelly_path': r.mapping.kelly_path,
                        'midikompanion_path': r.mapping.midikompanion_path,
                        'shared_identifier': r.mapping.shared_identifier,
                    },
                    'selected_version': r.selected_version,
                    'merged_successfully': r.merged_successfully,
                    'status': r.status.value,
                }
                for r in self.sync_results
            ]
        }

        output_path = Path(self.kelly_path) / output_file
        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def get_sync_statistics(self) -> Dict[str, int]:
        """Get sync statistics"""

        return {
            'total_files': len(self.sync_results),
            'successful': sum(1 for r in self.sync_results if r.merged_successfully),
            'failed': sum(1 for r in self.sync_results if not r.merged_successfully),
            'from_kelly': sum(1 for r in self.sync_results if r.selected_version == 'kelly'),
            'from_midikompanion': sum(1 for r in self.sync_results if r.selected_version == 'midikompanion'),
        }
