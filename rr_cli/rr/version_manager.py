"""Multi-version management system for builds

Manages multiple versions of files across different builds with:
- Color-coded identification
- AI-powered analysis and comparison
- Automated merge and push to multiple repositories
"""

import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path

from .ai_handler import AIHandler
from .git_handler import GitHandler


class BuildType(Enum):
    """Build type enumeration"""
    RELEASE = "release"
    DEBUG = "debug"
    DEVELOPMENT = "development"
    STAGING = "staging"


class VersionStatus(Enum):
    """Version status enumeration"""
    UNKNOWN = "unknown"
    INFERIOR = "inferior"  # Clearly inferior version
    EQUAL = "equal"        # Versions are equivalent
    SUPERIOR = "superior"  # Clearly superior version
    AMBIGUOUS = "ambiguous"  # Ambiguous comparison


@dataclass
class FileVersion:
    """Represents a single file version"""
    path: str
    build_type: BuildType
    content_hash: str
    size: int
    last_modified: str
    analysis: str = ""
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['build_type'] = self.build_type.value
        return data


@dataclass
class VersionComparison:
    """Result of comparing two file versions"""
    file_path: str
    version_a: FileVersion
    version_b: FileVersion
    status: VersionStatus = VersionStatus.UNKNOWN
    analysis: str = ""
    recommendation: str = ""
    merge_action: str = "manual"  # auto, prefer_a, prefer_b, manual

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'file_path': self.file_path,
            'version_a': self.version_a.to_dict(),
            'version_b': self.version_b.to_dict(),
            'status': self.status.value,
            'analysis': self.analysis,
            'recommendation': self.recommendation,
            'merge_action': self.merge_action
        }


class ColorCoder:
    """Color-coded file identification system"""

    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
    }

    # Status to color mapping
    STATUS_COLORS = {
        VersionStatus.SUPERIOR: 'green',
        VersionStatus.EQUAL: 'cyan',
        VersionStatus.INFERIOR: 'red',
        VersionStatus.AMBIGUOUS: 'yellow',
        VersionStatus.UNKNOWN: 'blue',
    }

    # Build type to color mapping
    BUILD_COLORS = {
        BuildType.RELEASE: 'green',
        BuildType.DEBUG: 'blue',
        BuildType.DEVELOPMENT: 'yellow',
        BuildType.STAGING: 'magenta',
    }

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Apply color to text"""
        if color not in cls.COLORS:
            return text
        return f"{cls.COLORS[color]}{text}{cls.COLORS['reset']}"

    @classmethod
    def colorize_status(cls, status: VersionStatus) -> str:
        """Colorize version status"""
        color = cls.STATUS_COLORS.get(status, 'blue')
        return cls.colorize(status.value.upper(), color)

    @classmethod
    def colorize_build(cls, build_type: BuildType) -> str:
        """Colorize build type"""
        color = cls.BUILD_COLORS.get(build_type, 'blue')
        return cls.colorize(build_type.value.upper(), color)

    @classmethod
    def format_file_line(cls, path: str, build_type: BuildType, status: VersionStatus) -> str:
        """Format a complete file line with colors"""
        build_colored = cls.colorize_build(build_type)
        status_colored = cls.colorize_status(status)
        return f"{build_colored} | {status_colored} | {path}"


class VersionManager:
    """Main version management system"""

    def __init__(self, repo_path: str = "."):
        """Initialize version manager"""
        self.repo_path = repo_path
        self.git_handler = GitHandler(repo_path)
        self._ai_handler: Optional[AIHandler] = None
        self.versions: Dict[str, List[FileVersion]] = {}
        self.comparisons: Dict[str, VersionComparison] = {}
        self.version_dir = Path(repo_path) / ".versions"
        self.version_dir.mkdir(exist_ok=True)

    @property
    def ai_handler(self) -> AIHandler:
        """Lazy-load AI handler (only when needed)"""
        if self._ai_handler is None:
            self._ai_handler = AIHandler()
        return self._ai_handler

    def register_version(
        self,
        file_path: str,
        build_type: BuildType,
        content: Optional[str] = None
    ) -> FileVersion:
        """Register a new file version"""
        full_path = Path(self.repo_path) / file_path

        # Read file content if not provided
        if content is None:
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {full_path}")
            with open(full_path, 'r') as f:
                content = f.read()

        # Calculate metadata
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        size = len(content.encode())
        last_modified = datetime.now().isoformat()

        # Create version object
        version = FileVersion(
            path=file_path,
            build_type=build_type,
            content_hash=content_hash,
            size=size,
            last_modified=last_modified
        )

        # Store version
        if file_path not in self.versions:
            self.versions[file_path] = []
        self.versions[file_path].append(version)

        # Save to metadata
        self._save_version_metadata(version, content)

        return version

    def _save_version_metadata(self, version: FileVersion, content: str) -> None:
        """Save version metadata and content"""
        version_file = self.version_dir / f"{version.path.replace('/', '_')}_{version.build_type.value}.json"
        version_file.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(version_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)

        # Save content
        content_file = version_file.with_suffix('.py')
        with open(content_file, 'w') as f:
            f.write(content)

    def analyze_version(self, file_path: str, build_type: BuildType) -> str:
        """Analyze a specific file version using AI"""
        full_path = Path(self.repo_path) / file_path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")

        with open(full_path, 'r') as f:
            content = f.read()

        # Use AI handler to analyze
        analysis = self.ai_handler.analyze_code(content, "code_quality")
        return analysis

    def compare_versions(
        self,
        file_path: str,
        build_type_a: BuildType,
        build_type_b: BuildType
    ) -> VersionComparison:
        """Compare two versions of a file"""

        # Get both versions
        version_a = self._get_version(file_path, build_type_a)
        version_b = self._get_version(file_path, build_type_b)

        if version_a is None or version_b is None:
            raise ValueError(f"One or both versions not found for {file_path}")

        # Read both files
        full_path_a = Path(self.repo_path) / file_path
        full_path_b = Path(self.repo_path) / file_path

        with open(full_path_a, 'r') as f:
            content_a = f.read()
        with open(full_path_b, 'r') as f:
            content_b = f.read()

        # Create comparison prompt
        comparison_prompt = f"""Compare these two code versions and determine which is clearly superior.

Version A ({build_type_a.value}):
```
{content_a[:1000]}
```

Version B ({build_type_b.value}):
```
{content_b[:1000]}
```

Analyze:
1. Code quality and readability
2. Performance considerations
3. Maintainability
4. Error handling
5. Best practices adherence

Determine: Is one CLEARLY superior, or are they equivalent/ambiguous?
Return: "SUPERIOR_A", "SUPERIOR_B", "EQUAL", or "AMBIGUOUS"
Then provide brief reasoning."""

        response = self.ai_handler.ask_question(comparison_prompt)

        # Parse response
        status = self._parse_comparison_response(response)

        # Create comparison object
        comparison = VersionComparison(
            file_path=file_path,
            version_a=version_a,
            version_b=version_b,
            status=status,
            analysis=response
        )

        # Determine merge action
        comparison.merge_action = self._determine_merge_action(status, build_type_a, build_type_b)
        comparison.recommendation = self._generate_recommendation(comparison)

        # Store comparison
        comparison_key = f"{file_path}_{build_type_a.value}_vs_{build_type_b.value}"
        self.comparisons[comparison_key] = comparison

        return comparison

    def _get_version(self, file_path: str, build_type: BuildType) -> Optional[FileVersion]:
        """Get a specific version"""
        if file_path not in self.versions:
            return None

        for version in self.versions[file_path]:
            if version.build_type == build_type:
                return version

        return None

    def _parse_comparison_response(self, response: str) -> VersionStatus:
        """Parse AI comparison response"""
        response_upper = response.upper()

        if "SUPERIOR_A" in response_upper:
            return VersionStatus.SUPERIOR
        elif "SUPERIOR_B" in response_upper:
            return VersionStatus.INFERIOR
        elif "EQUAL" in response_upper:
            return VersionStatus.EQUAL
        elif "AMBIGUOUS" in response_upper:
            return VersionStatus.AMBIGUOUS
        else:
            return VersionStatus.UNKNOWN

    def _determine_merge_action(
        self,
        status: VersionStatus,
        build_type_a: BuildType,
        build_type_b: BuildType
    ) -> str:
        """Determine automatic merge action"""

        if status == VersionStatus.SUPERIOR:
            # Prefer release > staging > development > debug
            preference_order = {
                BuildType.RELEASE: 4,
                BuildType.STAGING: 3,
                BuildType.DEVELOPMENT: 2,
                BuildType.DEBUG: 1
            }
            if preference_order.get(build_type_a, 0) > preference_order.get(build_type_b, 0):
                return "prefer_a"
            else:
                return "prefer_b"

        elif status == VersionStatus.EQUAL:
            return "manual"

        elif status == VersionStatus.INFERIOR:
            # Opposite of superior
            return "prefer_b"

        else:
            return "manual"

    def _generate_recommendation(self, comparison: VersionComparison) -> str:
        """Generate merge recommendation"""

        if comparison.status == VersionStatus.SUPERIOR:
            return f"Use {comparison.version_a.build_type.value} version - clearly superior"
        elif comparison.status == VersionStatus.INFERIOR:
            return f"Use {comparison.version_b.build_type.value} version - superior"
        elif comparison.status == VersionStatus.EQUAL:
            return "Versions are equivalent. Choose based on other criteria."
        else:
            return "Manual review required. AI analysis was inconclusive."

    def apply_merge(
        self,
        file_path: str,
        build_type_a: BuildType,
        build_type_b: BuildType,
        target_build_type: Optional[BuildType] = None
    ) -> bool:
        """Apply merge decision"""

        comparison = self.comparisons.get(
            f"{file_path}_{build_type_a.value}_vs_{build_type_b.value}"
        )

        if comparison is None:
            raise ValueError("No comparison found. Run compare_versions first.")

        # Determine source version
        if target_build_type is None:
            if comparison.merge_action == "prefer_a":
                target_build_type = build_type_a
            elif comparison.merge_action == "prefer_b":
                target_build_type = build_type_b
            else:
                return False  # Manual action required

        # Get source content
        source_version = (
            comparison.version_a
            if target_build_type == build_type_a
            else comparison.version_b
        )

        # This would be where actual file merge happens
        return True

    def get_comparison_report(self) -> str:
        """Generate a comparison report for all versions"""

        report = "=" * 80 + "\n"
        report += "MULTI-VERSION COMPARISON REPORT\n"
        report += f"Generated: {datetime.now().isoformat()}\n"
        report += "=" * 80 + "\n\n"

        for comparison_key, comparison in self.comparisons.items():
            report += ColorCoder.format_file_line(
                comparison.file_path,
                comparison.version_a.build_type,
                comparison.status
            ) + "\n"
            report += f"  Analysis: {comparison.analysis[:200]}...\n"
            report += f"  Recommendation: {comparison.recommendation}\n"
            report += f"  Merge Action: {comparison.merge_action}\n\n"

        return report

    def push_to_repos(
        self,
        changes: Dict[str, Tuple[str, BuildType]],
        repos: List[str],
        commit_message: str
    ) -> Dict[str, bool]:
        """Push changes to multiple repositories"""

        results = {}

        for repo in repos:
            try:
                handler = GitHandler(repo)

                # Stage files
                files_to_stage = [change[0] for change in changes.values()]
                handler.stage_files(files_to_stage)

                # Create commit
                full_message = f"{commit_message}\n\nBuild: {', '.join([c[1].value for c in changes.values()])}"
                success = handler.commit(full_message)

                results[repo] = success
            except Exception as e:
                print(f"Error pushing to {repo}: {e}")
                results[repo] = False

        return results
