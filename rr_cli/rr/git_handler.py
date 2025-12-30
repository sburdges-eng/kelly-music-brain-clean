"""Git integration module for RR CLI"""

from git import GitCommandError, Repo
from typing import Any


class GitHandler:
    """Handle git operations"""

    def __init__(self, repo_path: str = "."):
        """Initialize git handler with a repository path"""
        try:
            self.repo = Repo(repo_path)
        except Exception as e:
            msg = f"Invalid git repository: {repo_path}"
            raise ValueError(msg) from e

    def get_status(self) -> dict[str, Any]:
        """Get current git status"""
        return {
            "branch": self.repo.active_branch.name,
            "dirty": self.repo.is_dirty(),
            "untracked": list(self.repo.untracked_files),
            "staged": [item.a_path for item in self.repo.index.diff("HEAD")],
            "unstaged": [item.a_path for item in self.repo.index.diff(None)],
        }

    def get_diff(self, staged: bool = False) -> Any:
        """Get git diff output"""
        if staged:
            diff_index = self.repo.index.diff("HEAD")
        else:
            diff_index = self.repo.index.diff(None)

        if not diff_index:
            return ""

        # Use git command for proper diff output
        if staged:
            return self.repo.git.diff("--cached")
        return self.repo.git.diff()

    def get_recent_commits(self, count: int = 5) -> list[dict[str, Any]]:
        """Get recent commit history"""
        commits = []
        for commit in list(self.repo.iter_commits())[:count]:
            commits.append({
                "hash": commit.hexsha[:7],
                "author": commit.author.name or "Unknown",
                "message": commit.message.strip(),
                "date": commit.committed_datetime.isoformat(),
            })
        return commits

    def stage_files(self, files: list[str]) -> bool:
        """Stage specified files"""
        try:
            self.repo.index.add(files)
        except GitCommandError as e:
            print(f"Error staging files: {e}")
            return False
        return True

    def commit(self, message: str) -> bool:
        """Create a commit with the given message"""
        try:
            self.repo.index.commit(message)
        except GitCommandError as e:
            print(f"Error creating commit: {e}")
            return False
        return True

    def get_changed_files(self) -> list[str]:
        """Get list of changed files"""
        changed: list[str] = []
        for item in self.repo.index.diff("HEAD"):
            if item.a_path:
                changed.append(item.a_path)
        for item in self.repo.index.diff(None):
            if item.a_path:
                changed.append(item.a_path)
        return list(set(changed))
