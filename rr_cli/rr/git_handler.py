"""Git integration module for RR CLI"""

import os
from typing import List, Dict, Optional
from git import Repo, GitCommandError
from pathlib import Path


class GitHandler:
    """Handle git operations"""

    def __init__(self, repo_path: str = "."):
        """Initialize git handler with a repository path"""
        try:
            self.repo = Repo(repo_path)
        except Exception as e:
            raise ValueError(f"Invalid git repository: {repo_path}") from e

    def get_status(self) -> Dict[str, any]:
        """Get current git status"""
        return {
            "branch": self.repo.active_branch.name,
            "dirty": self.repo.is_dirty(),
            "untracked": [f.a_path for f in self.repo.untracked_files],
            "staged": [item[0] for item in self.repo.index.diff("HEAD")],
            "unstaged": [item[0] for item in self.repo.index.diff(None)],
        }

    def get_diff(self, staged: bool = False) -> str:
        """Get git diff output"""
        if staged:
            diff_index = self.repo.index.diff("HEAD")
        else:
            diff_index = self.repo.index.diff(None)

        diff_str = ""
        for item in diff_index:
            diff_str += f"\n{item.a_path}\n"
        return diff_str

    def get_recent_commits(self, count: int = 5) -> List[Dict[str, str]]:
        """Get recent commit history"""
        commits = []
        for commit in list(self.repo.iter_commits())[:count]:
            commits.append({
                "hash": commit.hexsha[:7],
                "author": commit.author.name,
                "message": commit.message.strip(),
                "date": commit.committed_datetime.isoformat(),
            })
        return commits

    def stage_files(self, files: List[str]) -> bool:
        """Stage specified files"""
        try:
            self.repo.index.add(files)
            return True
        except GitCommandError as e:
            print(f"Error staging files: {e}")
            return False

    def commit(self, message: str) -> bool:
        """Create a commit with the given message"""
        try:
            self.repo.index.commit(message)
            return True
        except GitCommandError as e:
            print(f"Error creating commit: {e}")
            return False

    def get_changed_files(self) -> List[str]:
        """Get list of changed files"""
        changed = []
        for item in self.repo.index.diff("HEAD"):
            changed.append(item.a_path)
        for item in self.repo.index.diff(None):
            changed.append(item.a_path)
        return list(set(changed))
