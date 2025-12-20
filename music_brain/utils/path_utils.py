"""
Cross-platform Safe File Path Utilities

Provides safe, cross-platform file path handling for Windows, macOS, and Linux.
Handles Unicode, case sensitivity, long paths, and forbidden characters.

This module implements P0-004 from the comprehensive system requirements.
"""

import os
import sys
import platform
from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from typing import Union, Optional, List
import re


# Platform detection
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# Windows forbidden characters
WINDOWS_FORBIDDEN_CHARS = '<>:"|?*'
WINDOWS_FORBIDDEN_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

# Windows long path prefix
WINDOWS_LONG_PATH_PREFIX = "\\\\?\\"


class PathError(Exception):
    """Exception raised for path-related errors."""
    pass


class CrossPlatformPath:
    """
    Cross-platform safe path handler.
    
    Provides normalized, validated paths that work across Windows, macOS, and Linux.
    """
    
    def __init__(self, path: Union[str, Path, "CrossPlatformPath"]):
        """
        Initialize a cross-platform path.
        
        Args:
            path: Path string, Path object, or CrossPlatformPath instance
            
        Raises:
            PathError: If path is invalid or contains forbidden characters
        """
        if isinstance(path, CrossPlatformPath):
            self._path = path._path
        else:
            self._path = self._normalize(path)
            self._validate()
    
    def _normalize(self, path: Union[str, Path]) -> Path:
        """Normalize path for current platform."""
        # Convert to string if Path object
        path_str = str(path) if isinstance(path, Path) else path
        
        # Check for null bytes early (before Path operations)
        if '\x00' in path_str:
            raise PathError(f"Path contains null byte: {path_str}")
        
        # Handle Windows long paths
        if IS_WINDOWS and path_str.startswith(WINDOWS_LONG_PATH_PREFIX):
            path_str = path_str[len(WINDOWS_LONG_PATH_PREFIX):]
        
        # Expand user directory
        path_str = os.path.expanduser(path_str)
        
        # Expand environment variables
        path_str = os.path.expandvars(path_str)
        
        # Resolve relative paths
        if not os.path.isabs(path_str):
            path_str = os.path.abspath(path_str)
        
        # Normalize separators and resolve
        path_obj = Path(path_str).resolve()
        
        return path_obj
    
    def _validate(self) -> None:
        """Validate path for current platform."""
        path_str = str(self._path)
        
        # Check for null bytes (already checked in _normalize, but double-check)
        if '\x00' in path_str:
            raise PathError(f"Path contains null byte: {path_str}")
        
        # Check for forbidden characters (Windows)
        if IS_WINDOWS:
            for char in WINDOWS_FORBIDDEN_CHARS:
                if char in path_str:
                    raise PathError(
                        f"Path contains forbidden character '{char}': {path_str}"
                    )
            
            # Check for forbidden names (Windows)
            for part in self._path.parts:
                name = part.upper()
                if name in WINDOWS_FORBIDDEN_NAMES:
                    raise PathError(
                        f"Path contains forbidden name '{part}': {path_str}"
                    )
    
    def sanitize_filename(self, filename: str, replacement: str = "_") -> str:
        """
        Sanitize a filename for the current platform.
        
        Args:
            filename: Original filename
            replacement: Character to replace forbidden characters with
            
        Returns:
            Sanitized filename safe for current platform
        """
        sanitized = filename
        
        # Remove null bytes first (all platforms)
        sanitized = sanitized.replace('\x00', replacement)
        
        # Remove or replace forbidden characters
        # Always sanitize Windows forbidden chars for cross-platform compatibility
        for char in WINDOWS_FORBIDDEN_CHARS:
            sanitized = sanitized.replace(char, replacement)
        
        # Remove trailing dots and spaces (Windows doesn't allow these)
        # Also sanitize for cross-platform safety
        sanitized = sanitized.rstrip('. ')
        
        # Check for forbidden names (Windows)
        # Always check for cross-platform safety
        name_upper = sanitized.upper()
        if name_upper in WINDOWS_FORBIDDEN_NAMES:
            sanitized = f"{sanitized}_{replacement}"
        
        # Limit length (Windows MAX_PATH is 260, but we'll be more conservative)
        max_length = 255
        if len(sanitized) > max_length:
            name, ext = os.path.splitext(sanitized)
            max_name_length = max_length - len(ext)
            sanitized = name[:max_name_length] + ext
        
        return sanitized
    
    def to_string(self, long_path: bool = False) -> str:
        """
        Convert to string representation.
        
        Args:
            long_path: If True and on Windows, use long path prefix
            
        Returns:
            String representation of path
        """
        path_str = str(self._path)
        
        if IS_WINDOWS and long_path:
            # Use long path prefix for Windows paths > 260 chars
            if len(path_str) > 260:
                if not path_str.startswith(WINDOWS_LONG_PATH_PREFIX):
                    # Convert to UNC path if needed
                    if path_str.startswith("\\\\"):
                        path_str = WINDOWS_LONG_PATH_PREFIX + "UNC" + path_str[1:]
                    else:
                        path_str = WINDOWS_LONG_PATH_PREFIX + path_str
        
        return path_str
    
    def to_path(self) -> Path:
        """Convert to standard Path object."""
        return self._path
    
    def exists(self) -> bool:
        """Check if path exists."""
        return self._path.exists()
    
    def is_file(self) -> bool:
        """Check if path is a file."""
        return self._path.is_file()
    
    def is_dir(self) -> bool:
        """Check if path is a directory."""
        return self._path.is_dir()
    
    def parent(self) -> "CrossPlatformPath":
        """Get parent directory."""
        return CrossPlatformPath(self._path.parent)
    
    def join(self, *parts: Union[str, Path]) -> "CrossPlatformPath":
        """
        Join path with additional parts.
        
        Args:
            *parts: Additional path components
            
        Returns:
            New CrossPlatformPath with joined components
        """
        new_path = self._path
        for part in parts:
            new_path = new_path / str(part)
        return CrossPlatformPath(new_path)
    
    def ensure_dir(self, parents: bool = True, exist_ok: bool = True) -> None:
        """
        Ensure directory exists, creating if necessary.
        
        Args:
            parents: Create parent directories if needed
            exist_ok: Don't raise error if directory already exists
        """
        if self._path.is_dir():
            return
        
        self._path.mkdir(parents=parents, exist_ok=exist_ok)
    
    def __str__(self) -> str:
        """String representation."""
        return str(self._path)
    
    def __repr__(self) -> str:
        """Representation."""
        return f"CrossPlatformPath({self._path!r})"
    
    def __truediv__(self, other: Union[str, Path]) -> "CrossPlatformPath":
        """Support / operator for path joining."""
        return self.join(other)
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if isinstance(other, CrossPlatformPath):
            return self._path == other._path
        elif isinstance(other, (str, Path)):
            return self._path == Path(other)
        return False
    
    def __hash__(self) -> int:
        """Hash support."""
        return hash(self._path)


def safe_path(path: Union[str, Path]) -> CrossPlatformPath:
    """
    Create a safe cross-platform path.
    
    Convenience function for creating CrossPlatformPath instances.
    
    Args:
        path: Path string or Path object
        
    Returns:
        CrossPlatformPath instance
    """
    return CrossPlatformPath(path)


def safe_filename(filename: str, replacement: str = "_") -> str:
    """
    Sanitize a filename for the current platform.
    
    Convenience function for sanitizing filenames.
    
    Args:
        filename: Original filename
        replacement: Character to replace forbidden characters with
        
    Returns:
        Sanitized filename
    """
    return CrossPlatformPath(".").sanitize_filename(filename, replacement)


def ensure_path_exists(path: Union[str, Path], is_file: bool = False) -> CrossPlatformPath:
    """
    Ensure a path exists, creating directories if necessary.
    
    Args:
        path: Path to ensure exists
        is_file: If True, ensure parent directory exists (for file paths)
        
    Returns:
        CrossPlatformPath instance
    """
    safe = CrossPlatformPath(path)
    
    if is_file:
        safe.parent().ensure_dir()
    else:
        safe.ensure_dir()
    
    return safe


def get_home_dir() -> CrossPlatformPath:
    """Get user home directory as CrossPlatformPath."""
    return CrossPlatformPath(Path.home())


def get_temp_dir() -> CrossPlatformPath:
    """Get system temp directory as CrossPlatformPath."""
    return CrossPlatformPath(Path(tempfile.gettempdir()) if 'tempfile' in sys.modules else Path("/tmp"))


# Import tempfile for get_temp_dir
import tempfile
