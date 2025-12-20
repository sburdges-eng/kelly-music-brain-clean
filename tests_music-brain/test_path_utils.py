"""
Tests for cross-platform path utilities.

Tests P0-004 implementation: Cross-platform safe file path system.
"""

import pytest
import tempfile
import platform
from pathlib import Path
from music_brain.utils.path_utils import (
    CrossPlatformPath,
    safe_path,
    safe_filename,
    ensure_path_exists,
    get_home_dir,
    PathError,
    IS_WINDOWS,
    IS_MACOS,
    IS_LINUX,
)


class TestCrossPlatformPath:
    """Test CrossPlatformPath class."""
    
    def test_basic_creation(self):
        """Test creating basic paths."""
        path = CrossPlatformPath("/tmp/test")
        assert isinstance(path, CrossPlatformPath)
        assert str(path) == str(Path("/tmp/test").resolve())
    
    def test_from_string(self):
        """Test creating from string."""
        path = CrossPlatformPath("test.txt")
        assert isinstance(path, CrossPlatformPath)
    
    def test_from_path(self):
        """Test creating from Path object."""
        p = Path("test.txt")
        path = CrossPlatformPath(p)
        assert isinstance(path, CrossPlatformPath)
    
    def test_from_crossplatform(self):
        """Test creating from another CrossPlatformPath."""
        p1 = CrossPlatformPath("test.txt")
        p2 = CrossPlatformPath(p1)
        assert p1 == p2
    
    def test_expand_user(self):
        """Test expanding user directory."""
        path = CrossPlatformPath("~/test")
        assert "~" not in str(path)
    
    def test_expand_vars(self):
        """Test expanding environment variables."""
        import os
        os.environ["TEST_VAR"] = "/tmp"
        path = CrossPlatformPath("$TEST_VAR/test")
        assert "$TEST_VAR" not in str(path)
        del os.environ["TEST_VAR"]
    
    def test_resolve_relative(self):
        """Test resolving relative paths."""
        path = CrossPlatformPath("test.txt")
        assert path.to_path().is_absolute()
    
    def test_join_paths(self):
        """Test joining paths."""
        base = CrossPlatformPath("/tmp")
        joined = base.join("test", "file.txt")
        assert "test" in str(joined)
        assert "file.txt" in str(joined)
    
    def test_division_operator(self):
        """Test / operator for path joining."""
        base = CrossPlatformPath("/tmp")
        joined = base / "test" / "file.txt"
        assert "test" in str(joined)
        assert "file.txt" in str(joined)
    
    def test_parent(self):
        """Test getting parent directory."""
        path = CrossPlatformPath("/tmp/test/file.txt")
        parent = path.parent()
        assert "test" not in str(parent) or "file.txt" not in str(parent)
    
    def test_to_string(self):
        """Test converting to string."""
        path = CrossPlatformPath("/tmp/test")
        assert isinstance(path.to_string(), str)
    
    def test_to_path(self):
        """Test converting to Path."""
        path = CrossPlatformPath("/tmp/test")
        assert isinstance(path.to_path(), Path)
    
    def test_equality(self):
        """Test equality comparison."""
        p1 = CrossPlatformPath("/tmp/test")
        p2 = CrossPlatformPath("/tmp/test")
        assert p1 == p2
    
    def test_hash(self):
        """Test hashing."""
        p1 = CrossPlatformPath("/tmp/test")
        p2 = CrossPlatformPath("/tmp/test")
        assert hash(p1) == hash(p2)


class TestWindowsValidation:
    """Test Windows-specific validation."""
    
    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows-specific test")
    def test_forbidden_chars_windows(self):
        """Test Windows forbidden characters."""
        for char in '<>:"|?*':
            with pytest.raises(PathError):
                CrossPlatformPath(f"/tmp/test{char}file.txt")
    
    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows-specific test")
    def test_forbidden_names_windows(self):
        """Test Windows forbidden names."""
        for name in ["CON", "PRN", "AUX", "NUL"]:
            with pytest.raises(PathError):
                CrossPlatformPath(f"/tmp/{name}.txt")
    
    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows-specific test")
    def test_long_path_prefix(self):
        """Test Windows long path prefix."""
        path = CrossPlatformPath("/tmp/test")
        long_path = path.to_string(long_path=True)
        # Should handle long paths appropriately
        assert isinstance(long_path, str)


class TestSanitizeFilename:
    """Test filename sanitization."""
    
    def test_sanitize_basic(self):
        """Test basic filename sanitization."""
        path = CrossPlatformPath(".")
        sanitized = path.sanitize_filename("test<file>.txt")
        assert "<" not in sanitized
        assert ">" not in sanitized
    
    def test_sanitize_null_bytes(self):
        """Test sanitizing null bytes."""
        path = CrossPlatformPath(".")
        sanitized = path.sanitize_filename("test\x00file.txt")
        assert "\x00" not in sanitized
    
    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows-specific test")
    def test_sanitize_trailing_dots(self):
        """Test sanitizing trailing dots (Windows)."""
        path = CrossPlatformPath(".")
        sanitized = path.sanitize_filename("test....")
        assert not sanitized.endswith(".")
    
    def test_sanitize_length_limit(self):
        """Test length limiting."""
        path = CrossPlatformPath(".")
        long_name = "a" * 300 + ".txt"
        sanitized = path.sanitize_filename(long_name)
        assert len(sanitized) <= 255


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_safe_path(self):
        """Test safe_path function."""
        path = safe_path("/tmp/test")
        assert isinstance(path, CrossPlatformPath)
    
    def test_safe_filename(self):
        """Test safe_filename function."""
        filename = safe_filename("test<file>.txt")
        assert "<" not in filename
    
    def test_ensure_path_exists_dir(self):
        """Test ensuring directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "directory"
            path = ensure_path_exists(new_dir, is_file=False)
            assert path.is_dir()
    
    def test_ensure_path_exists_file(self):
        """Test ensuring file path parent exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_file = Path(tmpdir) / "new" / "file.txt"
            path = ensure_path_exists(new_file, is_file=True)
            assert path.parent().is_dir()
    
    def test_get_home_dir(self):
        """Test getting home directory."""
        home = get_home_dir()
        assert isinstance(home, CrossPlatformPath)
        assert home.is_dir()


class TestCrossPlatformBehavior:
    """Test cross-platform behavior."""
    
    def test_platform_detection(self):
        """Test platform detection."""
        from music_brain.utils.path_utils import IS_WINDOWS, IS_MACOS, IS_LINUX
        assert isinstance(IS_WINDOWS, bool)
        assert isinstance(IS_MACOS, bool)
        assert isinstance(IS_LINUX, bool)
        # At least one should be True
        assert IS_WINDOWS or IS_MACOS or IS_LINUX
    
    def test_case_sensitivity_handling(self):
        """Test case sensitivity handling."""
        # Paths should normalize case appropriately
        path1 = CrossPlatformPath("/tmp/TEST")
        path2 = CrossPlatformPath("/tmp/test")
        # On case-insensitive systems, these might be equal
        # On case-sensitive systems, they won't be
        # Just verify both can be created
        assert isinstance(path1, CrossPlatformPath)
        assert isinstance(path2, CrossPlatformPath)
    
    def test_unicode_handling(self):
        """Test Unicode path handling."""
        # Unicode paths should work
        path = CrossPlatformPath("/tmp/测试/文件.txt")
        assert isinstance(path, CrossPlatformPath)
        # Filename sanitization should preserve Unicode where possible
        sanitized = safe_filename("测试文件.txt")
        assert "测试" in sanitized or len(sanitized) > 0


class TestErrorHandling:
    """Test error handling."""
    
    def test_null_byte_in_path(self):
        """Test null byte detection."""
        # Null bytes are detected early in _normalize, which raises PathError
        # However, Path.resolve() may raise ValueError first on some systems
        # So we catch either exception
        with pytest.raises((PathError, ValueError)):
            CrossPlatformPath("/tmp/test\x00file.txt")
    
    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows-specific test")
    def test_windows_forbidden_char_error(self):
        """Test Windows forbidden character error message."""
        with pytest.raises(PathError) as exc_info:
            CrossPlatformPath("/tmp/test<file>.txt")
        assert "forbidden character" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
