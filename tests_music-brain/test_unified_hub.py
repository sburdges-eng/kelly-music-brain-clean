"""
Tests for the unified_hub module.

Tests LocalVoiceSynth TTS functionality across platforms.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestLocalVoiceSynth:
    """Test LocalVoiceSynth text-to-speech functionality."""

    def test_import_local_voice_synth(self):
        """Test that LocalVoiceSynth can be imported."""
        from music_brain.agents.unified_hub import LocalVoiceSynth
        assert LocalVoiceSynth is not None

    def test_platform_detection(self):
        """Test that platform detection works."""
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        assert synth._platform in ["macos", "linux", "windows", "unknown"]

    @patch("platform.system")
    def test_detect_macos_platform(self, mock_system):
        """Test macOS platform detection."""
        mock_system.return_value = "Darwin"
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        assert synth._platform == "macos"

    @patch("platform.system")
    def test_detect_linux_platform(self, mock_system):
        """Test Linux platform detection."""
        mock_system.return_value = "Linux"
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        assert synth._platform == "linux"

    @patch("platform.system")
    def test_detect_windows_platform(self, mock_system):
        """Test Windows platform detection."""
        mock_system.return_value = "Windows"
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        assert synth._platform == "windows"

    @patch("subprocess.Popen")
    def test_speak_macos_calls_say(self, mock_popen):
        """Test that macOS speak uses the 'say' command."""
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        synth._platform = "macos"

        result = synth.speak("Hello World", rate=175)

        assert result is True
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == "say"
        assert "-r" in call_args
        assert "Hello World" in call_args

    @patch("subprocess.Popen")
    def test_speak_linux_calls_espeak(self, mock_popen):
        """Test that Linux speak uses 'espeak' command."""
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        synth._platform = "linux"

        result = synth.speak("Hello World", rate=175, pitch=50)

        assert result is True
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == "espeak"
        assert "-s" in call_args
        assert "-p" in call_args
        assert "Hello World" in call_args

    @patch("subprocess.Popen")
    def test_speak_windows_calls_powershell(self, mock_popen):
        """Test that Windows speak uses PowerShell with System.Speech."""
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        synth._platform = "windows"

        result = synth.speak("Hello World", rate=175)

        assert result is True
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == "powershell"
        assert "-Command" in call_args

        # Check the PowerShell command contains the expected elements
        ps_command = call_args[2]
        assert "System.Speech" in ps_command
        assert "SpeechSynthesizer" in ps_command
        assert "Hello World" in ps_command

    @patch("subprocess.Popen")
    def test_speak_windows_escapes_single_quotes(self, mock_popen):
        """Test that Windows speak properly escapes single quotes in text."""
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        synth._platform = "windows"

        result = synth.speak("It's a test", rate=175)

        assert result is True
        call_args = mock_popen.call_args[0][0]
        ps_command = call_args[2]
        # Single quotes should be escaped as '' in PowerShell
        assert "It''s a test" in ps_command

    @patch("subprocess.Popen")
    def test_speak_windows_rate_mapping(self, mock_popen):
        """Test that Windows rate is properly mapped to PowerShell rate scale."""
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        synth._platform = "windows"

        # Test with different rates
        synth.speak("Test", rate=175)  # Default, should map to rate ~0
        ps_command = mock_popen.call_args[0][0][2]
        assert "$synth.Rate = 0" in ps_command

        mock_popen.reset_mock()
        synth.speak("Test", rate=250)  # Fast, should map to positive rate
        ps_command = mock_popen.call_args[0][0][2]
        assert "$synth.Rate = 3" in ps_command

        mock_popen.reset_mock()
        synth.speak("Test", rate=100)  # Slow, should map to negative rate
        ps_command = mock_popen.call_args[0][0][2]
        assert "$synth.Rate = -3" in ps_command

    def test_speak_unknown_platform_returns_false(self):
        """Test that unknown platform returns False."""
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        synth._platform = "unknown"

        result = synth.speak("Hello World")
        assert result is False

    @patch("subprocess.Popen")
    def test_speak_handles_exception(self, mock_popen):
        """Test that speak handles exceptions gracefully."""
        from music_brain.agents.unified_hub import LocalVoiceSynth

        mock_popen.side_effect = Exception("Command not found")

        synth = LocalVoiceSynth(midi_bridge=None)
        synth._platform = "macos"

        result = synth.speak("Hello World")
        assert result is False


class TestLocalVoiceSynthVoiceProfiles:
    """Test voice profile functionality in LocalVoiceSynth."""

    def test_set_and_get_profile(self):
        """Test setting and getting the active voice profile."""
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)

        assert synth.get_profile() is None
        synth.set_profile("test_profile")
        assert synth.get_profile() == "test_profile"

    def test_list_profiles(self):
        """Test listing available voice profiles."""
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        profiles = synth.list_profiles()
        assert isinstance(profiles, list)

    def test_list_accents(self):
        """Test listing available accents."""
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        accents = synth.list_accents()
        assert isinstance(accents, list)
        assert len(accents) > 0

    def test_list_speech_patterns(self):
        """Test listing available speech patterns."""
        from music_brain.agents.unified_hub import LocalVoiceSynth

        synth = LocalVoiceSynth(midi_bridge=None)
        patterns = synth.list_speech_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0


class TestHubConfig:
    """Test HubConfig dataclass."""

    def test_hub_config_defaults(self):
        """Test HubConfig has sensible defaults."""
        from music_brain.agents.unified_hub import HubConfig

        config = HubConfig()
        assert config.osc_host == "127.0.0.1"
        assert config.osc_send_port == 9000
        assert config.osc_receive_port == 9001
        assert config.llm_model == "llama3"

    def test_hub_config_path_expansion(self):
        """Test that paths are expanded."""
        from music_brain.agents.unified_hub import HubConfig
        import os

        config = HubConfig()
        assert config.session_dir.startswith(os.path.expanduser("~"))
        assert config.config_dir.startswith(os.path.expanduser("~"))


class TestSessionConfig:
    """Test SessionConfig dataclass."""

    def test_session_config_defaults(self):
        """Test SessionConfig has sensible defaults."""
        from music_brain.agents.unified_hub import SessionConfig

        config = SessionConfig()
        assert config.name == "untitled"
        assert config.tempo == 120.0
        assert config.key == "C"
        assert config.mode == "major"

    def test_session_config_custom_values(self):
        """Test SessionConfig accepts custom values."""
        from music_brain.agents.unified_hub import SessionConfig

        config = SessionConfig(
            name="My Song",
            tempo=90.0,
            key="Am",
            mode="minor",
            emotion="grief"
        )
        assert config.name == "My Song"
        assert config.tempo == 90.0
        assert config.key == "Am"
        assert config.mode == "minor"
        assert config.emotion == "grief"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
