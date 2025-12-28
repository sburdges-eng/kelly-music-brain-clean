"""Tests for emotion sampler module."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from music_brain.emotion_sampler import (
    EmotionSampler,
    FreesoundAPI,
    EmotionHierarchy,
    SoundResult,
    DownloadedSound,
    BASE_EMOTIONS,
    INSTRUMENTS,
)


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory."""
    config_dir = tmp_path / ".kelly"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def mock_api_key(temp_config_dir):
    """Create mock API key in config file."""
    config_file = temp_config_dir / "freesound_config.json"
    config = {"freesound_api_key": "test_api_key_12345"}
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f)
    return config_file


class TestEmotionHierarchy:
    """Tests for EmotionHierarchy class."""

    def test_initialization(self):
        """Test hierarchy initializes with base emotions."""
        hierarchy = EmotionHierarchy()
        assert len(hierarchy.base_emotions) == 6
        assert "HAPPY" in hierarchy.base_emotions
        assert "SAD" in hierarchy.base_emotions

    def test_get_base_emotions(self):
        """Test getting base emotions list."""
        hierarchy = EmotionHierarchy()
        emotions = hierarchy.get_base_emotions()
        assert len(emotions) == 6
        assert emotions == BASE_EMOTIONS

    def test_normalize_emotion(self):
        """Test emotion normalization."""
        hierarchy = EmotionHierarchy()
        assert hierarchy.normalize_emotion("happy") == "HAPPY"
        assert hierarchy.normalize_emotion("HAPPY") == "HAPPY"
        assert hierarchy.normalize_emotion("Happy") == "HAPPY"
        assert hierarchy.normalize_emotion("sad") == "SAD"

    def test_normalize_non_base_emotion(self):
        """Test normalization of non-base emotion returns as-is."""
        hierarchy = EmotionHierarchy()
        result = hierarchy.normalize_emotion("melancholy")
        assert result == "melancholy"


class TestFreesoundAPI:
    """Tests for FreesoundAPI class."""

    def test_initialization_with_api_key(self):
        """Test API initialization with explicit key."""
        api = FreesoundAPI(api_key="test_key")
        assert api.api_key == "test_key"
        assert "Authorization" in api.session.headers

    def test_initialization_without_api_key(self, temp_config_dir):
        """Test API initialization without key."""
        config_file = temp_config_dir / "freesound_config.json"
        api = FreesoundAPI(config_file=config_file)
        assert api.api_key is None

    def test_load_api_key_from_config(self, mock_api_key):
        """Test loading API key from config file."""
        api = FreesoundAPI(config_file=mock_api_key)
        assert api.api_key == "test_api_key_12345"

    @patch.dict('os.environ', {'FREESOUND_API_KEY': 'env_test_key'})
    def test_load_api_key_from_environment(self):
        """Test loading API key from environment variable."""
        api = FreesoundAPI()
        assert api.api_key == "env_test_key"

    @patch('requests.Session.get')
    def test_search_success(self, mock_get):
        """Test successful search."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'results': [
                {
                    'id': 12345,
                    'name': 'happy piano',
                    'duration': 5.0,
                    'filesize': 1024000,
                    'tags': ['piano', 'happy'],
                    'previews': {'preview-hq-mp3': 'http://example.com/preview.mp3'},
                    'license': 'CC0',
                    'avg_rating': 4.5
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        api = FreesoundAPI(api_key="test_key")
        results = api.search("happy", instrument="piano")

        assert len(results) == 1
        assert results[0].id == 12345
        assert results[0].name == "happy piano"
        assert results[0].rating == 4.5

    def test_search_without_api_key(self):
        """Test search raises error without API key."""
        api = FreesoundAPI()
        with pytest.raises(ValueError, match="Freesound API key required"):
            api.search("happy")

    @patch('requests.Session.get')
    def test_search_with_instrument_filter(self, mock_get):
        """Test search includes instrument in query."""
        mock_response = Mock()
        mock_response.json.return_value = {'results': []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        api = FreesoundAPI(api_key="test_key")
        api.search("happy", instrument="guitar")

        # Verify the query includes instrument
        call_args = mock_get.call_args
        assert 'happy guitar' in call_args[1]['params']['query']

    @patch('requests.Session.get')
    @patch('requests.get')
    def test_download_preview(self, mock_download, mock_get, tmp_path):
        """Test downloading preview file."""
        # Mock sound details response
        mock_details_response = Mock()
        mock_details_response.json.return_value = {
            'previews': {'preview-hq-mp3': 'http://example.com/preview.mp3'}
        }
        mock_details_response.raise_for_status = Mock()
        mock_get.return_value = mock_details_response

        # Mock download response
        mock_download_response = Mock()
        mock_download_response.iter_content.return_value = [b'test audio data']
        mock_download_response.raise_for_status = Mock()
        mock_download.return_value = mock_download_response

        api = FreesoundAPI(api_key="test_key")
        output_path = tmp_path / "test.mp3"
        
        size = api.download_preview(12345, output_path)
        
        assert output_path.exists()
        assert size > 0


class TestEmotionSampler:
    """Tests for EmotionSampler class."""

    @pytest.fixture
    def sampler(self, temp_config_dir):
        """Create emotion sampler with temp directories."""
        config_file = temp_config_dir / "freesound_config.json"
        download_log = temp_config_dir / "downloads.json"
        staging_dir = temp_config_dir / "staging"
        
        # Create config with API key
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({"freesound_api_key": "test_key"}, f)
        
        return EmotionSampler(
            config_file=config_file,
            download_log=download_log,
            staging_dir=staging_dir
        )

    def test_initialization(self, sampler):
        """Test sampler initializes correctly."""
        assert sampler.api is not None
        assert sampler.hierarchy is not None
        assert sampler.staging_dir.exists()

    def test_load_download_log_new(self, sampler):
        """Test loading non-existent download log creates default."""
        assert "combinations" in sampler.download_log
        assert "total_size_mb" in sampler.download_log
        assert sampler.download_log["total_files"] == 0

    def test_combo_key_generation(self, sampler):
        """Test combination key generation."""
        key = sampler._get_combo_key("happy", "piano")
        assert key == "HAPPY_piano"
        
        key = sampler._get_combo_key("HAPPY", "PIANO")
        assert key == "HAPPY_piano"

    def test_get_combo_size_empty(self, sampler):
        """Test getting size for non-existent combination."""
        size = sampler._get_combo_size("happy", "piano")
        assert size == 0

    def test_can_download_more(self, sampler):
        """Test download limit checking."""
        # Should allow download for new combination
        assert sampler.can_download_more("happy", "piano", 1024000)
        
        # Simulate large existing download
        key = sampler._get_combo_key("happy", "piano")
        sampler.download_log['combinations'][key] = {
            'total_size_bytes': 25 * 1024 * 1024  # 25MB (at limit)
        }
        
        # Should not allow more
        assert not sampler.can_download_more("happy", "piano", 1024)

    @patch.object(FreesoundAPI, 'search')
    def test_search_samples(self, mock_search, sampler):
        """Test searching for samples."""
        mock_search.return_value = [
            SoundResult(
                id=12345,
                name="test sound",
                duration=5.0,
                filesize=1024000,
                tags=["piano"],
                preview_url="http://example.com/preview.mp3",
                license="CC0"
            )
        ]
        
        results = sampler.search_samples("happy", instrument="piano", max_results=5)
        
        assert len(results) == 1
        assert results[0].id == 12345
        mock_search.assert_called_once()

    @patch.object(FreesoundAPI, 'download_preview')
    def test_download_sample(self, mock_download, sampler):
        """Test downloading a single sample."""
        mock_download.return_value = 1024000  # 1MB
        
        result = sampler.download_sample(
            sound_id=12345,
            emotion="happy",
            instrument="piano",
            sound_name="test_sound"
        )
        
        assert result is not None
        assert result.id == 12345
        assert result.emotion == "HAPPY"
        assert result.instrument == "piano"
        assert result.size_bytes == 1024000
        
        # Verify tracking was updated
        key = sampler._get_combo_key("happy", "piano")
        assert key in sampler.download_log['combinations']
        assert len(sampler.download_log['combinations'][key]['files']) == 1

    @patch.object(FreesoundAPI, 'download_preview')
    def test_download_sample_at_limit(self, mock_download, sampler):
        """Test download fails when at size limit."""
        # Set combination to size limit
        key = sampler._get_combo_key("happy", "piano")
        sampler.download_log['combinations'][key] = {
            'emotion': 'HAPPY',
            'instrument': 'piano',
            'level': 'base',
            'total_size_bytes': 25 * 1024 * 1024,
            'files': [],
            'last_updated': '2024-01-01T00:00:00'
        }
        
        result = sampler.download_sample(
            sound_id=12345,
            emotion="happy",
            instrument="piano"
        )
        
        assert result is None
        mock_download.assert_not_called()

    @patch.object(FreesoundAPI, 'search')
    @patch.object(FreesoundAPI, 'download_preview')
    def test_fetch_for_combination(self, mock_download, mock_search, sampler):
        """Test fetching multiple samples for a combination."""
        # Mock search results
        mock_search.return_value = [
            SoundResult(
                id=i,
                name=f"sound_{i}",
                duration=5.0,
                filesize=1024000,
                tags=["piano"],
                preview_url=f"http://example.com/sound_{i}.mp3",
                license="CC0"
            )
            for i in range(1, 6)
        ]
        
        # Mock successful downloads
        mock_download.return_value = 1024000
        
        downloaded = sampler.fetch_for_combination(
            emotion="happy",
            instrument="piano",
            max_files=3
        )
        
        assert len(downloaded) == 3
        assert all(isinstance(s, DownloadedSound) for s in downloaded)

    def test_get_statistics_empty(self, sampler):
        """Test statistics for empty sampler."""
        stats = sampler.get_statistics()
        
        assert stats["total_combinations"] == 0
        assert stats["total_files"] == 0
        assert stats["total_size_mb"] == 0.0
        assert len(stats["combinations"]) == 0

    @patch.object(FreesoundAPI, 'download_preview')
    def test_get_statistics_with_downloads(self, mock_download, sampler):
        """Test statistics after downloads."""
        mock_download.return_value = 1024000
        
        # Download some samples
        sampler.download_sample(12345, "happy", "piano", sound_name="test1")
        sampler.download_sample(12346, "sad", "guitar", sound_name="test2")
        
        stats = sampler.get_statistics()
        
        assert stats["total_combinations"] == 2
        assert stats["total_files"] == 2
        assert stats["total_size_mb"] > 0
        assert len(stats["combinations"]) == 2


def test_constants():
    """Test that required constants are defined."""
    assert len(BASE_EMOTIONS) == 6
    assert "HAPPY" in BASE_EMOTIONS
    assert "SAD" in BASE_EMOTIONS
    assert "ANGRY" in BASE_EMOTIONS
    assert "FEAR" in BASE_EMOTIONS
    assert "SURPRISE" in BASE_EMOTIONS
    assert "DISGUST" in BASE_EMOTIONS
    
    assert len(INSTRUMENTS) == 4
    assert "piano" in INSTRUMENTS
    assert "guitar" in INSTRUMENTS
    assert "drums" in INSTRUMENTS
    assert "vocals" in INSTRUMENTS
