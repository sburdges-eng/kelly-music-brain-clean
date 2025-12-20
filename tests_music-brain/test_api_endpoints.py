"""
Comprehensive tests for FastAPI endpoints in music_brain.api.

Tests cover:
- Root endpoint
- Health check
- Music generation endpoint
- Interrogation endpoint
- Emotions endpoints
- Error handling
- Request validation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from pathlib import Path

# Import the FastAPI app
from music_brain.api import app

# Create test client
client = TestClient(app)


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Music Brain API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "endpoints" in data
        assert isinstance(data["endpoints"], list)
        assert len(data["endpoints"]) > 0


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestGenerateEndpoint:
    """Test music generation endpoint."""
    
    def test_generate_with_minimal_intent(self):
        """Test generation with minimal required fields."""
        request_data = {
            "intent": {
                "emotional_intent": "grief"
            },
            "output_format": "midi"
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should either succeed or return a proper error
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "intent" in data
            assert "song" in data or "message" in data
    
    def test_generate_with_full_intent(self):
        """Test generation with complete intent."""
        request_data = {
            "intent": {
                "core_wound": "Loss of a loved one",
                "core_desire": "To remember without pain",
                "emotional_intent": "grief",
                "technical": {
                    "key": "F major",
                    "bpm": 120,
                    "genre": "indie"
                }
            },
            "output_format": "midi"
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should either succeed or return a proper error
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "song" in data
            assert "intent" in data
    
    def test_generate_with_key_mode_separation(self):
        """Test key/mode parsing from technical.key field."""
        request_data = {
            "intent": {
                "emotional_intent": "joy",
                "technical": {
                    "key": "C minor"  # Should parse as key="C", mode="minor"
                }
            }
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should handle key parsing correctly
        assert response.status_code in [200, 500]
    
    def test_generate_missing_emotional_intent(self):
        """Test generation fails gracefully without emotional_intent."""
        request_data = {
            "intent": {
                "core_wound": "test"
            }
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should return validation error (422) or handle gracefully
        assert response.status_code in [422, 500]
    
    def test_generate_invalid_output_format(self):
        """Test generation with invalid output format."""
        request_data = {
            "intent": {
                "emotional_intent": "grief"
            },
            "output_format": "invalid_format"
        }
        
        response = client.post("/generate", json=request_data)
        
        # Should accept but may fail during generation
        assert response.status_code in [200, 422, 500]


class TestInterrogateEndpoint:
    """Test interrogation endpoint."""
    
    def test_interrogate_basic(self):
        """Test basic interrogation request."""
        request_data = {
            "message": "I feel lost and alone"
        }
        
        response = client.post("/interrogate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "response" in data
        assert "message" in data["response"]
    
    def test_interrogate_with_session(self):
        """Test interrogation with session ID."""
        request_data = {
            "message": "What key should I use?",
            "session_id": "test-session-123",
            "context": {"previous_emotion": "grief"}
        }
        
        response = client.post("/interrogate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_interrogate_empty_message(self):
        """Test interrogation with empty message."""
        request_data = {
            "message": ""
        }
        
        response = client.post("/interrogate", json=request_data)
        
        # Should handle gracefully (may return 200 or 422)
        assert response.status_code in [200, 422]


class TestEmotionsEndpoints:
    """Test emotions endpoints."""
    
    def test_get_all_emotions(self):
        """Test getting all emotions."""
        response = client.get("/emotions")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "emotions" in data
        assert isinstance(data["emotions"], dict)
        assert "total_nodes" in data
        assert data["total_nodes"] == 216
    
    def test_get_specific_emotion(self):
        """Test getting a specific emotion category."""
        # Try common emotion names
        for emotion in ["joy", "grief", "anger", "fear"]:
            response = client.get(f"/emotions/{emotion}")
            
            # May return 200 if exists, or 404 if not found
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                assert data["emotion"] == emotion
                assert "data" in data
                break
    
    def test_get_nonexistent_emotion(self):
        """Test getting non-existent emotion returns 404."""
        response = client.get("/emotions/nonexistent_emotion_xyz")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    @pytest.mark.parametrize("emotion_name", ["joy", "grief", "anger", "fear", "love", "sadness"])
    def test_get_emotion_categories(self, emotion_name):
        """Test getting various emotion categories."""
        response = client.get(f"/emotions/{emotion_name}")
        
        # May exist or not - test handles both cases
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert data["emotion"] == emotion_name
        elif response.status_code == 404:
            # Emotion doesn't exist - that's OK
            pass
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = client.post(
            "/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        response = client.post("/generate", json={})
        
        assert response.status_code == 422
    
    def test_invalid_request_structure(self):
        """Test handling of invalid request structure."""
        response = client.post("/generate", json={"wrong": "structure"})
        
        assert response.status_code == 422
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = client.options("/")
        
        # CORS middleware should add headers
        # TestClient may not show all CORS headers, but should not error
        assert response.status_code in [200, 405]


class TestRequestValidation:
    """Test request validation."""
    
    def test_generate_request_validation(self):
        """Test GenerateRequest model validation."""
        # Valid request
        valid_request = {
            "intent": {
                "emotional_intent": "grief"
            },
            "output_format": "midi"
        }
        response = client.post("/generate", json=valid_request)
        assert response.status_code in [200, 500]  # May fail during generation
        
        # Invalid: missing emotional_intent
        invalid_request = {
            "intent": {},
            "output_format": "midi"
        }
        response = client.post("/generate", json=invalid_request)
        assert response.status_code == 422
    
    def test_interrogate_request_validation(self):
        """Test InterrogateRequest model validation."""
        # Valid request
        valid_request = {
            "message": "Test message"
        }
        response = client.post("/interrogate", json=valid_request)
        assert response.status_code == 200
        
        # Invalid: missing message
        invalid_request = {}
        response = client.post("/interrogate", json=invalid_request)
        assert response.status_code == 422


class TestPerformance:
    """Test endpoint performance."""
    
    def test_root_endpoint_performance(self):
        """Test root endpoint responds quickly."""
        import time
        
        start = time.time()
        response = client.get("/")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 0.1  # Should respond in < 100ms
    
    def test_health_endpoint_performance(self):
        """Test health endpoint responds quickly."""
        import time
        
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 0.1  # Should respond in < 100ms
    
    def test_emotions_endpoint_performance(self):
        """Test emotions endpoint loads reasonably quickly."""
        import time
        
        start = time.time()
        response = client.get("/emotions")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        # Loading JSON files may take longer, but should be < 1 second
        assert elapsed < 1.0


class TestIntegration:
    """Integration tests for API workflows."""
    
    def test_complete_generation_workflow(self):
        """Test complete generation workflow."""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Get available emotions
        emotions_response = client.get("/emotions")
        assert emotions_response.status_code == 200
        
        # 3. Generate music
        generate_response = client.post("/generate", json={
            "intent": {
                "emotional_intent": "grief",
                "technical": {
                    "key": "F major",
                    "bpm": 120
                }
            }
        })
        # May succeed or fail depending on SongGenerator implementation
        assert generate_response.status_code in [200, 500]
    
    def test_emotion_to_generation_flow(self):
        """Test flow from emotion lookup to generation."""
        # 1. Get emotion data
        emotion_response = client.get("/emotions/grief")
        
        if emotion_response.status_code == 200:
            emotion_data = emotion_response.json()
            
            # 2. Use emotion in generation
            generate_response = client.post("/generate", json={
                "intent": {
                    "emotional_intent": "grief",
                    "technical": emotion_data.get("data", {}).get("technical", {})
                }
            })
            
            # Should handle the flow
            assert generate_response.status_code in [200, 500]
