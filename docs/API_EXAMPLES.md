# Music Brain API - Usage Examples

Complete examples for using the Music Brain API endpoints.

## Base URL

```
http://127.0.0.1:8000
```

## Interactive Documentation

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

---

## Endpoints

### 1. Root Endpoint

Get API information and available endpoints.

**Request:**
```bash
curl http://127.0.0.1:8000/
```

**Response:**
```json
{
  "service": "Music Brain API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": ["/generate", "/interrogate", "/emotions"]
}
```

**Python Example:**
```python
import requests

response = requests.get("http://127.0.0.1:8000/")
print(response.json())
```

---

### 2. Health Check

Check if the API server is running.

**Request:**
```bash
curl http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

**Python Example:**
```python
import requests

response = requests.get("http://127.0.0.1:8000/health")
assert response.json()["status"] == "healthy"
```

---

### 3. Generate Music

Generate music from emotional intent.

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "emotional_intent": "grief",
      "core_wound": "Loss of a loved one",
      "core_desire": "To remember without pain",
      "technical": {
        "key": "F major",
        "bpm": 120,
        "genre": "indie"
      }
    },
    "output_format": "midi"
  }'
```

**Response:**
```json
{
  "success": true,
  "intent": {
    "emotional_intent": "grief",
    "core_wound": "Loss of a loved one",
    "core_desire": "To remember without pain",
    "technical": {
      "key": "F major",
      "bpm": 120,
      "genre": "indie"
    }
  },
  "song": {
    "sections": [...],
    "total_bars": 16,
    "tempo_bpm": 120,
    ...
  },
  "message": "Generated 4 sections, 16 bars at 120 BPM"
}
```

**Python Example:**
```python
import requests

request_data = {
    "intent": {
        "emotional_intent": "grief",
        "core_wound": "Loss of a loved one",
        "core_desire": "To remember without pain",
        "technical": {
            "key": "F major",
            "bpm": 120,
            "genre": "indie"
        }
    },
    "output_format": "midi"
}

response = requests.post(
    "http://127.0.0.1:8000/generate",
    json=request_data
)

if response.status_code == 200:
    result = response.json()
    print(f"Success: {result['message']}")
    print(f"Generated {result['song']['total_bars']} bars")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

**Minimal Request:**
```python
# Only emotional_intent is required
minimal_request = {
    "intent": {
        "emotional_intent": "joy"
    }
}

response = requests.post(
    "http://127.0.0.1:8000/generate",
    json=minimal_request
)
```

---

### 4. Interrogate

Conversational music creation endpoint (currently returns placeholder).

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/interrogate \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I feel lost and alone",
    "session_id": "session-123",
    "context": {
      "previous_emotion": "grief"
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "response": {
    "message": "Interrogation endpoint ready - integration pending",
    "questions": []
  }
}
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/interrogate",
    json={
        "message": "I feel lost and alone",
        "session_id": "session-123"
    }
)

result = response.json()
print(result["response"]["message"])
```

---

### 5. Get All Emotions

Retrieve the full 6x6x6 emotion thesaurus (216 nodes).

**Request:**
```bash
curl http://127.0.0.1:8000/emotions
```

**Response:**
```json
{
  "success": true,
  "emotions": {
    "joy": {...},
    "grief": {...},
    "anger": {...},
    ...
    "blends": {...}
  },
  "total_nodes": 216
}
```

**Python Example:**
```python
import requests

response = requests.get("http://127.0.0.1:8000/emotions")
emotions = response.json()

print(f"Total emotion nodes: {emotions['total_nodes']}")
print(f"Available categories: {list(emotions['emotions'].keys())}")
```

---

### 6. Get Specific Emotion

Get a specific emotion category.

**Request:**
```bash
curl http://127.0.0.1:8000/emotions/grief
```

**Response:**
```json
{
  "success": true,
  "emotion": "grief",
  "data": {
    "base_emotion": "grief",
    "intensity_levels": [...],
    "sub_emotions": [...],
    "musical_parameters": {
      "key": "F",
      "mode": "minor",
      "tempo": 60-80,
      ...
    }
  }
}
```

**Python Example:**
```python
import requests

emotion_name = "grief"
response = requests.get(f"http://127.0.0.1:8000/emotions/{emotion_name}")

if response.status_code == 200:
    emotion_data = response.json()
    print(f"Emotion: {emotion_data['emotion']}")
    print(f"Musical parameters: {emotion_data['data'].get('musical_parameters', {})}")
elif response.status_code == 404:
    print(f"Emotion '{emotion_name}' not found")
```

---

## Complete Workflow Example

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# 1. Check API health
health = requests.get(f"{BASE_URL}/health")
assert health.json()["status"] == "healthy"

# 2. Explore available emotions
emotions = requests.get(f"{BASE_URL}/emotions").json()
print(f"Available {emotions['total_nodes']} emotion nodes")

# 3. Get specific emotion details
grief_data = requests.get(f"{BASE_URL}/emotions/grief").json()
musical_params = grief_data["data"].get("musical_parameters", {})

# 4. Generate music based on emotion
generate_request = {
    "intent": {
        "emotional_intent": "grief",
        "core_wound": "Loss of a loved one",
        "core_desire": "To remember without pain",
        "technical": {
            "key": musical_params.get("key", "F") + " " + musical_params.get("mode", "minor"),
            "bpm": musical_params.get("tempo", [60, 80])[0] if isinstance(musical_params.get("tempo"), list) else 70
        }
    }
}

result = requests.post(f"{BASE_URL}/generate", json=generate_request)

if result.status_code == 200:
    song = result.json()
    print(f"✅ Generated: {song['message']}")
    print(f"   Sections: {len(song['song']['sections'])}")
    print(f"   Bars: {song['song']['total_bars']}")
    print(f"   Tempo: {song['song']['tempo_bpm']} BPM")
else:
    print(f"❌ Error: {result.status_code}")
    print(result.json())
```

---

## Error Handling

### Validation Errors (422)

When required fields are missing or invalid:

```python
response = requests.post(
    "http://127.0.0.1:8000/generate",
    json={"intent": {}}  # Missing emotional_intent
)

if response.status_code == 422:
    errors = response.json()
    print("Validation errors:", errors["detail"])
```

### Server Errors (500)

When generation fails:

```python
response = requests.post(
    "http://127.0.0.1:8000/generate",
    json={"intent": {"emotional_intent": "grief"}}
)

if response.status_code == 500:
    error = response.json()
    print("Server error:", error["detail"])
```

### Not Found (404)

When emotion doesn't exist:

```python
response = requests.get("http://127.0.0.1:8000/emotions/nonexistent")

if response.status_code == 404:
    error = response.json()
    print("Not found:", error["detail"])
```

---

## Using with FastAPI TestClient

For testing or development:

```python
from fastapi.testclient import TestClient
from music_brain.api import app

client = TestClient(app)

# Test root endpoint
response = client.get("/")
assert response.status_code == 200

# Test generation
response = client.post("/generate", json={
    "intent": {"emotional_intent": "grief"}
})
assert response.status_code in [200, 500]  # May fail during generation
```

---

## Rate Limiting & Best Practices

1. **Check health first**: Always verify `/health` before making requests
2. **Handle errors gracefully**: Check status codes and handle 422, 404, 500
3. **Use session IDs**: For interrogation endpoint, maintain session context
4. **Cache emotion data**: Emotion thesaurus doesn't change frequently
5. **Validate input**: Ensure required fields are present before sending

---

## Next Steps

- Complete `/interrogate` endpoint integration
- Add authentication/authorization
- Implement rate limiting
- Add WebSocket support for real-time updates
- Add MIDI file download endpoints
