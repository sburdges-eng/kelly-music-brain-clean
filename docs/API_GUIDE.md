# Music Brain API User Guide

Complete guide to using the Music Brain API for emotional music generation.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Rate Limiting](#rate-limiting)
5. [Error Handling](#error-handling)
6. [Examples](#examples)
7. [Best Practices](#best-practices)

## Getting Started

### Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.yourdomain.com`

### API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

### Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Get API info
curl http://localhost:8000/
```

## Authentication

### JWT Tokens

The API supports JWT-based authentication for protected endpoints.

#### Getting a Token

```bash
# Login endpoint (to be implemented)
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'
```

#### Using a Token

Include the token in the Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/metrics
```

### API Keys

API keys can be used as an alternative authentication method:

```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/generate
```

## API Endpoints

### Health Check

Check API health status.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "service": "Music Brain API",
  "version": "1.0.0"
}
```

### Generate Music

Generate music from emotional intent.

**Endpoint**: `POST /generate`

**Request Body**:
```json
{
  "intent": {
    "emotional_intent": "joyful",
    "core_wound": null,
    "core_desire": null,
    "technical": {
      "key": "C major",
      "bpm": 120,
      "genre": "pop"
    }
  },
  "output_format": "midi"
}
```

**Response**:
```json
{
  "success": true,
  "intent": {
    "emotional_intent": "joyful",
    "technical": {
      "key": "C major",
      "bpm": 120,
      "genre": "pop"
    }
  },
  "song": {
    "sections": [...],
    "total_bars": 32,
    "tempo_bpm": 120
  },
  "message": "Generated 4 sections, 32 bars at 120 BPM"
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "emotional_intent": "melancholic",
      "technical": {
        "key": "A minor",
        "bpm": 80
      }
    }
  }'
```

### Interrogate

Conversational music creation endpoint.

**Endpoint**: `POST /interrogate`

**Request Body**:
```json
{
  "message": "I want to create a song about overcoming challenges",
  "session_id": "optional-session-id",
  "context": {}
}
```

**Response**:
```json
{
  "success": true,
  "response": {
    "message": "Interrogation endpoint ready - integration pending",
    "questions": []
  }
}
```

### Get Emotions

Retrieve the full emotion thesaurus.

**Endpoint**: `GET /emotions`

**Response**:
```json
{
  "success": true,
  "emotions": {
    "joy": {...},
    "sadness": {...},
    ...
  },
  "total_nodes": 216
}
```

### Get Emotion Category

Get a specific emotion category.

**Endpoint**: `GET /emotions/{base_emotion}`

**Example**:
```bash
curl http://localhost:8000/emotions/joy
```

**Response**:
```json
{
  "success": true,
  "emotion": "joy",
  "data": {
    "intensity_levels": [...],
    "variations": [...]
  }
}
```

### Metrics

Get API metrics (requires authentication).

**Endpoint**: `GET /metrics`

**Response**:
```json
{
  "counters": {
    "generate_requested": 150,
    "generate_success": 145,
    "generate_error": 5
  },
  "gauges": {
    "generate_duration": {
      "current": 1.23,
      "average": 1.15,
      "min": 0.89,
      "max": 2.45
    }
  },
  "histograms": {
    "generate_duration": {
      "count": 150,
      "average": 1.15,
      "p50": 1.10,
      "p95": 1.80,
      "p99": 2.20
    }
  }
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Per Minute**: 60 requests
- **Per Hour**: 1,000 requests
- **Per Day**: 10,000 requests

### Rate Limit Headers

Every response includes rate limit information:

```
X-RateLimit-Limit-Minute: 60
X-RateLimit-Limit-Hour: 1000
X-RateLimit-Limit-Day: 10000
X-RateLimit-Remaining-Minute: 59
X-RateLimit-Remaining-Hour: 999
X-RateLimit-Remaining-Day: 9999
```

### Rate Limit Exceeded

When rate limit is exceeded, you'll receive:

**Status Code**: `429 Too Many Requests`

**Response**:
```json
{
  "detail": "Rate limit exceeded: 60 requests per minute"
}
```

**Headers**:
```
Retry-After: 60
```

### Best Practices

1. **Monitor Headers**: Check rate limit headers to avoid hitting limits
2. **Implement Backoff**: Use exponential backoff when rate limited
3. **Cache Responses**: Cache emotion data and other static content
4. **Batch Requests**: Combine multiple operations when possible

## Error Handling

### Standard Error Response

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required or invalid
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### Error Examples

**Invalid Request**:
```json
{
  "detail": "Emotion 'invalid_emotion' not found"
}
```

**Authentication Required**:
```json
{
  "detail": "Invalid authentication credentials",
  "headers": {
    "WWW-Authenticate": "Bearer"
  }
}
```

## Examples

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Generate music
response = requests.post(
    f"{BASE_URL}/generate",
    json={
        "intent": {
            "emotional_intent": "energetic",
            "technical": {
                "key": "G major",
                "bpm": 140,
                "genre": "electronic"
            }
        }
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Generated: {data['message']}")
    print(f"Song: {data['song']}")
else:
    print(f"Error: {response.json()}")
```

### JavaScript/TypeScript Client

```typescript
const BASE_URL = 'http://localhost:8000';

async function generateMusic() {
  const response = await fetch(`${BASE_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      intent: {
        emotional_intent: 'peaceful',
        technical: {
          key: 'D major',
          bpm: 70,
          genre: 'ambient'
        }
      }
    })
  });

  if (response.ok) {
    const data = await response.json();
    console.log('Generated:', data.message);
    return data.song;
  } else {
    const error = await response.json();
    throw new Error(error.detail);
  }
}
```

### cURL Examples

**Generate Music**:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "emotional_intent": "nostalgic",
      "technical": {
        "key": "E minor",
        "bpm": 90
      }
    }
  }'
```

**Get All Emotions**:
```bash
curl http://localhost:8000/emotions
```

**Get Specific Emotion**:
```bash
curl http://localhost:8000/emotions/sadness
```

## Best Practices

### 1. Error Handling

Always check response status codes:

```python
response = requests.post(url, json=data)
if response.status_code == 200:
    # Success
    data = response.json()
else:
    # Handle error
    error = response.json()
    print(f"Error: {error['detail']}")
```

### 2. Rate Limit Management

Monitor rate limit headers and implement backoff:

```python
import time

def make_request_with_backoff(url, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(url, json=data)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            time.sleep(retry_after)
            continue
        
        return response
    
    raise Exception("Rate limit exceeded after retries")
```

### 3. Caching

Cache static data like emotions:

```python
import functools

@functools.lru_cache(maxsize=1)
def get_emotions():
    response = requests.get(f"{BASE_URL}/emotions")
    return response.json()
```

### 4. Session Management

Use session IDs for conversational endpoints:

```python
session_id = "user-123-session-456"

response = requests.post(
    f"{BASE_URL}/interrogate",
    json={
        "message": "I want to create a song",
        "session_id": session_id
    }
)
```

### 5. Async Requests

For high-throughput applications, use async:

```python
import asyncio
import aiohttp

async def generate_multiple(songs):
    async with aiohttp.ClientSession() as session:
        tasks = [
            session.post(f"{BASE_URL}/generate", json=song)
            for song in songs
        ]
        responses = await asyncio.gather(*tasks)
        return [await r.json() for r in responses]
```

## Support

For issues, questions, or feature requests:
- Check the API documentation at `/docs`
- Review error messages for troubleshooting
- Check rate limit headers if experiencing 429 errors
- Ensure authentication tokens are valid and not expired
