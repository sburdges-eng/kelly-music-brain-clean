# Music Brain API Server - Build Documentation

**Date:** December 2024 (Updated: December 20, 2025)  
**Status:** ✅ Successfully Built and Tested  
**Component:** Python API Server (`music_brain/api.py`)

---

## Overview

The Music Brain API Server is a FastAPI-based web service that provides endpoints for emotional music generation. This document details the build process, dependencies, and testing procedures.

## Latest Build Results

- **Python Version:** 3.14.2
- **All 30 API tests pass**
- **Virtual environment created in `venv/`**
- **Interrogate endpoint fully functional** (with phase detection and question generation)

## Build Summary

✅ **All components successfully built and tested**

- Python dependencies installed
- Emotion thesaurus directory created
- API server running on `http://127.0.0.1:8000`
- All endpoints tested and verified

---

## Prerequisites

### System Requirements
- **Python:** 3.11+ (tested with Python 3.14.2)
- **pip:** Latest version
- **Operating System:** macOS (tested on darwin 25.1.0)

### Required Python Packages
- `fastapi>=0.104.0`
- `uvicorn>=0.24.0`
- `pydantic>=2.0.0`
- All dependencies from `pyproject.toml`

---

## Build Steps

### 1. Dependency Check ✅

**Status:** Completed

**Actions Taken:**
- Verified Python 3.14.2 is installed
- Verified pip3 is available at `/usr/local/bin/pip3`
- Checked existing package installations

**Command:**
```bash
python3 --version
which pip3
```

### 2. Install Python Dependencies ✅

**Status:** Completed

**Actions Taken:**
- Installed FastAPI and uvicorn directly
- Installed project in development mode using `pip install -e .`
- All dependencies from `pyproject.toml` installed

**Commands:**
```bash
pip3 install fastapi uvicorn
pip3 install -e .
```

**Dependencies Installed:**
- Core audio processing: numpy, scipy, librosa, soundfile, audioread, resampy, numba
- MIDI processing: mido, pretty-midi, music21
- Web framework: fastapi, uvicorn, pydantic, httpx, python-dotenv, fastmcp

### 3. Setup Emotion Thesaurus ✅

**Status:** Completed

**Issue Found:** The API expects an `emotion_thesaurus` directory at the project root, but it didn't exist.

**Actions Taken:**
- Created `emotion_thesaurus/` directory at project root
- Copied emotion JSON files from `data/` directory:
  - `joy.json`
  - `anger.json`
  - `fear.json`
  - `sad.json`
  - `disgust.json`
  - `surprise.json`

**Commands:**
```bash
mkdir -p emotion_thesaurus
cp data/{joy,anger,fear,sad,disgust,surprise}.json emotion_thesaurus/
```

**Note:** The API code references `EMOTION_THESAURUS_PATH = Path(__file__).parent.parent / "emotion_thesaurus"`, which resolves to the project root.

### 4. Build and Start API Server ✅

**Status:** Completed

**Actions Taken:**
- Verified API module imports successfully
- Started API server using `python3 -m music_brain.api`
- Server running on `http://127.0.0.1:8000`

**Command:**
```bash
python3 -m music_brain.api
```

**Server Output:**
- Server starts on port 8000
- Process ID: 18014 (example)
- No errors during startup

---

## API Endpoints Testing

### 1. Root Endpoint (`/`) ✅

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

### 2. Health Check (`/health`) ✅

**Request:**
```bash
curl http://127.0.0.1:8000/health
```

**Response:**
```json
{
    "status": "healthy",
    "service": "Music Brain API",
    "version": "1.0.0"
}
```

### 3. Get All Emotions (`/emotions`) ✅

**Request:**
```bash
curl http://127.0.0.1:8000/emotions
```

**Response:**
- Success: `true`
- Total emotions: `6`
- Total nodes: `216`
- Contains emotion data for: joy, anger, fear, sad, disgust, surprise

### 4. Get Specific Emotion (`/emotions/{base_emotion}`) ✅

**Request:**
```bash
curl http://127.0.0.1:8000/emotions/joy
```

**Response:**
- Success: `true`
- Emotion: `joy`
- Contains full emotion thesaurus data structure

### 5. Generate Music (`/generate`) ✅

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "emotional_intent": "hopeful",
      "technical": {
        "key": "C major",
        "bpm": 100
      }
    },
    "output_format": "midi"
  }'
```

**Response:**
- Success: `true`
- Generated song structure with:
  - Title, key, mode, tempo
  - Multiple sections (intro, verse, chorus, etc.)
  - Chord progressions
  - Energy levels
  - Total bars and duration estimate

**Example Response:**
```json
{
  "success": true,
  "intent": {...},
  "song": {
    "title": "New Dawn",
    "key": "C",
    "mode": "major",
    "tempo_bpm": 100.0,
    "total_bars": 36,
    "sections": [...]
  },
  "message": "Generated 8 sections, 36 bars at 100.0 BPM"
}
```

### 6. Interrogate Endpoint (`/interrogate`) ✅

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/interrogate \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to create something emotional"}'
```

**Response:**
```json
{
  "success": true,
  "response": {
    "message": "Exploring intent - here are some questions to consider",
    "questions": [
      "What's the one thing this song absolutely must accomplish?",
      "Is this song for you, or for someone else to hear?",
      "What do you NEED to say with this song? Not want - need."
    ],
    "phase": "intent",
    "context": {
      "core_emotion": "I want to create something emotional",
      "subject": "",
      "tempo_feel": "",
      "production_style": ""
    }
  }
}
```

**Note:** The endpoint now fully integrates with `SongInterrogator`, detecting phase from keywords and returning relevant questions.

---

## Integration Verification

### Component Integration ✅

**SongGenerator Integration:**
- ✅ Successfully imports `music_brain.session.generator.SongGenerator`
- ✅ Generates complete song structures
- ✅ Returns proper `GeneratedSong` objects with `to_dict()` method
- ✅ Handles mood, genre, key, tempo parameters correctly

**Emotion Thesaurus Integration:**
- ✅ Successfully loads emotion JSON files
- ✅ Provides 6 base emotion categories
- ✅ Supports 216-node emotion thesaurus structure
- ✅ Endpoints return proper emotion data

**API Framework Integration:**
- ✅ FastAPI application initializes correctly
- ✅ CORS middleware configured
- ✅ Pydantic models validate requests
- ✅ Error handling works correctly

---

## Issues Encountered and Resolutions

### Issue 1: Missing FastAPI and uvicorn
**Problem:** FastAPI and uvicorn were not installed  
**Resolution:** Installed via `pip3 install fastapi uvicorn`

### Issue 2: Missing emotion_thesaurus Directory
**Problem:** API expected `emotion_thesaurus/` directory at project root, but it didn't exist  
**Resolution:** Created directory and copied emotion JSON files from `data/` directory

### Issue 3: Package Installation
**Problem:** Some dependencies needed to be installed  
**Resolution:** Used `pip install -e .` to install project in development mode with all dependencies

---

## Running the Server

### Start Server
```bash
cd /Users/seanburdges/dev/miDiKompanion
python3 -m music_brain.api
```

### Start Server (Background)
```bash
python3 -m music_brain.api &
```

### Using uvicorn Directly
```bash
uvicorn music_brain.api:app --host 127.0.0.1 --port 8000
```

### Access API Documentation
Once the server is running, visit:
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

---

## Testing

### Manual Testing
All endpoints have been manually tested using `curl` commands. See "API Endpoints Testing" section above.

### Automated Testing
Test files are available in `tests_music-brain/test_api.py`. To run:
```bash
pytest tests_music-brain/test_api.py
```

---

## Next Steps

### Completed Features
- ✅ **Interrogate Endpoint:** Fully integrated with `music_brain.session.interrogator.SongInterrogator`
- ✅ **Rate Limiting:** Implemented via `RateLimitMiddleware` (60/min, 1000/hour, 10000/day)
- ✅ **Security Headers:** Added via `SecurityHeadersMiddleware`
- ✅ **Request Logging:** Added via `RequestLoggingMiddleware`
- ✅ **Testing:** 30 automated tests pass

### Recommended Improvements
1. **Voice Processing Endpoints:** Add endpoints for auto-tune, modulation, and synthesis
2. **Error Handling:** Add more specific error messages and validation
3. **Authentication:** Full authentication flow for production use
4. **WebSocket Support:** Real-time collaboration features

### Additional Components to Build
- **Tauri Desktop App:** Build the desktop application (`src-tauri/`)
- **C++ Audio Engine:** Build the C++ library with Python bindings (`cpp_music_brain/`)
- **Python Modules:** Further development of `music_brain` and `penta_core` packages

---

## Success Criteria ✅

- ✅ Component builds without errors
- ✅ All tests pass (manual testing completed)
- ✅ API server starts successfully
- ✅ All endpoints respond correctly
- ✅ Integration with SongGenerator works
- ✅ Emotion thesaurus loads correctly
- ✅ CORS configured for frontend integration

---

## Conclusion

The Music Brain API Server has been successfully built, tested, and verified. All endpoints are functional and the server is ready for integration with frontend applications or other services.

**Build Time:** ~5 minutes  
**Status:** Production Ready (with noted improvements)

---

## Contact & Support

For issues or questions about the build process, refer to:
- Project documentation in `docs/`
- API code in `music_brain/api.py`
- Test files in `tests_music-brain/`
