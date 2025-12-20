"""
Music Brain API Server
Provides endpoints for emotional music generation
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Music Brain API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load emotion thesaurus
EMOTION_THESAURUS_PATH = Path(__file__).parent.parent / "emotion_thesaurus"


class EmotionalIntent(BaseModel):
    core_wound: Optional[str] = None
    core_desire: Optional[str] = None
    emotional_intent: str
    technical: Optional[Dict[str, Any]] = None


class GenerateRequest(BaseModel):
    intent: EmotionalIntent
    output_format: str = "midi"


class InterrogateRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint providing API information."""
    return {
        "service": "Music Brain API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/generate", "/interrogate", "/emotions"],
    }

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/generate")
async def generate_music(request: GenerateRequest) -> Dict[str, Any]:
    """Generate music from emotional intent.

    Args:
        request: GenerateRequest containing emotional intent and technical parameters

    Returns:
        Dictionary containing success status, intent, song data, and message

    Raises:
        HTTPException: If generation fails or parameters are invalid
    """
    import traceback

    from music_brain.session.generator import SongGenerator

    try:
        # Extract parameters from intent
        intent = request.intent
        technical = intent.technical or {}

        # Parse key and mode from technical.key (e.g., "F major" or "C minor")
        key = "C"
        mode = "major"
        if technical.get("key"):
            key_str = str(technical["key"]).strip()
            if " " in key_str:
                parts = key_str.split(" ", 1)
                key = parts[0]
                mode = parts[1].lower() if len(parts) > 1 else "major"
            else:
                key = key_str

        # Extract other parameters
        tempo = technical.get("bpm")
        genre = technical.get("genre")
        mood = intent.emotional_intent or None

        # Generate song
        generator = SongGenerator()
        song = generator.generate(
            key=key, mode=mode, mood=mood, genre=genre, tempo=float(tempo) if tempo else None
        )

        # Convert to dict for JSON response using the song's to_dict method
        result = {
            "success": True,
            "intent": intent.dict(),
            "song": song.to_dict(),
            "message": (
                f"Generated {len(song.sections)} sections, {song.total_bars} bars "
                f"at {song.tempo_bpm} BPM"
            ),
        }

        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interrogate")
async def interrogate(request: InterrogateRequest) -> Dict[str, Any]:
    """Conversational music creation endpoint.

    Args:
        request: InterrogateRequest with message and optional session context

    Returns:
        Dictionary containing success status and response data

    Raises:
        HTTPException: If interrogation fails
    """
    try:
        # TODO: Integrate with interrogator module
        return {
            "success": True,
            "response": {
                "message": "Interrogation endpoint ready - integration pending",
                "questions": [],
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions")
async def get_emotions() -> Dict[str, Any]:
    """Get the full 6x6x6 emotion thesaurus.

    Returns:
        Dictionary containing success status, emotions data, and total node count

    Raises:
        HTTPException: If emotion data cannot be loaded
    """
    try:
        emotions = {}

        for emotion_file in EMOTION_THESAURUS_PATH.glob("*.json"):
            if emotion_file.stem not in ["metadata", "blends"]:
                with open(emotion_file) as f:
                    emotions[emotion_file.stem] = json.load(f)

        blends_path = EMOTION_THESAURUS_PATH / "blends.json"
        if blends_path.exists():
            with open(blends_path) as f:
                emotions["blends"] = json.load(f)

        return {"success": True, "emotions": emotions, "total_nodes": 216}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions/{base_emotion}")
async def get_emotion_category(base_emotion: str) -> Dict[str, Any]:
    """Get specific emotion category.

    Args:
        base_emotion: Name of the emotion category to retrieve

    Returns:
        Dictionary containing success status, emotion name, and emotion data

    Raises:
        HTTPException: If emotion category not found or cannot be loaded
    """
    emotion_file = EMOTION_THESAURUS_PATH / f"{base_emotion}.json"

    if not emotion_file.exists():
        raise HTTPException(status_code=404, detail=f"Emotion '{base_emotion}' not found")

    try:
        with open(emotion_file) as f:
            data = json.load(f)

        return {"success": True, "emotion": base_emotion, "data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
