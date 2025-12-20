"""Enhanced HTTP/WebSocket server for penta-core with REST API and real-time streaming."""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional
import threading

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets not available. Install with: pip install websockets")

try:
    from penta_core import PentaCore
    import numpy as np
    PENTA_CORE_AVAILABLE = True
except ImportError:
    PENTA_CORE_AVAILABLE = False
    logging.warning("penta_core not available. Music analysis features disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("penta_server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from file or environment."""
    config_file = Path("server_config.json")
    default_config = {
        "port": int(os.environ.get("PORT", "8000")),
        "host": os.environ.get("HOST", "0.0.0.0"),
        "ws_port": int(os.environ.get("WS_PORT", "8001")),
        "enable_cors": os.environ.get("ENABLE_CORS", "true").lower() == "true",
        "sample_rate": 48000.0,
        "log_requests": True,
        "max_connections": 100
    }
    
    if config_file.exists():
        try:
            with open(config_file) as f:
                file_config = json.load(f)
                default_config.update(file_config)
                logger.info("Loaded configuration from %s", config_file)
        except Exception as e:
            logger.warning("Failed to load config file: %s. Using defaults.", e)
    
    return default_config


CONFIG = load_config()

# Analytics tracking
class Analytics:
    """Simple request analytics."""
    def __init__(self):
        self.requests = []
        self.start_time = datetime.now()
        self.endpoint_counts = {}
    
    def log_request(self, endpoint: str, method: str, duration_ms: float):
        """Log a request."""
        self.requests.append({
            "endpoint": endpoint,
            "method": method,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        })
        key = f"{method} {endpoint}"
        self.endpoint_counts[key] = self.endpoint_counts.get(key, 0) + 1
        
        # Keep only last 1000 requests
        if len(self.requests) > 1000:
            self.requests = self.requests[-1000:]
    
    def get_stats(self) -> dict:
        """Get analytics statistics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "uptime_seconds": uptime,
            "total_requests": len(self.requests),
            "endpoint_counts": self.endpoint_counts,
            "recent_requests": self.requests[-10:]
        }


analytics = Analytics()

# Music engine instance
music_engine = None
if PENTA_CORE_AVAILABLE:
    try:
        music_engine = PentaCore(sample_rate=CONFIG["sample_rate"])
        logger.info("Initialized Penta Core music engine")
    except Exception as e:
        logger.error("Failed to initialize music engine: %s", e)


class EnhancedRequestHandler(SimpleHTTPRequestHandler):
    """Enhanced request handler with REST API and CORS support."""

    def end_headers(self):
        """Add CORS headers if enabled."""
        if CONFIG["enable_cors"]:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight CORS requests."""
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests with REST API."""
        start_time = datetime.now()
        
        try:
            if self.path == "/health":
                self._handle_health()
            elif self.path == "/api/status":
                self._handle_status()
            elif self.path == "/api/analytics":
                self._handle_analytics()
            elif self.path.startswith("/api/harmony"):
                self._handle_harmony()
            elif self.path.startswith("/api/groove"):
                self._handle_groove()
            elif self.path == "/api/state":
                self._handle_state()
            else:
                super().do_GET()
        finally:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            if CONFIG["log_requests"]:
                analytics.log_request(self.path, "GET", duration)
    
    def do_POST(self):
        """Handle POST requests for music analysis."""
        start_time = datetime.now()
        
        try:
            if self.path == "/api/analyze/midi":
                self._handle_midi_analysis()
            elif self.path == "/api/analyze/audio":
                self._handle_audio_analysis()
            elif self.path == "/api/analyze/chord":
                self._handle_chord_analysis()
            else:
                self._send_error(404, "Endpoint not found")
        finally:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            if CONFIG["log_requests"]:
                analytics.log_request(self.path, "POST", duration)
    
    def _handle_health(self):
        """Health check endpoint."""
        self._send_json({
            "status": "healthy",
            "version": "1.0.0",
            "engine_available": music_engine is not None,
            "websockets_available": WEBSOCKETS_AVAILABLE
        })
    
    def _handle_status(self):
        """System status endpoint."""
        status = {
            "server": "running",
            "config": CONFIG,
            "music_engine": music_engine is not None
        }
        if music_engine:
            status["diagnostics"] = music_engine.diagnostics.get_stats()
        self._send_json(status)
    
    def _handle_analytics(self):
        """Analytics endpoint."""
        self._send_json(analytics.get_stats())
    
    def _handle_harmony(self):
        """Harmony analysis endpoint."""
        if not music_engine:
            self._send_error(503, "Music engine not available")
            return
        
        chord = music_engine.harmony.get_current_chord()
        scale = music_engine.harmony.get_current_scale()
        self._send_json({
            "chord": chord,
            "scale": scale
        })
    
    def _handle_groove(self):
        """Groove analysis endpoint."""
        if not music_engine:
            self._send_error(503, "Music engine not available")
            return
        
        self._send_json(music_engine.groove.get_analysis())
    
    def _handle_state(self):
        """Complete state endpoint."""
        if not music_engine:
            self._send_error(503, "Music engine not available")
            return
        
        self._send_json(music_engine.get_state())
    
    def _handle_midi_analysis(self):
        """Analyze MIDI notes."""
        if not music_engine:
            self._send_error(503, "Music engine not available")
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            
            notes = [(n["pitch"], n["velocity"]) for n in data.get("notes", [])]
            music_engine.harmony.process_midi_notes(notes)
            
            self._send_json({
                "success": True,
                "chord": music_engine.harmony.get_current_chord()
            })
        except Exception as e:
            self._send_error(400, f"Invalid request: {e}")
    
    def _handle_audio_analysis(self):
        """Analyze audio data."""
        if not music_engine:
            self._send_error(503, "Music engine not available")
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            
            audio_data = np.array(data.get("samples", []), dtype=np.float32)
            music_engine.groove.process_audio(audio_data)
            
            self._send_json({
                "success": True,
                "analysis": music_engine.groove.get_analysis()
            })
        except Exception as e:
            self._send_error(400, f"Invalid request: {e}")
    
    def _handle_chord_analysis(self):
        """Analyze chord progression."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            
            chords = data.get("chords", [])
            # Simple progression analysis
            progression = self._analyze_progression(chords)
            
            self._send_json({
                "success": True,
                "progression": progression
            })
        except Exception as e:
            self._send_error(400, f"Invalid request: {e}")
    
    def _analyze_progression(self, chords: List[str]) -> dict:
        """Analyze chord progression."""
        # Basic progression analysis
        common_progressions = {
            ("I", "IV", "V"): "Classic I-IV-V",
            ("I", "V", "vi", "IV"): "Pop progression",
            ("ii", "V", "I"): "Jazz ii-V-I",
            ("I", "vi", "IV", "V"): "50s progression"
        }
        
        return {
            "chords": chords,
            "length": len(chords),
            "pattern": "Custom" if tuple(chords) not in common_progressions else common_progressions[tuple(chords)]
        }
    
    def _send_json(self, data: dict):
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())
    
    def log_message(self, format, *args):
        """Override to use logger."""
        if CONFIG["log_requests"]:
            logger.info(format % args)


# WebSocket server for real-time streaming
async def websocket_handler(websocket, path):
    """Handle WebSocket connections for real-time music data."""
    logger.info("WebSocket client connected from %s", websocket.remote_address)
    
    try:
        async for message in websocket:
            # Echo back current state
            if music_engine:
                state = music_engine.get_state()
                await websocket.send(json.dumps(state))
            else:
                await websocket.send(json.dumps({"error": "Engine not available"}))
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket client disconnected")


def run_websocket_server():
    """Run WebSocket server in separate thread."""
    if not WEBSOCKETS_AVAILABLE:
        logger.warning("WebSocket server disabled - websockets package not installed")
        return
    
    async def start_ws():
        async with websockets.serve(websocket_handler, CONFIG["host"], CONFIG["ws_port"]):
            logger.info("WebSocket server running on ws://%s:%d", CONFIG["host"], CONFIG["ws_port"])
            await asyncio.Future()  # run forever
    
    try:
        asyncio.run(start_ws())
    except Exception as e:
        logger.error("WebSocket server error: %s", e)


def run_http_server():
    """Run HTTP server."""
    server_address = (CONFIG["host"], CONFIG["port"])
    httpd = HTTPServer(server_address, EnhancedRequestHandler)

    def shutdown_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutting down server...")
        httpd.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    logger.info("HTTP server running on http://%s:%d", CONFIG["host"], CONFIG["port"])
    logger.info("REST API available at /api/*")
    logger.info("Health check: http://%s:%d/health", CONFIG["host"], CONFIG["port"])
    httpd.serve_forever()


if __name__ == "__main__":
    # Start WebSocket server in separate thread
    if WEBSOCKETS_AVAILABLE:
        ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
        ws_thread.start()
    
    # Start HTTP server (blocking)
    run_http_server()
