# Kelly - Therapeutic iDAW (Desktop)

Kelly is a therapeutic desktop app that turns emotional intent into music. The current stack pairs a React + Tauri shell (UI + desktop bridge) with a Python “Music Brain” API that serves generation and interrogation endpoints.

## What’s working
- React + Tauri UI renders and routes buttons (Load Emotions, Generate Music, Start Interrogation, Side A/B toggle) to Tauri commands.
- Tauri Rust commands forward to the Music Brain API at `http://127.0.0.1:8000`.
- Error boundary and API status indicator surface connectivity issues.

## Architecture (high level)
- **Frontend:** React (Vite) bundled by Tauri. Lives in `src/` with hooks such as `useMusicBrain`.
- **Desktop bridge:** Tauri 2 Rust commands (`get_emotions`, `generate_music`, `interrogate`) forward HTTP calls to the Music Brain API.
- **Music Brain API (Python):** Expected to run locally on `127.0.0.1:8000`, exposing `/emotions`, `/generate`, and `/interrogate`.

Flow:
```
React UI → Tauri command → HTTP → Music Brain API → JSON response → UI
```

## Prerequisites
- Node 18+ and npm
- Rust toolchain with Cargo (required by Tauri 2 CLI)
- Python 3.9+ with `pip` (virtualenv recommended)

## Setup

From the repository root:

```bash
npm install
python -m pip install -e ".[dev]"
```

## Run (development)
1) Start the Music Brain API server  
   - Preferred: `./scripts/start_music_brain_api.sh`  
   - Default host/port: `127.0.0.1:8000`

2) Launch the desktop app  
   - `npm run tauri dev` (opens the Tauri window; proxies to the dev server on http://localhost:1420)  
   - UI-only iteration (no Tauri shell): `npm run dev` (API calls still target `127.0.0.1:8000`)

3) Smoke-test the API  
```bash
curl http://127.0.0.1:8000/emotions
```
If the call fails, the UI will show “API Offline.”

## Tauri command → API contract
The UI uses the hook from `.agents/handoffs/CURRENT_STATE.md` (mirrored here for quick reference):

| Tauri command       | HTTP call                     | Purpose                       |
|---------------------|-------------------------------|-------------------------------|
| `get_emotions`      | `GET /emotions`               | List available emotions/presets |
| `generate_music`    | `POST /generate`              | Generate music for an intent  |
| `interrogate`       | `POST /interrogate`           | Ask follow-ups / refine intent |

Example payloads:
- `POST /generate`
```json
{
  "intent": {
    "core_wound": "fear of being forgotten",
    "core_desire": "to feel seen",
    "emotional_intent": "anxious but hopeful",
    "technical": {
      "key": "C",
      "bpm": 90,
      "progression": ["I", "V", "vi", "IV"],
      "genre": "indie"
    }
  },
  "output_format": "midi"
}
```

- `POST /interrogate`
```json
{
  "message": "Make it feel more grounded",
  "session_id": "optional-session-id",
  "context": {}
}
```

- `GET /emotions`  
Returns a JSON list/dictionary of available emotions.

## Troubleshooting
- If the UI shows “API Offline,” ensure the Music Brain API server is running on `127.0.0.1:8000`.
- Use `curl http://127.0.0.1:8000/emotions` to verify availability.
- Tauri CLI needs Rust + system toolchain; on macOS install Xcode command line tools (`xcode-select --install`).

## License
MIT