use serde::{Deserialize, Serialize};
use tauri::command;

/// Emotional intent structure for music generation requests.
#[derive(Debug, Serialize, Deserialize)]
pub struct EmotionalIntent {
    /// Core emotional wound or trauma to express
    pub core_wound: Option<String>,
    /// Core desire or longing to express
    pub core_desire: Option<String>,
    /// Primary emotional intent (e.g., "grief", "joy", "anger")
    pub emotional_intent: String,
    /// Technical parameters (key, tempo, genre, etc.)
    pub technical: Option<serde_json::Value>,
}

/// Request structure for music generation.
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Emotional intent for the music generation
    pub intent: EmotionalIntent,
    /// Output format (e.g., "midi", "json")
    pub output_format: Option<String>,
}

/// Request structure for conversational music creation.
#[derive(Debug, Serialize, Deserialize)]
pub struct InterrogateRequest {
    /// User message or question
    pub message: String,
    /// Optional session ID for maintaining conversation context
    pub session_id: Option<String>,
    /// Optional additional context for the conversation
    pub context: Option<serde_json::Value>,
}

/// Generate music from emotional intent.
///
/// # Arguments
/// * `request` - GenerateRequest containing emotional intent and technical parameters
///
/// # Returns
/// * `Ok(Value)` - JSON response containing generated music data
/// * `Err(String)` - Error message if generation fails
#[command]
pub async fn generate_music(request: GenerateRequest) -> Result<serde_json::Value, String> {
    crate::bridge::musicbrain::generate(request)
        .await
        .map_err(|e| format!("Music generation failed: {}", e))
}

/// Conversational music creation endpoint.
///
/// # Arguments
/// * `request` - InterrogateRequest with message and optional session context
///
/// # Returns
/// * `Ok(Value)` - JSON response with conversation data
/// * `Err(String)` - Error message if interrogation fails
#[command]
pub async fn interrogate(request: InterrogateRequest) -> Result<serde_json::Value, String> {
    crate::bridge::musicbrain::interrogate(request)
        .await
        .map_err(|e| format!("Interrogation failed: {}", e))
}

/// Get the full emotion thesaurus.
///
/// # Returns
/// * `Ok(Value)` - JSON response containing all emotion categories
/// * `Err(String)` - Error message if emotions cannot be loaded
#[command]
pub async fn get_emotions() -> Result<serde_json::Value, String> {
    crate::bridge::musicbrain::get_emotions()
        .await
        .map_err(|e| format!("Failed to load emotions: {}", e))
}

/// Check if the Music Brain API server is available.
///
/// # Returns
/// * `Ok(Value)` - Health check response
/// * `Err(String)` - Error message if health check fails
#[command]
pub async fn health_check() -> Result<serde_json::Value, String> {
    crate::bridge::musicbrain::health_check()
        .await
        .map_err(|e| format!("Health check failed: {}", e))
}

/// Check if the Music Brain API server is available (boolean).
///
/// # Returns
/// * `Ok(bool)` - `true` if available, `false` otherwise
#[command]
pub async fn is_api_available() -> bool {
    crate::bridge::musicbrain::is_available().await
}
