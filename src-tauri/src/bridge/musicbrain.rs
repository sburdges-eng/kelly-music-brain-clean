use crate::bridge::client::MusicBrainClient;
use crate::bridge::error::MusicBrainError;
use crate::commands::{GenerateRequest, InterrogateRequest};
use serde_json::Value;

/// Generate music from emotional intent via Music Brain API.
///
/// This function sends a POST request to the `/generate` endpoint with the provided
/// emotional intent and technical parameters. It includes automatic retry logic
/// with exponential backoff for transient failures.
///
/// # Arguments
/// * `request` - GenerateRequest with emotional intent and parameters
///
/// # Returns
/// * `Ok(Value)` - JSON response with generated music data
/// * `Err(MusicBrainError)` - Detailed error if API request fails
///
/// # Errors
/// Returns `MusicBrainError` for:
/// - Network connectivity issues
/// - API errors (4xx, 5xx status codes)
/// - Timeout errors
/// - Serialization errors
///
/// # Example
/// ```rust,no_run
/// use crate::commands::{GenerateRequest, EmotionalIntent};
///
/// let intent = EmotionalIntent {
///     emotional_intent: "joy".to_string(),
///     core_wound: None,
///     core_desire: None,
///     technical: None,
/// };
///
/// let request = GenerateRequest {
///     intent,
///     output_format: Some("midi".to_string()),
/// };
///
/// match generate(request).await {
///     Ok(music_data) => println!("Generated: {:?}", music_data),
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// ```
pub async fn generate(request: GenerateRequest) -> Result<Value, MusicBrainError> {
    // Validate request
    if request.intent.emotional_intent.is_empty() {
        return Err(MusicBrainError::InvalidRequest(
            "emotional_intent cannot be empty".to_string(),
        ));
    }

    let client = MusicBrainClient::global();
    let url = format!("{}/generate", client.base_url());

    client
        .retry_request(|| async {
            let response = client
                .client()
                .post(&url)
                .json(&request)
                .send()
                .await?;

            let status = response.status();

            if !status.is_success() {
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                return Err(MusicBrainError::ApiError {
                    status: status.as_u16(),
                    message: error_text,
                });
            }

            response.json::<Value>().await.map_err(Into::into)
        })
        .await
}

/// Interrogate endpoint for conversational music creation.
///
/// This function enables conversational interaction with the Music Brain API,
/// allowing users to ask questions or provide context for music generation.
/// Supports session-based conversations via optional session_id.
///
/// # Arguments
/// * `request` - InterrogateRequest with message and optional context
///
/// # Returns
/// * `Ok(Value)` - JSON response with conversation data
/// * `Err(MusicBrainError)` - Detailed error if API request fails
///
/// # Errors
/// Returns `MusicBrainError` for:
/// - Network connectivity issues
/// - API errors (4xx, 5xx status codes)
/// - Timeout errors
/// - Serialization errors
///
/// # Example
/// ```rust,no_run
/// use crate::commands::InterrogateRequest;
///
/// let request = InterrogateRequest {
///     message: "What emotions work well for a sad ballad?".to_string(),
///     session_id: Some("session-123".to_string()),
///     context: None,
/// };
///
/// match interrogate(request).await {
///     Ok(response) => println!("Response: {:?}", response),
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// ```
pub async fn interrogate(request: InterrogateRequest) -> Result<Value, MusicBrainError> {
    // Validate request
    if request.message.trim().is_empty() {
        return Err(MusicBrainError::InvalidRequest(
            "message cannot be empty".to_string(),
        ));
    }

    let client = MusicBrainClient::global();
    let url = format!("{}/interrogate", client.base_url());

    client
        .retry_request(|| async {
            let response = client
                .client()
                .post(&url)
                .json(&request)
                .send()
                .await?;

            let status = response.status();

            if !status.is_success() {
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                return Err(MusicBrainError::ApiError {
                    status: status.as_u16(),
                    message: error_text,
                });
            }

            response.json::<Value>().await.map_err(Into::into)
        })
        .await
}

/// Get all emotions from the emotion thesaurus.
///
/// Retrieves the complete 216-node emotion thesaurus structure, including
/// all base emotions, sub-emotions, and their VAD (Valence, Arousal, Dominance)
/// coordinates.
///
/// # Returns
/// * `Ok(Value)` - JSON response with all emotion categories and nodes
/// * `Err(MusicBrainError)` - Detailed error if API request fails
///
/// # Errors
/// Returns `MusicBrainError` for:
/// - Network connectivity issues
/// - API errors (4xx, 5xx status codes)
/// - Timeout errors
/// - Serialization errors
///
/// # Example
/// ```rust,no_run
/// match get_emotions().await {
///     Ok(emotions) => {
///         println!("Loaded {} emotion categories", emotions);
///     }
///     Err(e) => eprintln!("Error loading emotions: {}", e),
/// }
/// ```
pub async fn get_emotions() -> Result<Value, MusicBrainError> {
    let client = MusicBrainClient::global();
    let url = format!("{}/emotions", client.base_url());

    client
        .retry_request(|| async {
            let response = client.client().get(&url).send().await?;

            let status = response.status();

            if !status.is_success() {
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                return Err(MusicBrainError::ApiError {
                    status: status.as_u16(),
                    message: error_text,
                });
            }

            response.json::<Value>().await.map_err(Into::into)
        })
        .await
}

/// Health check endpoint to verify API server availability.
///
/// This is a lightweight endpoint that can be used to check if the Music Brain
/// API server is running and responsive before making actual requests.
///
/// # Returns
/// * `Ok(Value)` - Health check response (usually contains status and version)
/// * `Err(MusicBrainError)` - Error if health check fails
///
/// # Example
/// ```rust,no_run
/// match health_check().await {
///     Ok(health) => println!("API is healthy: {:?}", health),
///     Err(e) => eprintln!("API health check failed: {}", e),
/// }
/// ```
pub async fn health_check() -> Result<Value, MusicBrainError> {
    let client = MusicBrainClient::global();
    let url = format!("{}/health", client.base_url());

    // Health check should be fast, use shorter timeout
    let response = client
        .client()
        .get(&url)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await?;

    let status = response.status();

    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        return Err(MusicBrainError::ApiError {
            status: status.as_u16(),
            message: error_text,
        });
    }

    response.json::<Value>().await.map_err(Into::into)
}

/// Check if the API server is available and responding.
///
/// This is a convenience function that attempts a health check and returns
/// a boolean indicating availability.
///
/// # Returns
/// * `true` - API server is available and responding
/// * `false` - API server is unavailable or not responding
///
/// # Example
/// ```rust,no_run
/// if is_available().await {
///     println!("Ready to generate music!");
/// } else {
///     println!("API server is not available");
/// }
/// ```
pub async fn is_available() -> bool {
    health_check().await.is_ok()
}
