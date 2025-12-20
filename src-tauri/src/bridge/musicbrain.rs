use crate::commands::{GenerateRequest, InterrogateRequest};
use reqwest;
use serde_json::Value;

/// Base URL for the Music Brain API server
const MUSIC_BRAIN_API: &str = "http://127.0.0.1:8000";

/// Generate music from emotional intent via Music Brain API.
///
/// # Arguments
/// * `request` - GenerateRequest with emotional intent and parameters
///
/// # Returns
/// * `Ok(Value)` - JSON response with generated music data
/// * `Err(Box<dyn Error>)` - Error if API request fails
pub async fn generate(
    request: GenerateRequest,
) -> Result<Value, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let res = client
        .post(format!("{}/generate", MUSIC_BRAIN_API))
        .json(&request)
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(res)
}

/// Interrogate endpoint for conversational music creation.
///
/// # Arguments
/// * `request` - InterrogateRequest with message and context
///
/// # Returns
/// * `Ok(Value)` - JSON response with conversation data
/// * `Err(Box<dyn Error>)` - Error if API request fails
pub async fn interrogate(
    request: InterrogateRequest,
) -> Result<Value, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let res = client
        .post(format!("{}/interrogate", MUSIC_BRAIN_API))
        .json(&request)
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(res)
}

/// Get all emotions from the emotion thesaurus.
///
/// # Returns
/// * `Ok(Value)` - JSON response with all emotion categories
/// * `Err(Box<dyn Error>)` - Error if API request fails
pub async fn get_emotions() -> Result<Value, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let res = client
        .get(format!("{}/emotions", MUSIC_BRAIN_API))
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(res)
}
