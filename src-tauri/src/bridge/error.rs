use std::fmt;

/// Custom error types for Music Brain API operations.
#[derive(Debug)]
pub enum MusicBrainError {
    /// Network/HTTP errors
    Network(String),
    /// API returned an error response
    ApiError {
        status: u16,
        message: String,
    },
    /// JSON serialization/deserialization errors
    Serialization(String),
    /// Timeout errors
    Timeout(String),
    /// Configuration errors
    Configuration(String),
    /// Invalid request data
    InvalidRequest(String),
    /// Server unavailable or connection refused
    Unavailable(String),
}

impl fmt::Display for MusicBrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MusicBrainError::Network(msg) => write!(f, "Network error: {}", msg),
            MusicBrainError::ApiError { status, message } => {
                write!(f, "API error (status {}): {}", status, message)
            }
            MusicBrainError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            MusicBrainError::Timeout(msg) => write!(f, "Timeout error: {}", msg),
            MusicBrainError::Configuration(msg) => write!(f, "Configuration error: {}", msg),
            MusicBrainError::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            MusicBrainError::Unavailable(msg) => write!(f, "Service unavailable: {}", msg),
        }
    }
}

impl std::error::Error for MusicBrainError {}

impl From<reqwest::Error> for MusicBrainError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            MusicBrainError::Timeout(err.to_string())
        } else if err.is_connect() {
            MusicBrainError::Unavailable(err.to_string())
        } else {
            MusicBrainError::Network(err.to_string())
        }
    }
}

impl From<serde_json::Error> for MusicBrainError {
    fn from(err: serde_json::Error) -> Self {
        MusicBrainError::Serialization(err.to_string())
    }
}
