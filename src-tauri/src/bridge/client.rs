use crate::bridge::error::MusicBrainError;
use once_cell::sync::Lazy;
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;

/// Configuration for the Music Brain API client.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Base URL for the Music Brain API server
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Connection timeout in seconds
    pub connect_timeout_secs: u64,
    /// Maximum number of retries
    pub max_retries: u32,
    /// Retry delay base in milliseconds (exponential backoff)
    pub retry_delay_base_ms: u64,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_url: std::env::var("MUSIC_BRAIN_API")
                .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string()),
            timeout_secs: 30,
            connect_timeout_secs: 5,
            max_retries: 3,
            retry_delay_base_ms: 100,
        }
    }
}

/// Shared HTTP client with connection pooling and configuration.
pub struct MusicBrainClient {
    client: Client,
    config: Arc<ClientConfig>,
}

impl MusicBrainClient {
    /// Get or create the global client instance.
    pub fn global() -> &'static MusicBrainClient {
        static CLIENT: Lazy<MusicBrainClient> = Lazy::new(|| {
            MusicBrainClient::new(ClientConfig::default())
        });
        &CLIENT
    }

    /// Create a new client with the given configuration.
    pub fn new(config: ClientConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
            .pool_max_idle_per_host(6)
            .pool_idle_timeout(Duration::from_secs(90))
            .user_agent("miDiKompanion/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            config: Arc::new(config),
        }
    }

    /// Get the HTTP client.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Get the configuration.
    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Retry a request with exponential backoff.
    pub async fn retry_request<F, Fut, T>(&self, mut f: F) -> Result<T, MusicBrainError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, MusicBrainError>>,
    {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match f().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);

                    // Don't retry on certain errors
                    if let Some(MusicBrainError::ApiError { status, .. }) = last_error.as_ref() {
                        // Don't retry on 4xx errors (client errors)
                        if (400..500).contains(status) {
                            break;
                        }
                    }

                    // Don't retry on invalid requests
                    if matches!(last_error.as_ref(), Some(MusicBrainError::InvalidRequest(_))) {
                        break;
                    }

                    // Wait before retrying (exponential backoff)
                    if attempt < self.config.max_retries {
                        let delay_ms = self.config.retry_delay_base_ms * (1 << attempt);
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            MusicBrainError::Network("Max retries exceeded".to_string())
        }))
    }
}
