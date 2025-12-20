# Music Brain Bridge - Enhanced Implementation

This module provides a robust, production-ready bridge to the Music Brain API server with comprehensive error handling, retry logic, and connection pooling.

## Features

### ✅ Custom Error Types
- **`MusicBrainError`**: Comprehensive error enum covering all failure scenarios
  - Network errors (connection issues, timeouts)
  - API errors (HTTP status codes with detailed messages)
  - Serialization errors (JSON parsing issues)
  - Configuration errors (invalid settings)
  - Invalid request errors (validation failures)
  - Service unavailable errors

### ✅ Connection Pooling & Performance
- **Reusable HTTP client**: Single client instance shared across all requests
- **Connection pooling**: Configurable pool size and idle timeout
- **Connection reuse**: Reduces overhead of establishing new connections
- **User agent**: Identifies requests as coming from miDiKompanion

### ✅ Retry Logic with Exponential Backoff
- **Automatic retries**: Configurable max retries (default: 3)
- **Exponential backoff**: Delays increase exponentially (100ms, 200ms, 400ms...)
- **Smart retry logic**: 
  - Retries on network errors and 5xx server errors
  - Skips retries on 4xx client errors (invalid requests)
  - Skips retries on validation errors

### ✅ Configuration Management
- **Environment variable support**: Set `MUSIC_BRAIN_API` to override default URL
- **Configurable timeouts**: 
  - Request timeout: 30 seconds (default)
  - Connection timeout: 5 seconds (default)
- **Flexible configuration**: All settings can be customized via `ClientConfig`

### ✅ Request Validation
- **Input validation**: Validates requests before sending
  - Empty emotional_intent is rejected
  - Empty messages are rejected
- **Clear error messages**: Validation errors provide helpful feedback

### ✅ Health Check Support
- **`health_check()`**: Full health check endpoint
- **`is_available()`**: Quick boolean availability check
- **Fast timeouts**: Health checks use shorter timeout (5 seconds)

### ✅ Enhanced Error Handling
- **Detailed error messages**: Errors include context and status codes
- **Error conversion**: Automatic conversion from reqwest and serde_json errors
- **User-friendly messages**: Errors formatted for display to end users

## Usage Examples

### Basic Music Generation

```rust
use crate::commands::{GenerateRequest, EmotionalIntent};

let intent = EmotionalIntent {
    emotional_intent: "joy".to_string(),
    core_wound: None,
    core_desire: None,
    technical: None,
};

let request = GenerateRequest {
    intent,
    output_format: Some("midi".to_string()),
};

match generate(request).await {
    Ok(music_data) => {
        println!("Generated music: {:?}", music_data);
    }
    Err(e) => {
        eprintln!("Generation failed: {}", e);
        // Error will be detailed: "API error (status 500): Internal server error"
    }
}
```

### Health Check Before Operations

```rust
// Check if API is available before making requests
if is_available().await {
    // Safe to proceed with generation
    let result = generate(request).await;
} else {
    eprintln!("API server is not available");
}
```

### Custom Configuration

```rust
use crate::bridge::client::{MusicBrainClient, ClientConfig};

let config = ClientConfig {
    base_url: "http://localhost:9000".to_string(),
    timeout_secs: 60,
    connect_timeout_secs: 10,
    max_retries: 5,
    retry_delay_base_ms: 200,
};

let client = MusicBrainClient::new(config);
// Use client for custom requests
```

### Environment Variable Configuration

```bash
# Set custom API URL
export MUSIC_BRAIN_API=http://localhost:9000

# The client will automatically use this URL
```

## API Endpoints

### `generate(request: GenerateRequest) -> Result<Value, MusicBrainError>`
Generates music from emotional intent. Includes automatic retry logic.

### `interrogate(request: InterrogateRequest) -> Result<Value, MusicBrainError>`
Conversational music creation endpoint. Supports session-based conversations.

### `get_emotions() -> Result<Value, MusicBrainError>`
Retrieves the complete 216-node emotion thesaurus.

### `health_check() -> Result<Value, MusicBrainError>`
Checks API server health and availability.

### `is_available() -> bool`
Quick boolean check for API availability.

## Error Handling

All functions return `Result<T, MusicBrainError>`, which provides detailed error information:

```rust
match generate(request).await {
    Ok(data) => { /* success */ }
    Err(MusicBrainError::Network(msg)) => {
        // Network connectivity issue
    }
    Err(MusicBrainError::ApiError { status, message }) => {
        // API returned error (e.g., status 500)
    }
    Err(MusicBrainError::Timeout(msg)) => {
        // Request timed out
    }
    Err(MusicBrainError::InvalidRequest(msg)) => {
        // Request validation failed
    }
    // ... other error types
}
```

## Performance Considerations

- **Connection pooling**: Reuses connections for better performance
- **Idle timeout**: Connections kept alive for 90 seconds
- **Max idle per host**: Up to 6 idle connections per host
- **Global client**: Single client instance shared across all requests

## Thread Safety

- All functions are `async` and can be called from any async context
- The global client uses `OnceCell` for thread-safe initialization
- All operations are safe for concurrent use

## Migration from Old Implementation

The enhanced bridge is a drop-in replacement. The function signatures remain compatible:

**Before:**
```rust
pub async fn generate(request: GenerateRequest) -> Result<Value, Box<dyn std::error::Error>>
```

**After:**
```rust
pub async fn generate(request: GenerateRequest) -> Result<Value, MusicBrainError>
```

The error type is more specific and provides better error information, but the usage pattern remains the same.

## Testing

To test the bridge:

1. **Start the Music Brain API server**:
   ```bash
   python brain_server.py
   # or
   uvicorn music_brain.api:app --host 127.0.0.1 --port 8000
   ```

2. **Run health check**:
   ```rust
   let health = health_check().await?;
   ```

3. **Test generation**:
   ```rust
   let request = GenerateRequest { /* ... */ };
   let result = generate(request).await?;
   ```

## Future Enhancements

Potential future improvements:
- [ ] Request/response logging
- [ ] Metrics collection (request count, latency, etc.)
- [ ] Circuit breaker pattern for fault tolerance
- [ ] Request rate limiting
- [ ] Caching for `get_emotions()` responses
- [ ] WebSocket support for streaming responses
