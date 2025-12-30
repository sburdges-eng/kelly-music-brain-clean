# Google Services Configuration

## google-services.json

This file is required for Firebase push notifications demo in the JUCE examples.

### Setup Instructions

1. Create a Firebase project at https://console.firebase.google.com/
2. Add an Android app to your Firebase project
3. Download your `google-services.json` file from Firebase Console
4. Replace the placeholder API key in `google-services.json` with your actual configuration

### Security Note

**IMPORTANT**: Never commit real API keys to version control. The `google-services.json` file in this repository contains only placeholder values. Each developer should use their own Firebase project configuration for testing.

For production use, ensure API keys are:
- Stored securely (environment variables, secrets management)
- Restricted to specific app signatures
- Never committed to public repositories
