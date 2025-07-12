# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Voice Gemini App (音声対話型AIアプリケーション) project that enables voice-based interactions with Google's Gemini AI. The project uses a post-recording processing approach where audio is recorded, converted to text, processed by Gemini AI, and the response is synthesized back to audio.

## Key Technologies

- **genai-processors**: Core async pipeline processing framework for chaining AI operations
- **vibe-logger**: AI-optimized structured logging library
- **sounddevice**: Cross-platform audio recording
- **Whisper/Google Cloud Speech**: Speech-to-text processing
- **Gemini API**: AI conversation processing
- **gTTS**: Text-to-speech synthesis

## Project Structure

The intended project structure follows this pattern:

```
voice-gemini-app/
├── docs/                    # Documentation
├── src/
│   ├── audio/              # Audio recording and synthesis
│   ├── processors/         # genai-processors implementations
│   └── utils/              # Logger setup and configuration
├── logs/                   # Log output directory
└── tests/                  # Test files
```

## Development Commands

Since the project is not yet implemented, here are the intended commands based on the documentation:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies (when requirements.txt is created)
pip install -r requirements.txt

# Run the application (planned)
python src/main.py

# Check available audio devices
python -m sounddevice
```

## Architecture Overview

The application follows a three-phase architecture:

1. **Audio Recording Phase** (Synchronous)
   - Records audio from microphone to WAV file buffer
   - Uses sounddevice for cross-platform compatibility

2. **Async Processing Pipeline** (genai-processors)
   - AudioProcessor → STTProcessor → GeminiProcessor → TTSProcessor
   - Each processor is a separate async unit that can be chained
   - Processors communicate via async streams of ProcessorParts

3. **Logging Layer** (vibe-logger)
   - Structured logging optimized for AI analysis
   - Async wrapper for non-blocking logging in the pipeline
   - Tracks metrics: processing times, token usage, errors

## Key Implementation Notes

- **Async Processing**: All processors must implement the genai-processors Processor interface
- **Error Handling**: Each processor should log errors with full context using vibe-logger
- **Audio Format**: Default to 16kHz WAV format for compatibility
- **Whisper Models**: Start with 'base' model for balance of speed/accuracy
- **Environment Variables**: Configuration via .env file (see docs/voice-gemini-app.md)

## Current Status

- Project documentation is complete (docs/voice-gemini-app.md)
- Implementation has not yet started
- Focus on building the processor pipeline architecture first