# Media Production Guide

This guide covers media production concepts and workflows for the penta-core team.

## Overview

Media production encompasses the creation of audio, video, and interactive content. This includes music production, podcast creation, video editing, streaming, and multimedia application development.

## Audio Production

### Production Stages

1. **Pre-Production**
   - Concept development
   - Reference gathering
   - Session setup (tempo, key, structure)
   - Equipment preparation

2. **Recording**
   - Microphone selection and placement
   - Gain staging
   - Performance capture
   - Multiple takes and overdubs

3. **Editing**
   - Comping (selecting best takes)
   - Timing correction
   - Pitch correction
   - Noise reduction

4. **Mixing**
   - Balance and levels
   - EQ and tonal shaping
   - Dynamics processing
   - Spatial effects (reverb, delay)
   - Automation

5. **Mastering**
   - Final EQ adjustments
   - Multiband compression
   - Stereo enhancement
   - Limiting and loudness
   - Format conversion

### Audio Formats

| Format | Type | Quality | Use Case |
|--------|------|---------|----------|
| WAV | Uncompressed | Lossless | Production, archival |
| AIFF | Uncompressed | Lossless | Production (Apple) |
| FLAC | Compressed | Lossless | Distribution, streaming |
| MP3 | Compressed | Lossy | Consumer distribution |
| AAC | Compressed | Lossy | Streaming, mobile |
| OGG | Compressed | Lossy | Gaming, open source |

### Loudness Standards

| Standard | Target | Platform |
|----------|--------|----------|
| -14 LUFS | Integrated | Spotify, YouTube |
| -16 LUFS | Integrated | Apple Music |
| -24 LUFS | Integrated | Broadcast (EBU R128) |
| -24 LKFS | Integrated | Broadcast (ATSC A/85) |

## Video Production

### Video Formats and Codecs

| Codec | Type | Use Case |
|-------|------|----------|
| H.264 | Lossy | Web distribution, streaming |
| H.265/HEVC | Lossy | 4K/HDR content |
| ProRes | Intermediate | Professional editing |
| DNxHD/HR | Intermediate | Professional editing |
| VP9 | Lossy | YouTube, web |
| AV1 | Lossy | Next-gen streaming |

### Frame Rates

| Rate | Use Case |
|------|----------|
| 24 fps | Film, cinematic look |
| 25 fps | PAL broadcast |
| 29.97 fps | NTSC broadcast |
| 30 fps | Web, general use |
| 60 fps | Sports, gaming, smooth motion |
| 120 fps | Slow motion, VR |

### Resolution Standards

| Resolution | Name | Aspect Ratio |
|------------|------|--------------|
| 1920×1080 | Full HD (1080p) | 16:9 |
| 2560×1440 | QHD (1440p) | 16:9 |
| 3840×2160 | 4K UHD | 16:9 |
| 4096×2160 | 4K DCI | 1.9:1 |
| 7680×4320 | 8K UHD | 16:9 |

## Streaming and Live Production

### Streaming Protocols

| Protocol | Description | Latency |
|----------|-------------|---------|
| RTMP | Real-Time Messaging Protocol | 2-5 sec |
| HLS | HTTP Live Streaming | 6-30 sec |
| DASH | Dynamic Adaptive Streaming | 6-30 sec |
| WebRTC | Web Real-Time Communication | < 1 sec |
| SRT | Secure Reliable Transport | 1-2 sec |

### Streaming Architecture

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Source    │ ──▶  │   Encoder   │ ──▶  │    CDN      │
│  (Camera/   │      │  (OBS/HW)   │      │  (Delivery) │
│   Screen)   │      └─────────────┘      └─────────────┘
└─────────────┘                                  │
                                                 ▼
                                          ┌─────────────┐
                                          │   Player    │
                                          │  (Browser/  │
                                          │    App)     │
                                          └─────────────┘
```

### Bitrate Recommendations

| Quality | Video Bitrate | Audio Bitrate |
|---------|--------------|---------------|
| 720p30 | 2,500-4,000 kbps | 128 kbps |
| 1080p30 | 4,000-6,000 kbps | 192 kbps |
| 1080p60 | 6,000-9,000 kbps | 256 kbps |
| 4K30 | 13,000-34,000 kbps | 256 kbps |

## Development Tools

### FFmpeg

FFmpeg is essential for media processing:

```bash
# Convert video format
ffmpeg -i input.mov -c:v libx264 -crf 23 -c:a aac output.mp4

# Extract audio from video
ffmpeg -i video.mp4 -vn -acodec libmp3lame audio.mp3

# Create video from images
ffmpeg -framerate 30 -i image_%03d.png -c:v libx264 output.mp4

# Stream to RTMP
ffmpeg -i source.mp4 -c:v libx264 -f flv rtmp://server/live/stream
```

### GStreamer

GStreamer provides a pipeline-based media framework:

```bash
# Play audio file
gst-launch-1.0 filesrc location=audio.mp3 ! decodebin ! audioconvert ! autoaudiosink

# Record from camera
gst-launch-1.0 v4l2src ! videoconvert ! x264enc ! mp4mux ! filesink location=output.mp4
```

### LibAV/FFmpeg API

```cpp
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>

class MediaReader {
    AVFormatContext* formatContext = nullptr;
    
public:
    MediaReader() = default;
    
    // Disable copy operations
    MediaReader(const MediaReader&) = delete;
    MediaReader& operator=(const MediaReader&) = delete;
    
    // Enable move operations
    MediaReader(MediaReader&& other) noexcept 
        : formatContext(other.formatContext) {
        other.formatContext = nullptr;
    }
    
    MediaReader& operator=(MediaReader&& other) noexcept {
        if (this != &other) {
            close();
            formatContext = other.formatContext;
            other.formatContext = nullptr;
        }
        return *this;
    }
    
    bool open(const char* filename) {
        if (avformat_open_input(&formatContext, filename, nullptr, nullptr) < 0) {
            return false;
        }
        
        if (avformat_find_stream_info(formatContext, nullptr) < 0) {
            // Clean up on failure to prevent resource leak
            avformat_close_input(&formatContext);
            return false;
        }
        
        return true;
    }
    
    void close() {
        if (formatContext) {
            avformat_close_input(&formatContext);
            formatContext = nullptr;
        }
    }
    
    ~MediaReader() {
        close();
    }
};
```

## Workflow Automation

### Task Runners

```javascript
// Example: Gulp task for audio processing
const gulp = require('gulp');
const exec = require('child_process').exec;

gulp.task('normalize-audio', function(cb) {
    exec('ffmpeg-normalize input/*.wav -o output/ -ext wav', 
         function(err, stdout, stderr) {
        console.log(stdout);
        cb(err);
    });
});
```

### Batch Processing

```python
import subprocess
from pathlib import Path

def batch_convert(input_dir: Path, output_dir: Path):
    """Convert all WAV files to MP3."""
    for wav_file in input_dir.glob("*.wav"):
        output_file = output_dir / wav_file.with_suffix(".mp3").name
        subprocess.run([
            "ffmpeg", "-i", str(wav_file),
            "-codec:a", "libmp3lame",
            "-qscale:a", "2",
            str(output_file)
        ])
```

## Best Practices

1. **Organize assets systematically** with clear naming conventions
2. **Use version control** for project files and scripts
3. **Create backups** of raw recordings and source files
4. **Document workflows** for reproducibility
5. **Calibrate monitoring** for consistent results
6. **Use reference material** for quality benchmarking
7. **Automate repetitive tasks** to reduce errors

## Resources

- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/)
- [Pro Tools Reference](https://www.avid.com/resource-center/pro-tools)
- [OBS Studio](https://obsproject.com/)
- [Blackmagic Design](https://www.blackmagicdesign.com/)
