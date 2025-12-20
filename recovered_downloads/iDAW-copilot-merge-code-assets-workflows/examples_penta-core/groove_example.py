"""
Example: Real-time groove analysis from audio
"""

import numpy as np
from penta_core import GrooveEngine

def generate_test_audio(sample_rate: float, duration: float, tempo: float) -> np.ndarray:
    """Generate test audio with clicks at tempo"""
    samples = int(sample_rate * duration)
    audio = np.zeros(samples, dtype=np.float32)
    
    # Add clicks at beat positions
    beats_per_second = tempo / 60.0
    samples_per_beat = int(sample_rate / beats_per_second)
    
    for i in range(0, samples, samples_per_beat):
        if i < samples - 100:
            # Short click
            audio[i:i+100] = np.linspace(1.0, 0.0, 100)
    
    return audio

def main():
    sample_rate = 48000.0
    tempo = 120.0  # BPM
    
    # Initialize groove engine
    groove = GrooveEngine(
        sample_rate=sample_rate,
        min_tempo=60.0,
        max_tempo=180.0
    )
    
    print(f"Generating test audio at {tempo} BPM...")
    audio = generate_test_audio(sample_rate, duration=4.0, tempo=tempo)
    
    # Process audio in chunks (simulating real-time)
    chunk_size = 512
    print("\nProcessing audio...")
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            # Pad last chunk
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        groove.process_audio(chunk)
    
    # Get analysis results
    analysis = groove.get_analysis()
    
    print("\n=== Groove Analysis ===")
    print(f"Detected Tempo: {analysis['tempo']:.1f} BPM")
    print(f"Tempo Confidence: {analysis['tempo_confidence']:.2f}")
    print(f"Time Signature: {analysis['time_signature']}")
    print(f"Swing Amount: {analysis['swing']:.2f}")
    print(f"Onset Count: {analysis['onset_count']}")
    
    # Test quantization
    test_timestamp = 12000  # arbitrary sample position
    quantized = groove.quantize_timestamp(test_timestamp)
    print(f"\nQuantization test:")
    print(f"Original: {test_timestamp}, Quantized: {quantized}")


if __name__ == '__main__':
    main()
