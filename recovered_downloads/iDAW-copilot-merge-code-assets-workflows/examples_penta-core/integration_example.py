"""
Example: Complete integration with OSC communication
"""

import numpy as np
import time
from penta_core import PentaCore

def main():
    print("=== Penta Core Integration Example ===\n")
    
    # Initialize integrated engine
    penta = PentaCore(sample_rate=48000.0)
    
    # Start OSC communication
    print("Starting OSC communication...")
    if penta.start_osc():
        print("OSC server started on port 8000")
        print("OSC client sending to port 9000\n")
    else:
        print("Failed to start OSC\n")
    
    # Simulate processing loop
    print("Processing audio and MIDI...\n")
    
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Generate test audio (simple sine wave)
        duration = 0.1  # 100ms chunks
        samples = int(penta.sample_rate * duration)
        audio = np.sin(2 * np.pi * 440 * np.arange(samples) / penta.sample_rate)
        audio = audio.astype(np.float32)
        
        # Generate test MIDI (C major chord)
        midi_notes = [
            (60, 80),
            (64, 75),
            (67, 70),
        ]
        
        # Process
        penta.process(audio, midi_notes)
        
        # Get state
        state = penta.get_state()
        
        # Display results
        print(f"Chord: {state['chord']['name']} (confidence: {state['chord']['confidence']:.2f})")
        print(f"Tempo: {state['groove']['tempo']:.1f} BPM")
        print(f"CPU: {state['diagnostics']['cpu_usage']:.1f}%")
        print(f"Latency: {state['diagnostics']['average_latency_ms']:.2f} ms")
        
        # Send OSC message
        penta.osc.send_message(
            '/penta/state',
            state['chord']['root'],
            state['groove']['tempo']
        )
        
        # Check for incoming OSC messages
        msg = penta.osc.receive_message()
        if msg:
            print(f"Received OSC: {msg['address']} - {msg['arguments']}")
        
        print()
        time.sleep(0.5)
    
    # Cleanup
    print("\nStopping OSC communication...")
    penta.stop_osc()
    
    # Final diagnostics
    print("\n=== Final Diagnostics ===")
    print(penta.diagnostics.get_performance_report())


if __name__ == '__main__':
    main()
