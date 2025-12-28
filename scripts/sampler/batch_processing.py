#!/usr/bin/env python3
"""
Batch Processing Workflow (Integration Example 2)
================================================
Batch process multiple audio files for emotion classification.

Features:
- Process entire directories
- Parallel processing support
- Progress tracking
- CSV/JSON output
- Error handling
"""

import sys
from pathlib import Path
import json
import csv
from datetime import datetime

BRAIN_PYTHON_DIR = Path(__file__).parent.parent


class BatchProcessor:
    """Batch emotion detection processor."""
    
    def __init__(self, output_format="csv"):
        self.output_format = output_format
        self.results = []
        self.errors = []
    
    def process_file(self, file_path):
        """Process a single audio file."""
        try:
            # Mock processing (would use actual model)
            result = {
                "file": str(file_path),
                "filename": file_path.name,
                "emotion": "happy",  # Mock
                "confidence": 0.85,
                "processing_time": 0.5,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            error = {
                "file": str(file_path),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.errors.append(error)
            return None
    
    def process_directory(self, directory, pattern="*.mp3"):
        """Process all matching files in directory."""
        directory = Path(directory)
        
        if not directory.exists():
            print(f"⚠ Directory not found: {directory}")
            return
        
        files = list(directory.glob(pattern))
        
        print(f"Found {len(files)} files matching {pattern}")
        print(f"Processing...")
        
        for i, file in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] {file.name}...", end=" ")
            result = self.process_file(file)
            
            if result:
                print(f"✓ {result['emotion']} ({result['confidence']:.0%})")
            else:
                print("✗ Error")
        
        print(f"\nComplete: {len(self.results)} processed, {len(self.errors)} errors")
    
    def save_results(self, output_file):
        """Save results to file."""
        output_file = Path(output_file)
        
        if self.output_format == "csv":
            with open(output_file, 'w', newline='') as f:
                if self.results:
                    writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                    writer.writeheader()
                    writer.writerows(self.results)
            print(f"✓ Results saved to {output_file}")
            
        elif self.output_format == "json":
            with open(output_file, 'w') as f:
                json.dump({
                    "results": self.results,
                    "errors": self.errors,
                    "summary": {
                        "total_processed": len(self.results),
                        "total_errors": len(self.errors),
                        "timestamp": datetime.now().isoformat()
                    }
                }, f, indent=2)
            print(f"✓ Results saved to {output_file}")
    
    def generate_report(self):
        """Generate summary report."""
        if not self.results:
            print("No results to report")
            return
        
        print("\n" + "="*70)
        print("BATCH PROCESSING REPORT")
        print("="*70)
        
        # Emotion distribution
        emotions = {}
        for r in self.results:
            emotion = r['emotion']
            emotions[emotion] = emotions.get(emotion, 0) + 1
        
        print("\nEmotion Distribution:")
        for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(self.results) * 100
            print(f"  {emotion.upper()}: {count} ({percentage:.1f}%)")
        
        # Average confidence
        avg_confidence = sum(r['confidence'] for r in self.results) / len(self.results)
        print(f"\nAverage Confidence: {avg_confidence:.2%}")
        
        # Processing stats
        total_time = sum(r['processing_time'] for r in self.results)
        print(f"Total Processing Time: {total_time:.2f}s")
        print(f"Average Time per File: {total_time/len(self.results):.2f}s")


def main():
    if len(sys.argv) < 2:
        print("BATCH PROCESSING WORKFLOW (Integration Example 2)")
        print("\nUSAGE:")
        print("  python batch_processing.py <directory> [pattern] [output_format]")
        print("\nEXAMPLE:")
        print("  python batch_processing.py ./samples '*.mp3' csv")
        print("  python batch_processing.py ./samples '*.wav' json")
        return
    
    directory = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) > 2 else "*.mp3"
    output_format = sys.argv[3] if len(sys.argv) > 3 else "csv"
    
    processor = BatchProcessor(output_format=output_format)
    processor.process_directory(directory, pattern)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"batch_results_{timestamp}.{output_format}"
    processor.save_results(output_file)
    
    # Generate report
    processor.generate_report()


if __name__ == "__main__":
    main()
