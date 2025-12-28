"""
CLI Commands for OSC Learning Module

Provides command-line interface for recording, training, and generating OSC sequences.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import time

from music_brain.ml_osc import (
    OSCRecorder,
    OSCSequence,
    OSCDataset,
    OSCMessageEncoder,
    OSCPatternLearner,
    OSCPredictor,
    OSCSequenceGenerator,
)


def cmd_record(args):
    """Record OSC messages to a file."""
    print(f"Starting OSC recorder on {args.ip}:{args.port}")
    print(f"Recording for {args.duration} seconds...")
    print("Send OSC messages to this address to record them.")
    
    recorder = OSCRecorder(port=args.port, ip=args.ip)
    
    metadata = {
        "description": args.description or "OSC recording",
        "port": args.port,
        "ip": args.ip,
    }
    
    sequence = recorder.start_recording(metadata=metadata)
    
    try:
        # Record for specified duration
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    
    sequence = recorder.stop_recording()
    
    # Save to file
    output_path = Path(args.output)
    sequence.save(output_path)
    
    print(f"\nRecorded {len(sequence.messages)} messages")
    print(f"Duration: {sequence.duration:.2f} seconds")
    print(f"Saved to: {output_path}")


def cmd_play(args):
    """Play back a recorded OSC sequence."""
    print(f"Loading sequence from {args.input}")
    sequence = OSCSequence.load(Path(args.input))
    
    print(f"Sequence has {len(sequence.messages)} messages over {sequence.duration:.2f} seconds")
    
    if args.show:
        # Just display the sequence
        for i, msg in enumerate(sequence.messages):
            print(f"{i:4d} [{msg.timestamp:8.3f}s] {msg.address} {msg.args}")
        return
    
    # TODO: Implement actual OSC playback to a target
    print("Playback not yet implemented - use --show to view messages")


def cmd_train(args):
    """Train an OSC pattern learning model."""
    print(f"Loading dataset from {args.data_dir}")
    
    # Load dataset
    dataset = OSCDataset(Path(args.data_dir))
    print(f"Loaded {len(dataset)} sequences")
    
    if len(dataset) == 0:
        print("Error: No sequences found in dataset")
        return
    
    # Get training data
    print(f"Preparing training data with context length {args.context_length}")
    X, y = dataset.get_all_training_pairs(context_length=args.context_length)
    print(f"Generated {len(X)} training samples")
    
    # Create model
    print(f"Creating model with hidden_dim={args.hidden_dim}, layers={args.num_layers}")
    predictor = OSCPredictor(
        feature_dim=dataset.encoder.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate
    )
    
    # Train
    print(f"Training for {args.epochs} epochs...")
    predictor.train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )
    
    # Save model
    output_path = Path(args.output)
    predictor.save(output_path)
    
    # Save encoder
    encoder_path = output_path.parent / f"{output_path.stem}_encoder.json"
    with open(encoder_path, 'w') as f:
        json.dump({
            'address_vocab': dataset.encoder.address_vocab,
            'max_args': dataset.encoder.max_args,
        }, f, indent=2)
    
    print(f"Model saved to {output_path}")
    print(f"Encoder saved to {encoder_path}")


def cmd_generate(args):
    """Generate OSC sequences using a trained model."""
    print(f"Loading model from {args.model}")
    
    # Load encoder
    encoder_path = Path(args.model).parent / f"{Path(args.model).stem}_encoder.json"
    with open(encoder_path, 'r') as f:
        encoder_data = json.load(f)
    
    encoder = OSCMessageEncoder(
        address_vocab=encoder_data['address_vocab'],
        max_args=encoder_data['max_args']
    )
    
    # Load model
    predictor = OSCPredictor(
        feature_dim=encoder.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    predictor.load(Path(args.model))
    
    # Load seed sequence
    print(f"Loading seed sequence from {args.seed}")
    seed_sequence = OSCSequence.load(Path(args.seed))
    
    # Generate
    print(f"Generating {args.length} messages with temperature={args.temperature}")
    generator = OSCSequenceGenerator(predictor, encoder)
    generated = generator.generate(
        seed_sequence=seed_sequence,
        length=args.length,
        temperature=args.temperature,
        context_length=args.context_length
    )
    
    # Save
    output_path = Path(args.output)
    generated.save(output_path)
    
    print(f"Generated sequence saved to {output_path}")
    print(f"Total messages: {len(generated.messages)}")
    print(f"Duration: {generated.duration:.2f} seconds")


def cmd_info(args):
    """Show information about a dataset or sequence."""
    path = Path(args.path)
    
    if path.is_file():
        # Single sequence
        sequence = OSCSequence.load(path)
        print(f"Sequence: {path}")
        print(f"Messages: {len(sequence.messages)}")
        print(f"Duration: {sequence.duration:.2f} seconds")
        print(f"Metadata: {json.dumps(sequence.metadata, indent=2)}")
        
        # Show unique addresses
        addresses = set(msg.address for msg in sequence.messages)
        print(f"\nUnique OSC addresses ({len(addresses)}):")
        for addr in sorted(addresses):
            count = sum(1 for msg in sequence.messages if msg.address == addr)
            print(f"  {addr}: {count} messages")
    
    elif path.is_dir():
        # Dataset
        dataset = OSCDataset(path)
        print(f"Dataset: {path}")
        print(f"Sequences: {len(dataset)}")
        print(f"Total messages: {sum(len(seq.messages) for seq in dataset.sequences)}")
        print(f"Vocabulary size: {dataset.encoder.vocab_size}")
        print(f"Feature dimension: {dataset.encoder.feature_dim}")
        
        # Show addresses
        print(f"\nOSC addresses in dataset:")
        for addr, idx in sorted(dataset.encoder.address_vocab.items(), key=lambda x: x[1]):
            print(f"  [{idx:3d}] {addr}")
    else:
        print(f"Error: {path} does not exist")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Kelly OSC Learning - ML-powered OSC sequence tools"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Record command
    record_parser = subparsers.add_parser('record', help='Record OSC messages')
    record_parser.add_argument('output', help='Output JSON file')
    record_parser.add_argument('--port', type=int, default=8000, help='OSC port to listen on')
    record_parser.add_argument('--ip', default='127.0.0.1', help='IP address to bind to')
    record_parser.add_argument('--duration', type=float, default=10.0, help='Recording duration in seconds')
    record_parser.add_argument('--description', help='Description of the recording')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play back OSC sequence')
    play_parser.add_argument('input', help='Input JSON sequence file')
    play_parser.add_argument('--show', action='store_true', help='Just show messages, don\'t send')
    play_parser.add_argument('--target-port', type=int, default=9000, help='Target OSC port')
    play_parser.add_argument('--target-ip', default='127.0.0.1', help='Target IP address')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train OSC pattern model')
    train_parser.add_argument('data_dir', help='Directory containing training sequences')
    train_parser.add_argument('output', help='Output model file (.pt)')
    train_parser.add_argument('--context-length', type=int, default=5, help='Number of previous messages for context')
    train_parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
    train_parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--validation-split', type=float, default=0.1, help='Validation split fraction')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate OSC sequences')
    generate_parser.add_argument('model', help='Trained model file (.pt)')
    generate_parser.add_argument('seed', help='Seed sequence JSON file')
    generate_parser.add_argument('output', help='Output sequence JSON file')
    generate_parser.add_argument('--length', type=int, default=50, help='Number of messages to generate')
    generate_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    generate_parser.add_argument('--context-length', type=int, default=5, help='Context length')
    generate_parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension (must match model)')
    generate_parser.add_argument('--num-layers', type=int, default=2, help='Number of layers (must match model)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show dataset/sequence info')
    info_parser.add_argument('path', help='Path to sequence file or dataset directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to command handlers
    commands = {
        'record': cmd_record,
        'play': cmd_play,
        'train': cmd_train,
        'generate': cmd_generate,
        'info': cmd_info,
    }
    
    if args.command in commands:
        try:
            commands[args.command](args)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
