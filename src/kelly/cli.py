"""Command-line interface for Kelly."""
from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from kelly.core.emotion_thesaurus import EmotionThesaurus
from kelly.core.intent_processor import IntentProcessor, Wound
from kelly.core.midi_generator import MidiGenerator

app = typer.Typer(
    name="kelly",
    help="Kelly - Therapeutic iDAW translating emotions to music"
)
console = Console()


@app.command()
def list_emotions() -> None:
    """List all available emotions in the thesaurus."""
    thesaurus = EmotionThesaurus()
    
    table = Table(title="Kelly Emotion Thesaurus")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Category", style="green")
    table.add_column("Intensity", style="yellow")
    table.add_column("Valence", style="blue")
    table.add_column("Arousal", style="red")
    
    for node in thesaurus.nodes.values():
        table.add_row(
            str(node.id),
            node.name,
            node.category.value,
            f"{node.intensity:.2f}",
            f"{node.valence:+.2f}",
            f"{node.arousal:.2f}"
        )
    
    console.print(table)


@app.command()
def process(
    wound: str = typer.Argument(..., help="Description of the wound/trigger"),
    intensity: float = typer.Option(0.7, help="Intensity of the wound (0.0-1.0)"),
    output: Optional[Path] = typer.Option(None, help="Output MIDI file path"),
    tempo: int = typer.Option(120, help="Tempo in BPM"),
    groove: str = typer.Option("straight", help="Groove template (straight/swing/syncopated)")
) -> None:
    """
    Process a therapeutic wound and generate music.
    
    Example:
        kelly process "feeling of loss" --intensity 0.8 --output output.mid
    """
    console.print(f"\n[bold cyan]Processing wound:[/bold cyan] {wound}")
    console.print(f"[cyan]Intensity:[/cyan] {intensity}\n")
    
    # Process intent
    processor = IntentProcessor()
    wound_obj = Wound(
        description=wound,
        intensity=intensity,
        source="user_input"
    )
    
    result = processor.process_intent(wound_obj)
    
    # Display results
    emotion = result["emotion"]
    console.print(f"[bold green]Mapped Emotion:[/bold green] {emotion.name}")
    console.print(f"  Category: {emotion.category.value}")
    console.print(f"  Valence: {emotion.valence:+.2f}")
    console.print(f"  Arousal: {emotion.arousal:.2f}\n")
    
    console.print(f"[bold yellow]Rule Breaks:[/bold yellow]")
    for rb in result["rule_breaks"]:
        console.print(f"  • {rb.rule_type}: {rb.description} (severity: {rb.severity:.2f})")
    
    # Generate MIDI
    generator = MidiGenerator(tempo=tempo)
    params = result["musical_params"]
    
    mode = params.get("mode", "minor")
    allow_dissonance = params.get("allow_dissonance", False)
    
    chord_progression = generator.generate_chord_progression(
        mode=mode,
        allow_dissonance=allow_dissonance
    )
    
    # Save or display
    if output:
        midi_file = generator.create_midi_file(chord_progression, groove, str(output))
        console.print(f"\n[bold green]✓[/bold green] MIDI file saved to: {output}")
    else:
        console.print(f"\n[dim]No output file specified. Use --output to save MIDI.[/dim]")
    
    console.print(f"\n[bold]Musical Parameters:[/bold]")
    console.print(f"  Mode: {mode}")
    console.print(f"  Tempo: {tempo} BPM")
    console.print(f"  Groove: {groove}")
    console.print(f"  Dissonance: {allow_dissonance}\n")


@app.command()
def version() -> None:
    """Show Kelly version."""
    from kelly import __version__
    rprint(f"[bold cyan]Kelly[/bold cyan] version [green]{__version__}[/green]")


if __name__ == "__main__":
    app()
