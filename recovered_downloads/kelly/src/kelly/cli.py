"""Command-line interface for Kelly."""

from typing import Optional
from pathlib import Path

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    HAS_CLI_DEPS = True
except ImportError:
    HAS_CLI_DEPS = False

from kelly.core.emotion_thesaurus import EmotionThesaurus
from kelly.core.intent_processor import IntentProcessor, Wound
from kelly.core.midi_generator import MidiGenerator

if HAS_CLI_DEPS:
    app = typer.Typer(
        name="kelly",
        help="Kelly - Therapeutic iDAW translating emotions to music"
    )
    console = Console()


    @app.command()
    def list_emotions() -> None:
        """List all available emotions in the thesaurus."""
        thesaurus = EmotionThesaurus()
        
        table = Table(title="Kelly Emotion Thesaurus (72 emotions)")
        table.add_column("Name", style="magenta")
        table.add_column("Category", style="green")
        table.add_column("Intensity", style="yellow")
        table.add_column("Valence", style="blue")
        table.add_column("Arousal", style="red")
        
        for node in list(thesaurus.nodes.values())[:20]:
            table.add_row(
                node.name,
                node.category.value,
                f"{node.intensity:.2f}",
                f"{node.valence:+.2f}",
                f"{node.arousal:.2f}"
            )
        
        console.print(table)
        console.print(f"\n[dim]Showing 20 of {len(thesaurus.nodes)} emotions. Use --all to see all.[/dim]")


    @app.command()
    def process(
        wound: str = typer.Argument(..., help="Description of the wound/trigger"),
        intensity: float = typer.Option(0.7, help="Intensity (0.0-1.0)"),
        output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output MIDI file"),
        tempo: int = typer.Option(100, help="Tempo in BPM"),
        key: str = typer.Option("C", help="Musical key"),
        groove: str = typer.Option("straight", help="Groove (straight/swing/behind)")
    ) -> None:
        """Process an emotional wound and generate music."""
        console.print(f"\n[bold cyan]Processing wound:[/bold cyan] {wound}")
        console.print(f"[cyan]Intensity:[/cyan] {intensity}\n")
        
        processor = IntentProcessor()
        wound_obj = Wound(description=wound, intensity=intensity, source="cli")
        result = processor.process_intent(wound_obj)
        
        emotion = result.emotion
        console.print(f"[bold green]Mapped Emotion:[/bold green] {emotion.name}")
        console.print(f"  Category: {emotion.category.value}")
        console.print(f"  Valence: {emotion.valence:+.2f}")
        console.print(f"  Arousal: {emotion.arousal:.2f}\n")
        
        console.print("[bold yellow]Rule Breaks:[/bold yellow]")
        for rb in result.rule_breaks[:3]:
            console.print(f"  • {rb.rule_type}: {rb.description}")
        
        if output:
            generator = MidiGenerator(tempo=tempo, key=key, mode=emotion.musical_mapping.mode)
            progression = generator.generate_chord_progression(bars=4)
            midi_path = generator.create_midi_file(progression, groove, str(output))
            if midi_path:
                console.print(f"\n[bold green]✓[/bold green] MIDI saved to: {output}")
        else:
            console.print(f"\n[dim]Use --output to save MIDI file.[/dim]")
        
        console.print(f"\n[bold]Musical Parameters:[/bold]")
        console.print(f"  Mode: {emotion.musical_mapping.mode}")
        console.print(f"  Tempo: {tempo} BPM")
        console.print(f"  Key: {key}")
        console.print(f"  Groove: {groove}\n")


    @app.command()
    def generate(
        emotion: str = typer.Argument(..., help="Emotion name"),
        output: Path = typer.Option(Path("kelly_output.mid"), "-o", "--output"),
        bars: int = typer.Option(4, help="Number of bars"),
        tempo: int = typer.Option(100, help="Tempo in BPM"),
        key: str = typer.Option("C", help="Musical key"),
    ) -> None:
        """Quick generate MIDI from emotion name."""
        thesaurus = EmotionThesaurus()
        node = thesaurus.get_emotion(emotion)
        
        if not node:
            console.print(f"[red]Unknown emotion: {emotion}[/red]")
            console.print(f"Try: {', '.join(list(thesaurus.name_index.keys())[:10])}...")
            raise typer.Exit(1)
        
        console.print(f"[cyan]Generating from:[/cyan] {node.name} ({node.category.value})")
        
        generator = MidiGenerator(
            tempo=int(tempo * node.musical_mapping.tempo_modifier),
            key=key,
            mode=node.musical_mapping.mode,
        )
        
        midi_path = generator.create_full_arrangement(
            bars=bars,
            output_path=str(output)
        )
        
        if midi_path:
            console.print(f"[green]✓[/green] Generated: {output}")
        else:
            console.print("[yellow]Warning: mido not installed, MIDI not saved[/yellow]")


    @app.command()
    def analyze(
        wound: str = typer.Argument(..., help="Describe your emotional state")
    ) -> None:
        """Analyze emotional wound and suggest musical approach."""
        processor = IntentProcessor()
        wound_obj = Wound(description=wound, intensity=0.7)
        result = processor.process_intent(wound_obj)
        
        console.print("\n[bold]═══ EMOTIONAL ANALYSIS ═══[/bold]\n")
        
        console.print(f"[cyan]Wound Type:[/cyan] {result.wound.wound_type.value}")
        console.print(f"[cyan]Primary Emotion:[/cyan] {result.emotion.name}")
        console.print(f"[cyan]Category:[/cyan] {result.emotion.category.value}")
        
        console.print(f"\n[bold]Emotional Dimensions:[/bold]")
        console.print(f"  Valence: {result.emotion.valence:+.2f} ({'positive' if result.emotion.valence > 0 else 'negative'})")
        console.print(f"  Arousal: {result.emotion.arousal:.2f} ({'high energy' if result.emotion.arousal > 0.5 else 'low energy'})")
        console.print(f"  Intensity: {result.emotion.intensity:.2f}")
        
        console.print(f"\n[bold]Suggested Musical Approach:[/bold]")
        mapping = result.emotion.musical_mapping
        console.print(f"  Mode: {mapping.mode}")
        console.print(f"  Tempo: {int(100 * mapping.tempo_modifier)} BPM")
        console.print(f"  Articulation: {mapping.articulation}")
        console.print(f"  Register: {mapping.register_preference}")
        
        console.print(f"\n[bold]Rule Breaks to Consider:[/bold]")
        for rb in result.rule_breaks[:3]:
            console.print(f"  • [yellow]{rb.rule_type}[/yellow]")
            console.print(f"    {rb.justification}")
        
        console.print(f"\n[bold]Narrative Arc:[/bold] {result.narrative_arc}")
        console.print(f"[bold]Imagery:[/bold] {', '.join(result.imagery)}")
        console.print()


    @app.command()
    def version() -> None:
        """Show Kelly version."""
        from kelly import __version__
        rprint(f"[bold cyan]Kelly[/bold cyan] version [green]{__version__}[/green]")


    def main():
        app()


else:
    def main():
        print("CLI dependencies not installed. Run: pip install typer rich")
        print("\nYou can still use Kelly as a library:")
        print("  from kelly import EmotionThesaurus, IntentProcessor, MidiGenerator")


if __name__ == "__main__":
    main()
