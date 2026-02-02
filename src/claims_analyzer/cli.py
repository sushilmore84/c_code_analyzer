"""Command-line interface for claims analyzer."""

import typer
from rich.console import Console
from pathlib import Path

app = typer.Typer(
    name="claims-analyzer",
    help="Business rule extraction from C claims processing systems"
)
console = Console()


@app.command()
def analyze(
    source_dir: Path = typer.Argument(
        ..., 
        help="Path to C source directory",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        "output",
        "--output", "-o",
        help="Output directory for results"
    ),
):
    """Analyze C codebase and extract business rules."""
    console.print(f"[bold green]Analyzing:[/bold green] {source_dir}")
    console.print(f"[bold blue]Output:[/bold blue] {output_dir}")
    
    # Import here to avoid circular imports
    # from .ast_parser import ClaimsSystemParser
    
    console.print("[yellow]Add your analysis code here![/yellow]")


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"[bold]Claims Business Rule Analyzer[/bold]")
    console.print(f"Version: {__version__}")


if __name__ == "__main__":
    app()
