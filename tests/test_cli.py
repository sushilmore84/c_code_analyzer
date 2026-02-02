"""Tests for CLI."""

from typer.testing import CliRunner
from claims_analyzer.cli import app

runner = CliRunner()


def test_version():
    """Test version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Claims Business Rule Analyzer" in result.stdout


# Add more tests here
