"""Tests for AST parser."""

import pytest
from claims_analyzer.ast_parser import ClaimsSystemParser


def test_parser_initialization():
    """Test parser can be initialized."""
    parser = ClaimsSystemParser("../claims_system")
    assert parser.source_dir.name == "claims_system"


# Add more tests here
