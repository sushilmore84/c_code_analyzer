# Claims Business Rule Analyzer

Extract business rules from legacy C claims processing systems using AST analysis and LLM.

## Setup
```bash
# Install dependencies
uv sync

# Configure API key
cp .env.template .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage
```bash
# Analyze C codebase
uv run python src/claims_analyzer/ast_parser.py ../claims_system

# Extract rules with LLM
uv run python src/claims_analyzer/llm_extractor.py

# Run tests
uv run pytest

# Format code
uv run black src/

# Lint
uv run ruff check src/
```

## Project Structure
