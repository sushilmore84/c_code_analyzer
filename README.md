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

# In Mac If you already have .venv file
uv venv
source .venv/bin/activate

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

## LLM Related
openai - gpt-4o-mini-2024-07-18
anthropic - claude-sonnet-4-20250514

## Execution Flow 

# Run all 5 stages
cd /path/to/claims-analyzer

# Stage 1
uv run python src/claims_analyzer/ast_parser.py ../claims_system

# Stage 2
uv run python src/claims_analyzer/graph_builder.py output/enhanced_analysis.json

# Stage 3
uv run python src/claims_analyzer/rule_modernizer.py output/graph_analysis_for_llm.json

# Stage 4
uv run python src/claims_analyzer/llm_synthesizer.py output/modernized_emergency_rule.json

# Stage 5
uv run python src/claims_analyzer/llm_markdown_doc_generator.py

## 📋 Checklist Format
# Stage 1: AST Parser
  Command: uv run python src/claims_analyzer/ast_parser.py ../claims_system
  Output: output/enhanced_analysis.json
  
# Stage 2: Graph Builder
  Command: uv run python src/claims_analyzer/graph_builder.py output/enhanced_analysis.json
  Output: output/graph_analysis_for_llm.json + 3 PNG files
  
# Stage 3: Rule Modernizer
  Command: uv run python src/claims_analyzer/rule_modernizer.py output/graph_analysis_for_llm.json
  Output: output/modernized_emergency_rule.json + MD file
  
# Stage 4: LLM Synthesizer (requires API key)
  Command: uv run python src/claims_analyzer/llm_synthesizer.py output/modernized_emergency_rule.json
  Output: output/dmn_emergency_rule.json + MD + HTML

