#!/usr/bin/env python3
"""
LLM Synthesizer - Convert modernized rules to DMN-style decision tables
Uses Claude API to generate structured decision tables from business rules
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import anthropic


@dataclass
class DMNCondition:
    """A single condition column in DMN table."""
    column_name: str
    input_expression: str  # What field to evaluate


@dataclass
class DMNRule:
    """A single row in DMN table."""
    rule_number: int
    conditions: Dict[str, str]  # column_name -> value
    result: str


@dataclass
class DMNTable:
    """Complete DMN-style decision table."""
    table_name: str
    description: str
    condition_columns: List[DMNCondition]
    result_column_name: str
    rules: List[DMNRule]
    hit_policy: str  # "FIRST", "ANY", "PRIORITY", etc.


class LLMSynthesizer:
    """Convert modernized business rules to DMN decision tables using Claude."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude API client."""
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def create_dmn_prompt(self, modernized_rule: Dict) -> str:
        """Create prompt for Claude to generate DMN table."""
        
        prompt = f"""You are a business rules expert. Convert this business rule into a DMN-style decision table.

BUSINESS RULE:
{json.dumps(modernized_rule, indent=2)}

Your task is to create a DMN (Decision Model and Notation) decision table with:

1. **Condition Columns**: Extract input conditions from the rule
   - Each condition should be a separate column
   - Use clear, business-friendly column names
   - Specify what field/expression each column evaluates

2. **Result Column**: What the rule produces (the action/outcome)

3. **Rules (Rows)**: All possible combinations that lead to the result
   - Use the conditions to create truth table rows
   - Show which combinations trigger which outcomes
   - Use "-" for "don't care" conditions

4. **Hit Policy**: Determine if this is FIRST, ANY, PRIORITY, etc.
   - FIRST: Use first matching rule
   - ANY: All matching rules must agree
   - PRIORITY: Use highest priority match

OUTPUT FORMAT (JSON):
{{
  "table_name": "Emergency_Classification",
  "description": "Business rule description",
  "hit_policy": "FIRST",
  "condition_columns": [
    {{
      "column_name": "Facility Type",
      "input_expression": "claim.facility_type"
    }},
    {{
      "column_name": "Emergency Procedure Valid",
      "input_expression": "verify_emergency_procedure(claim)"
    }}
  ],
  "result_column_name": "Classification",
  "rules": [
    {{
      "rule_number": 1,
      "conditions": {{
        "Facility Type": "ER",
        "Emergency Procedure Valid": "Yes"
      }},
      "result": "Emergency"
    }},
    {{
      "rule_number": 2,
      "conditions": {{
        "Facility Type": "ER",
        "Emergency Procedure Valid": "No"
      }},
      "result": "Emergency"
    }},
    {{
      "rule_number": 3,
      "conditions": {{
        "Facility Type": "Non-ER",
        "Emergency Procedure Valid": "No"
      }},
      "result": "Emergency"
    }},
    {{
      "rule_number": 4,
      "conditions": {{
        "Facility Type": "Non-ER",
        "Emergency Procedure Valid": "Yes"
      }},
      "result": "Not Emergency"
    }}
  ]
}}

IMPORTANT:
- Enumerate ALL possible combinations (create a truth table)
- Use simple, clear values (Yes/No, ER/Non-ER, etc.)
- The "result" should match what the rule sets (flag name, action, etc.)
- Include ALL rows needed to cover the logic completely
- Use business-friendly language

Generate the DMN table now:"""
        
        return prompt
    
    def call_claude(self, prompt: str) -> str:
        """Call Claude API to generate DMN table."""
        
        print("🤖 Calling Claude API...")
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Extract text from response
        result = response.content[0].text
        
        return result
    
    def parse_dmn_response(self, response: str) -> Dict:
        """Parse Claude's response to extract DMN table."""
        
        # Try to extract JSON from response
        # Claude might wrap it in ```json ... ```
        
        import re
        
        # Look for JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
        
        # Parse JSON
        try:
            dmn_data = json.loads(json_str)
            return dmn_data
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse JSON: {e}")
            print(f"Response was: {response[:500]}...")
            raise
    
    def generate_dmn_table(self, modernized_rule: Dict) -> Dict:
        """Generate DMN table from modernized rule using LLM."""
        
        print(f"\n📊 Generating DMN table for: {modernized_rule.get('rule_name', 'Unknown')}")
        
        # Create prompt
        prompt = self.create_dmn_prompt(modernized_rule)
        
        # Call Claude
        response = self.call_claude(prompt)
        
        # Parse response
        dmn_table = self.parse_dmn_response(response)
        
        print(f"✓ Generated DMN table with {len(dmn_table.get('rules', []))} rows")
        
        return dmn_table
    
    def format_dmn_as_markdown(self, dmn_table: Dict) -> str:
        """Format DMN table as markdown for display."""
        
        output = []
        
        output.append(f"# {dmn_table['table_name']}")
        output.append("")
        output.append(f"**Description:** {dmn_table['description']}")
        output.append(f"**Hit Policy:** {dmn_table['hit_policy']}")
        output.append("")
        
        # Build table header
        header = ["Rule #"]
        for col in dmn_table['condition_columns']:
            header.append(col['column_name'])
        header.append(dmn_table['result_column_name'])
        
        output.append("| " + " | ".join(header) + " |")
        output.append("|" + "|".join(["---"] * len(header)) + "|")
        
        # Build table rows
        for rule in dmn_table['rules']:
            row = [str(rule['rule_number'])]
            
            for col in dmn_table['condition_columns']:
                col_name = col['column_name']
                value = rule['conditions'].get(col_name, "-")
                row.append(value)
            
            row.append(rule['result'])
            
            output.append("| " + " | ".join(row) + " |")
        
        output.append("")
        output.append("## Input Expressions")
        output.append("")
        for col in dmn_table['condition_columns']:
            output.append(f"- **{col['column_name']}**: `{col['input_expression']}`")
        
        return "\n".join(output)
    
    def format_dmn_as_html(self, dmn_table: Dict) -> str:
        """Format DMN table as HTML for display."""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{dmn_table['table_name']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .metadata {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 10px;
            border: 1px solid #ddd;
        }}
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        tr:hover {{
            background: #e8f4f8;
        }}
        .result-col {{
            background: #2ecc71;
            color: white;
            font-weight: bold;
        }}
        .expressions {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }}
        .expressions h3 {{
            margin-top: 0;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{dmn_table['table_name']}</h1>
        
        <div class="metadata">
            <p><strong>Description:</strong> {dmn_table['description']}</p>
            <p><strong>Hit Policy:</strong> {dmn_table['hit_policy']}</p>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Rule #</th>
"""
        
        # Add condition column headers
        for col in dmn_table['condition_columns']:
            html += f"                    <th>{col['column_name']}</th>\n"
        
        # Add result column header
        html += f"                    <th class='result-col'>{dmn_table['result_column_name']}</th>\n"
        html += """                </tr>
            </thead>
            <tbody>
"""
        
        # Add rows
        for rule in dmn_table['rules']:
            html += "                <tr>\n"
            html += f"                    <td><strong>{rule['rule_number']}</strong></td>\n"
            
            for col in dmn_table['condition_columns']:
                col_name = col['column_name']
                value = rule['conditions'].get(col_name, "-")
                html += f"                    <td>{value}</td>\n"
            
            html += f"                    <td class='result-col'>{rule['result']}</td>\n"
            html += "                </tr>\n"
        
        html += """            </tbody>
        </table>
        
        <div class="expressions">
            <h3>Input Expressions</h3>
            <ul>
"""
        
        for col in dmn_table['condition_columns']:
            html += f"                <li><strong>{col['column_name']}:</strong> <code>{col['input_expression']}</code></li>\n"
        
        html += """            </ul>
        </div>
    </div>
</body>
</html>
"""
        
        return html


def main():
    """Demo: Convert Emergency Rule to DMN table."""
    import sys
    
    # Load modernized rule
    rule_file = sys.argv[1] if len(sys.argv) > 1 else 'output/modernized_emergency_rule.json'
    
    print("\n" + "="*70)
    print("LLM Synthesizer - DMN Table Generation")
    print("="*70)
    
    print(f"\n📂 Loading rule from: {rule_file}")
    
    with open(rule_file) as f:
        modernized_rule = json.load(f)
    
    print(f"✓ Loaded: {modernized_rule.get('rule_name', 'Unknown')}")
    
    # Initialize synthesizer
    try:
        synthesizer = LLMSynthesizer("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")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Set your API key:")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    # Generate DMN table
    dmn_table = synthesizer.generate_dmn_table(modernized_rule)
    
    # Save JSON
    json_output = 'output/dmn_emergency_rule.json'
    with open(json_output, 'w') as f:
        json.dump(dmn_table, f, indent=2)
    print(f"\n💾 Saved JSON: {json_output}")
    
    # Save Markdown
    markdown = synthesizer.format_dmn_as_markdown(dmn_table)
    md_output = 'output/dmn_emergency_rule.md'
    with open(md_output, 'w') as f:
        f.write(markdown)
    print(f"💾 Saved Markdown: {md_output}")
    
    # Save HTML
    html = synthesizer.format_dmn_as_html(dmn_table)
    html_output = 'output/dmn_emergency_rule.html'
    with open(html_output, 'w') as f:
        f.write(html)
    print(f"💾 Saved HTML: {html_output}")
    
    # Display
    print("\n" + "="*70)
    print("DMN TABLE PREVIEW")
    print("="*70)
    print(markdown)
    
    print("\n✅ DMN table generation complete!")
    print("\n💡 Next steps:")
    print("  1. Review the DMN table in HTML: open output/dmn_emergency_rule.html")
    print("  2. Verify all rule combinations are covered")
    print("  3. Use JSON for rule engine import")
    print("  4. Generate tables for remaining rules")


if __name__ == '__main__':
    main()