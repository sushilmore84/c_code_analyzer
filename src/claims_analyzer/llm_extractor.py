#!/usr/bin/env python3
"""
Business Rule Extractor using LLM
Demonstrates using Claude API to understand scattered business logic
"""

import os
import json
from pathlib import Path
from anthropic import Anthropic
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BusinessRuleExtractor:
    """Uses LLM to extract business rules from C code"""
    
    def __init__(self, api_key: str = None): # type: ignore
        """Initialize with Anthropic API key"""
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Set it in .env file or environment")
        
        self.client = Anthropic(api_key=self.api_key)
    
    def read_function_code(self, file_path: str, function_name: str) -> str:
        """Read the source code of a specific function"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Simple extraction - find function and get its code
        in_function = False
        function_code = []
        brace_count = 0
        
        for line in lines:
            if function_name in line and '(' in line:
                in_function = True
            
            if in_function:
                function_code.append(line)
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0 and '{' in ''.join(function_code):
                    break
        
        return ''.join(function_code)
    
    def analyze_scattered_rule(self, 
                               functions: List[Dict[str, str]], 
                               context: str = "") -> Dict:
        """
        Use LLM to analyze multiple functions that form a scattered rule
        
        Args:
            functions: List of dicts with 'name', 'file', 'code'
            context: Additional context about the system
        """
        # Build the prompt
        prompt = f"""You are analyzing a legacy C codebase for a healthcare claims processing system.

{context}

I have identified {len(functions)} functions that appear to be related based on data flow analysis.
These functions work together to implement a business rule, but the logic is scattered across multiple files.

"""
        
        for i, func in enumerate(functions, 1):
            prompt += f"""
FUNCTION {i}: {func['name']}
File: {func['file']}

```c
{func['code']}
```
"""
        
        prompt += """

TASK:
Please analyze these functions and:

1. **Identify the Business Rule**: What complete business rule do these functions implement together?

2. **Business Name**: Give this rule a clear business name (e.g., "Emergency Procedure Premium Rule")

3. **Complete Description**: Describe the rule in plain English that a business analyst would understand

4. **Conditions**: What are ALL the conditions that must be met for this rule to apply?

5. **Actions**: What happens when the rule is triggered?

6. **Business Rationale**: Why does this rule exist? What business problem does it solve?

7. **Execution Flow**: How do these functions work together? What's the sequence?

8. **Example Scenario**: Provide a concrete example with actual values

Respond in JSON format:
{
  "rule_name": "...",
  "rule_type": "validation/calculation/audit/assignment",
  "description": "...",
  "conditions": ["condition 1", "condition 2", ...],
  "actions": ["action 1", "action 2", ...],
  "business_rationale": "...",
  "execution_flow": ["step 1", "step 2", ...],
  "example": {
    "scenario": "...",
    "input": "...",
    "output": "..."
  }
}
"""
        
        print(f"🤖 Sending {len(functions)} functions to Claude for analysis...")
        
        # Call Claude API
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract the response
        response_text = response.content[0].text
        
        # Try to parse as JSON
        try:
            # Find JSON in response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
            rule = json.loads(json_str)
            
            # Add metadata
            rule['functions'] = [{'name': f['name'], 'file': f['file']} for f in functions]
            
            return rule
        except json.JSONDecodeError:
            # Return raw response if not JSON
            return {
                'rule_name': 'Parse Error',
                'raw_response': response_text
            }
    
    def analyze_single_function(self, function_code: str, function_name: str) -> str:
        """Analyze a single function to understand its purpose"""
        prompt = f"""Analyze this C function from a claims processing system:

Function: {function_name}

```c
{function_code}
```

Explain in 2-3 sentences:
1. What business purpose does this function serve?
2. What are the key business logic elements?
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text


def demo_emergency_premium_rule():
    """
    Demonstrate extracting the Emergency Premium Rule
    This rule is scattered across 4 files!
    """
    print("=" * 70)
    print("DEMO: Extracting Emergency Procedure Premium Rule")
    print("=" * 70)
    print()
    
    # Initialize extractor
    extractor = BusinessRuleExtractor()
    
    # Define the source directory
    source_dir = Path('../claims_system')
    
    # The Emergency Premium rule involves these functions
    functions = [
        {
            'name': 'validate_emergency_facility',
            'file': 'claim_validation.c',
            'code': extractor.read_function_code(
                source_dir / 'claim_validation.c',  # type: ignore
                'validate_emergency_facility'
            )
        },
        {
            'name': 'verify_emergency_procedure',
            'file': 'claim_validation.c',
            'code': extractor.read_function_code(
                source_dir / 'claim_validation.c', # type: ignore
                'verify_emergency_procedure'
            )
        },
        {
            'name': 'is_provider_in_network',
            'file': 'network_verification.c',
            'code': extractor.read_function_code(
                source_dir / 'network_verification.c', # type: ignore
                'is_provider_in_network'
            )
        },
        {
            'name': 'apply_emergency_premium',
            'file': 'cos_calculation.c',
            'code': extractor.read_function_code(
                source_dir / 'cos_calculation.c', # type: ignore
                'apply_emergency_premium'
            )
        }
    ]
    
    context = """
The system processes healthcare insurance claims. The main entity is a 'claim' which flows through:
1. Validation (Edit module)
2. Audit
3. Cost of Service (COS) calculation
4. Payment

Claims have flags that communicate state between modules. One important flag is FLAG_EMERGENCY.
    """
    
    # Analyze with LLM
    print("Analyzing functions with Claude AI...")
    print()
    
    rule = extractor.analyze_scattered_rule(functions, context)
    
    # Display results
    print("✅ EXTRACTED BUSINESS RULE")
    print("=" * 70)
    print()
    print(f"📋 Rule Name: {rule.get('rule_name', 'N/A')}")
    print(f"📊 Rule Type: {rule.get('rule_type', 'N/A')}")
    print()
    print("📝 Description:")
    print(f"   {rule.get('description', 'N/A')}")
    print()
    
    if 'conditions' in rule:
        print("✓ Conditions:")
        for i, condition in enumerate(rule['conditions'], 1):
            print(f"   {i}. {condition}")
        print()
    
    if 'actions' in rule:
        print("⚡ Actions:")
        for i, action in enumerate(rule['actions'], 1):
            print(f"   {i}. {action}")
        print()
    
    if 'business_rationale' in rule:
        print("💼 Business Rationale:")
        print(f"   {rule['business_rationale']}")
        print()
    
    if 'example' in rule:
        print("📌 Example:")
        example = rule['example']
        print(f"   Scenario: {example.get('scenario', 'N/A')}")
        print(f"   Input: {example.get('input', 'N/A')}")
        print(f"   Output: {example.get('output', 'N/A')}")
        print()
    
    print("📍 Code Locations:")
    for func in rule.get('functions', []):
        print(f"   • {func['file']}: {func['name']}()")
    print()
    
    # Save to file
    output_file = 'emergency_premium_rule.json'
    with open(output_file, 'w') as f:
        json.dump(rule, f, indent=2)
    
    print(f"💾 Full results saved to: {output_file}")
    print()
    print("=" * 70)
    print("✅ This demonstrates how LLM understands scattered business logic!")
    print("=" * 70)


def main():
    """Main function"""
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("⚠️  ERROR: ANTHROPIC_API_KEY not set")
        print()
        print("To use this script, you need a Claude API key.")
        print()
        print("Set it in one of these ways:")
        print("1. Create a .env file with: ANTHROPIC_API_KEY=your-key-here")
        print("2. Export in terminal: export ANTHROPIC_API_KEY=your-key-here")
        print()
        print("Get your API key from: https://console.anthropic.com/")
        return
    
    # Run the demo
    demo_emergency_premium_rule()


if __name__ == '__main__':
    main()
