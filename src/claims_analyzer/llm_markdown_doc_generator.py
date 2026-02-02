#!/usr/bin/env python3
"""
LLM-Powered Business Rule Documentation Generator - Markdown Version
Pure Python - No npm/Node.js/docx dependencies required!
Generates professional Markdown documentation using Claude API.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class RuleDocumentation:
    """Complete documentation package for a business rule."""
    rule_id: str
    rule_name: str
    
    # LLM-generated business perspective
    executive_summary: str
    business_description: str
    business_value: str
    business_rationale: str
    business_examples: List[Dict]
    
    # LLM-generated detailed specification
    conditions_plain_english: List[str]
    actions_plain_english: List[str]
    decision_logic: str
    execution_flow: List[str]
    
    # LLM-generated context and relationships
    related_rules: List[str]
    upstream_dependencies: List[str]
    downstream_impacts: List[str]
    potential_risks: List[str]
    compliance_considerations: List[str]
    
    # Metadata
    rule_category: str
    business_domains: List[str]
    execution_phases: List[str]
    criticality: str
    
    # Technical reference
    technical_summary: Dict
    implementation_locations: List[str]
    
    # Version info
    last_modified: str
    version: str
    review_status: str


class MarkdownDocGenerator:
    """Generate comprehensive Markdown documentation using Claude API."""
    
    def __init__(self, modernized_rules_file: str, graph_analysis_file: str):
        """Initialize with modernized rules and graph analysis."""
        
        # Check for API key
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Please set it in .env file or environment.\n"
                "Get your API key from: https://console.anthropic.com/"
            )
        
        self.client = Anthropic(api_key=self.api_key)
        
        # Load modernized rules
        with open(modernized_rules_file) as f:
            data = json.load(f)
            # Handle both single rule and array of rules
            if isinstance(data, dict):
                self.modernized_rules = [data]
            else:
                self.modernized_rules = data
        
        # Load graph analysis for additional context
        with open(graph_analysis_file) as f:
            self.graph_data = json.load(f)
    
    def generate_executive_summary(self, rule: Dict) -> str:
        """Use LLM to generate executive summary."""
        
        prompt = f"""You are a business analyst creating executive documentation for a healthcare claims processing system.

Generate a 2-3 sentence executive summary for the following business rule. The summary should be:
- Written for C-suite executives (CEO, CFO, COO)
- Focus on business impact and value
- Avoid technical jargon
- Mention financial, operational, or compliance impact

RULE INFORMATION:
Rule Name: {rule.get('rule_name', 'Unknown')}
Category: {rule.get('rule_category', 'processing')}
Criticality: {rule.get('rule_criticality', 'medium')}
Business Domains: {', '.join(rule.get('business_domain', []))}

Business Description: {rule.get('business_description', '')}

Conditions: {json.dumps(rule.get('conditions_if', []), indent=2)}
Actions: {json.dumps(rule.get('actions_then', []), indent=2)}

Generate ONLY the executive summary text (2-3 sentences). Do not include any preamble or explanation.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
    
    def generate_business_value(self, rule: Dict) -> str:
        """Use LLM to generate business value analysis."""
        
        prompt = f"""You are a business analyst evaluating the business value of a healthcare claims processing rule.

Analyze the business value for this rule and provide:
1. **Business Value** - Why this rule is important (3-5 bullet points)
2. **Estimated Impact** - Quantifiable or qualitative impact (2-3 bullet points)

Use realistic healthcare industry metrics where appropriate.

RULE INFORMATION:
Rule Name: {rule.get('rule_name', 'Unknown')}
Category: {rule.get('rule_category', 'processing')}
Business Description: {rule.get('business_description', '')}

Impacts: {json.dumps(rule.get('impacts', []), indent=2)}
Used By: {json.dumps(rule.get('used_by', []), indent=2)}

Format your response as markdown with clear headers:
**Business Value:**
- Point 1
- Point 2
...

**Estimated Impact:**
- Impact 1
- Impact 2
...

Be specific and realistic. For financial estimates, use placeholder format like "$X million" or "XX%" if exact numbers aren't known.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
    
    def generate_business_rationale(self, rule: Dict) -> str:
        """Use LLM to explain why this rule exists."""
        
        prompt = f"""You are a business analyst explaining why business rules exist in a healthcare claims processing system.

Explain the business rationale for this rule. Answer these questions:
1. Why does this rule exist?
2. What business problem does it solve?
3. What would happen without this rule?

RULE INFORMATION:
Rule Name: {rule.get('rule_name', 'Unknown')}
Business Description: {rule.get('business_description', '')}
Conditions: {json.dumps(rule.get('conditions_if', []), indent=2)}
Actions: {json.dumps(rule.get('actions_then', []), indent=2)}

Provide a clear, concise explanation (2-3 paragraphs) that a non-technical business stakeholder would understand.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
    
    def generate_plain_english_conditions(self, rule: Dict) -> List[str]:
        """Use LLM to convert conditions to plain English."""
        
        conditions = rule.get('conditions_if', [])
        if not conditions:
            return ["This rule applies to all eligible claims."]
        
        # Extract condition text
        condition_texts = []
        for cond in conditions:
            if isinstance(cond, dict):
                condition_texts.append(cond.get('condition', str(cond)))
            else:
                condition_texts.append(str(cond))
        
        prompt = f"""You are translating technical rule conditions into plain English for business stakeholders.

Convert these technical conditions into clear, simple English sentences that anyone can understand.

TECHNICAL CONDITIONS:
{json.dumps(condition_texts, indent=2)}

CONTEXT:
- This is a healthcare claims processing rule
- Conditions determine when the rule applies
- Use simple business language, avoid technical terms

Provide a JSON array of plain English conditions. Each should be a complete sentence starting with "When" or "If".

Example format:
[
  "When the claim is for an emergency room visit",
  "When the procedure is classified as emergency care",
  "When the provider is in the network"
]

Return ONLY the JSON array, nothing else.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON response
        try:
            result_text = response.content[0].text.strip()
            start = result_text.find('[')
            end = result_text.rfind(']') + 1
            if start >= 0 and end > start:
                json_text = result_text[start:end]
                return json.loads(json_text)
            else:
                return condition_texts
        except json.JSONDecodeError:
            return condition_texts
    
    def generate_plain_english_actions(self, rule: Dict) -> List[str]:
        """Use LLM to convert actions to plain English."""
        
        actions = rule.get('actions_then', [])
        if not actions:
            return ["The system processes the claim according to standard procedures."]
        
        prompt = f"""You are translating technical rule actions into plain English for business stakeholders.

Convert these technical actions into clear, simple English sentences that describe what the system does.

TECHNICAL ACTIONS:
{json.dumps(actions, indent=2)}

CONTEXT:
- This is a healthcare claims processing rule
- Actions describe what happens when the rule is triggered
- Use active voice and simple business language

Provide a JSON array of plain English actions. Each should be a complete sentence starting with "The system will" or similar.

Example format:
[
  "The system will mark the claim as emergency",
  "The system will apply a premium rate adjustment",
  "The system will calculate the final reimbursement amount"
]

Return ONLY the JSON array, nothing else.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON response
        try:
            result_text = response.content[0].text.strip()
            start = result_text.find('[')
            end = result_text.rfind(']') + 1
            if start >= 0 and end > start:
                json_text = result_text[start:end]
                return json.loads(json_text)
            else:
                return actions
        except json.JSONDecodeError:
            return actions
    
    def generate_decision_logic_narrative(self, rule: Dict, 
                                         conditions: List[str], 
                                         actions: List[str]) -> str:
        """Use LLM to generate decision logic narrative."""
        
        prompt = f"""You are creating a narrative explanation of business rule logic for non-technical stakeholders.

Create a clear, flowing narrative that explains how this rule works. The narrative should:
- Explain the decision flow from conditions to actions
- Use simple language
- Be easy to follow

RULE NAME: {rule.get('rule_name', 'Unknown')}

CONDITIONS (IF):
{json.dumps(conditions, indent=2)}

ACTIONS (THEN):
{json.dumps(actions, indent=2)}

Write a narrative explanation (2-3 paragraphs) using markdown formatting with headers:
**Decision Flow:**
[Explanation of when the rule triggers]

**Resulting Actions:**
[Explanation of what happens]

Keep it concise and business-focused.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
    
    def generate_execution_flow(self, rule: Dict) -> List[str]:
        """Use LLM to generate step-by-step execution flow."""
        
        prompt = f"""You are creating a step-by-step execution flow for a business rule.

Create a numbered sequence of steps that shows exactly how this rule executes, from trigger to final action.

RULE INFORMATION:
Rule Name: {rule.get('rule_name', 'Unknown')}
Triggers: {json.dumps(rule.get('triggers', []), indent=2)}
Conditions: {json.dumps(rule.get('conditions_if', []), indent=2)}
Actions: {json.dumps(rule.get('actions_then', []), indent=2)}
Used By: {json.dumps(rule.get('used_by', []), indent=2)}

Create a JSON array of execution steps. Each step should be a simple string describing what happens.

Example:
[
  "Claim enters the validation pipeline",
  "System checks if facility is Emergency Room",
  "Claim is marked as emergency",
  "Premium pricing module applies emergency rates",
  "Final reimbursement amount is calculated"
]

Typically 4-6 steps. Return ONLY the JSON array.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON response
        try:
            result_text = response.content[0].text.strip()
            start = result_text.find('[')
            end = result_text.rfind(']') + 1
            if start >= 0 and end > start:
                json_text = result_text[start:end]
                return json.loads(json_text)
            else:
                return [
                    "Rule is triggered by claim processing",
                    "Conditions are evaluated",
                    "Appropriate actions are taken"
                ]
        except json.JSONDecodeError:
            return [
                "Rule is triggered by claim processing",
                "Conditions are evaluated",
                "Appropriate actions are taken"
            ]
    
    def generate_business_examples(self, rule: Dict) -> List[Dict]:
        """Use LLM to generate realistic business examples."""
        
        prompt = f"""You are creating realistic business examples for a healthcare claims processing rule.

Generate 2-3 concrete business scenarios that demonstrate how this rule works in practice.

RULE INFORMATION:
Rule Name: {rule.get('rule_name', 'Unknown')}
Business Description: {rule.get('business_description', '')}
Conditions: {json.dumps(rule.get('conditions_if', []), indent=2)}
Actions: {json.dumps(rule.get('actions_then', []), indent=2)}

For each example, provide:
1. **scenario** - A brief title (e.g., "Emergency Room Visit for Chest Pain")
2. **input** - A dictionary of input values with realistic data
3. **rule_evaluation** - How the rule evaluates the input
4. **outcome** - A dictionary showing the results
5. **business_rationale** - Why this outcome makes business sense

Generate examples that show:
- A typical case where the rule applies
- An edge case or exception
- (Optional) A case where the rule does NOT apply

Return a JSON array of example objects. Use realistic healthcare values, patient names (anonymized like "John Doe"), facility names, dollar amounts, etc.

Example format:
[
  {{
    "scenario": "Emergency Room Visit",
    "input": {{
      "Patient": "John Doe",
      "Facility": "City Hospital ER",
      "Procedure": "Cardiac evaluation",
      "Base Cost": "$2,500"
    }},
    "rule_evaluation": "Facility type = ER AND Procedure = Emergency",
    "outcome": {{
      "Rule Applied": "Yes",
      "Premium": "1.5x",
      "Final Amount": "$3,750"
    }},
    "business_rationale": "Emergency services require premium rates..."
  }}
]

Return ONLY the JSON array with 2-3 examples.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON response
        try:
            result_text = response.content[0].text.strip()
            start = result_text.find('[')
            end = result_text.rfind(']') + 1
            if start >= 0 and end > start:
                json_text = result_text[start:end]
                return json.loads(json_text)
            else:
                return [{
                    'scenario': 'Standard Claim Processing',
                    'input': {'claim_data': 'Various attributes'},
                    'rule_evaluation': 'Rule conditions evaluated',
                    'outcome': {'result': 'Rule applied'},
                    'business_rationale': 'Ensures consistent processing'
                }]
        except json.JSONDecodeError:
            return [{
                'scenario': 'Standard Claim Processing',
                'input': {'claim_data': 'Various attributes'},
                'rule_evaluation': 'Rule conditions evaluated',
                'outcome': {'result': 'Rule applied'},
                'business_rationale': 'Ensures consistent processing'
            }]
    
    def identify_related_rules(self, rule: Dict) -> List[str]:
        """Use LLM to identify related rules."""
        
        prompt = f"""You are analyzing relationships between business rules in a healthcare claims system.

Identify rules that are likely related to this rule. Consider:
- Rules that share data fields
- Rules that execute in sequence
- Rules that have similar business purposes
- Rules that handle exceptions to this rule

CURRENT RULE:
Name: {rule.get('rule_name', 'Unknown')}
Category: {rule.get('rule_category', '')}
Data Inputs: {json.dumps(rule.get('data_inputs', []), indent=2)}
Data Outputs: {json.dumps(rule.get('data_outputs', []), indent=2)}
Used By: {json.dumps(rule.get('used_by', []), indent=2)}

Based on typical healthcare claims processing, what other rules would likely interact with this one?

Provide a JSON array of related rule descriptions. Each entry should explain the relationship.

Example:
[
  "Network Verification Rule (checks provider network status before applying rates)",
  "Prior Authorization Rule (may require authorization for high-cost procedures)",
  "Fraud Detection Rule (reviews claims flagged by this rule for fraud indicators)"
]

Return ONLY the JSON array with 3-5 related rules.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON response
        try:
            result_text = response.content[0].text.strip()
            start = result_text.find('[')
            end = result_text.rfind(']') + 1
            if start >= 0 and end > start:
                json_text = result_text[start:end]
                return json.loads(json_text)
            else:
                return []
        except json.JSONDecodeError:
            return []
    
    def identify_risks_and_compliance(self, rule: Dict) -> tuple[List[str], List[str]]:
        """Use LLM to identify potential risks and compliance considerations."""
        
        prompt = f"""You are a compliance and risk analyst reviewing a healthcare claims processing rule.

Identify:
1. **Potential Risks** - What could go wrong if this rule malfunctions or is incorrectly configured?
2. **Compliance Considerations** - What regulations or standards does this rule help meet?

RULE INFORMATION:
Name: {rule.get('rule_name', 'Unknown')}
Category: {rule.get('rule_category', '')}
Criticality: {rule.get('rule_criticality', '')}
Business Description: {rule.get('business_description', '')}
Impacts: {json.dumps(rule.get('impacts', []), indent=2)}

Provide your response as JSON:
{{
  "potential_risks": [
    "Risk 1 description",
    "Risk 2 description"
  ],
  "compliance_considerations": [
    "Compliance item 1",
    "Compliance item 2"
  ]
}}

Focus on realistic healthcare industry risks and compliance requirements (HIPAA, state regulations, etc.).
Return ONLY the JSON object.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON response
        try:
            result_text = response.content[0].text.strip()
            start = result_text.find('{')
            end = result_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_text = result_text[start:end]
                data = json.loads(json_text)
                return data.get('potential_risks', []), data.get('compliance_considerations', [])
            else:
                return [], []
        except json.JSONDecodeError:
            return [], []
    
    def generate_full_documentation(self, rule: Dict) -> RuleDocumentation:
        """Generate complete documentation for a single rule using LLM."""
        
        print(f"\n🤖 Generating documentation for: {rule.get('rule_name', 'Unknown')}")
        print("   Using Claude API for all content generation...")
        
        # Generate all components using LLM
        print("   📝 Executive summary...")
        exec_summary = self.generate_executive_summary(rule)
        
        print("   💰 Business value...")
        business_value = self.generate_business_value(rule)
        
        print("   🎯 Business rationale...")
        business_rationale = self.generate_business_rationale(rule)
        
        print("   ✅ Plain English conditions...")
        plain_conditions = self.generate_plain_english_conditions(rule)
        
        print("   ⚡ Plain English actions...")
        plain_actions = self.generate_plain_english_actions(rule)
        
        print("   📊 Decision logic narrative...")
        decision_logic = self.generate_decision_logic_narrative(rule, plain_conditions, plain_actions)
        
        print("   🔄 Execution flow...")
        execution_flow = self.generate_execution_flow(rule)
        
        print("   📋 Business examples...")
        examples = self.generate_business_examples(rule)
        
        print("   🔗 Related rules...")
        related_rules = self.identify_related_rules(rule)
        
        print("   ⚠️  Risks and compliance...")
        risks, compliance = self.identify_risks_and_compliance(rule)
        
        # Get dependencies from existing data
        upstream = rule.get('triggers', [])
        if rule.get('data_inputs'):
            upstream.append(f"Data inputs: {', '.join(rule.get('data_inputs', [])[:3])}")
        
        downstream = rule.get('used_by', [])
        downstream.extend(rule.get('impacts', []))
        
        # Technical summary
        technical_summary = {
            'flag_name': rule.get('flag_name', ''),
            'setter_functions': len(rule.get('setter_functions', [])),
            'checker_functions': len(rule.get('checker_functions', [])),
            'implementation_files': rule.get('implementation_files', [])
        }
        
        print("   ✅ Documentation complete!")
        
        return RuleDocumentation(
            rule_id=rule.get('rule_id', ''),
            rule_name=rule.get('rule_name', ''),
            executive_summary=exec_summary,
            business_description=rule.get('business_description', ''),
            business_value=business_value,
            business_rationale=business_rationale,
            business_examples=examples,
            conditions_plain_english=plain_conditions,
            actions_plain_english=plain_actions,
            decision_logic=decision_logic,
            execution_flow=execution_flow,
            related_rules=related_rules,
            upstream_dependencies=upstream,
            downstream_impacts=downstream,
            potential_risks=risks,
            compliance_considerations=compliance,
            rule_category=rule.get('rule_category', ''),
            business_domains=rule.get('business_domain', []),
            execution_phases=rule.get('execution_phase', []),
            criticality=rule.get('rule_criticality', ''),
            technical_summary=technical_summary,
            implementation_locations=rule.get('implementation_files', []),
            last_modified=datetime.now().strftime('%Y-%m-%d'),
            version='1.0',
            review_status='AI-Generated - Pending Review'
        )
    
    def create_markdown_document(self, doc: RuleDocumentation, output_file: str):
        """Create a professional Markdown document for the rule."""
        
        md = []
        
        # Title and metadata
        md.append(f"# {doc.rule_name}")
        md.append("")
        md.append("**Business Rule Documentation**")
        md.append("")
        md.append(f"**Rule ID:** {doc.rule_id}")
        md.append("")
        md.append("---")
        md.append("")
        
        # Metadata table
        md.append("## Metadata")
        md.append("")
        md.append("| Field | Value |")
        md.append("|-------|-------|")
        md.append(f"| **Category** | {doc.rule_category.upper()} |")
        md.append(f"| **Criticality** | {doc.criticality.upper()} |")
        md.append(f"| **Business Domains** | {', '.join(doc.business_domains)} |")
        md.append(f"| **Execution Phase** | {', '.join(doc.execution_phases)} |")
        md.append(f"| **Version** | {doc.version} |")
        md.append(f"| **Last Modified** | {doc.last_modified} |")
        md.append(f"| **Review Status** | {doc.review_status} |")
        md.append("")
        md.append("---")
        md.append("")
        
        # Executive Summary
        md.append("## Executive Summary")
        md.append("")
        md.append(doc.executive_summary)
        md.append("")
        md.append("---")
        md.append("")
        
        # Business Description
        md.append("## Business Description")
        md.append("")
        md.append(doc.business_description)
        md.append("")
        md.append("---")
        md.append("")
        
        # Business Value
        md.append("## Business Value")
        md.append("")
        md.append(doc.business_value)
        md.append("")
        md.append("---")
        md.append("")
        
        # Business Rationale
        md.append("## Business Rationale")
        md.append("")
        md.append(doc.business_rationale)
        md.append("")
        md.append("---")
        md.append("")
        
        # Rule Logic
        md.append("## Rule Logic")
        md.append("")
        
        md.append("### Conditions (IF)")
        md.append("")
        for i, condition in enumerate(doc.conditions_plain_english, 1):
            md.append(f"{i}. {condition}")
        md.append("")
        
        md.append("### Actions (THEN)")
        md.append("")
        for i, action in enumerate(doc.actions_plain_english, 1):
            md.append(f"{i}. {action}")
        md.append("")
        
        md.append("### Decision Logic")
        md.append("")
        md.append(doc.decision_logic)
        md.append("")
        md.append("---")
        md.append("")
        
        # Execution Flow
        md.append("## Execution Flow")
        md.append("")
        for i, step in enumerate(doc.execution_flow, 1):
            md.append(f"{i}. {step}")
        md.append("")
        md.append("---")
        md.append("")
        
        # Business Examples
        md.append("## Business Examples")
        md.append("")
        
        for i, example in enumerate(doc.business_examples, 1):
            md.append(f"### Example {i}: {example.get('scenario', 'Scenario')}")
            md.append("")
            
            # Input
            if 'input' in example and example['input']:
                md.append("#### Input")
                md.append("")
                for key, value in example['input'].items():
                    md.append(f"- **{key}:** {value}")
                md.append("")
            
            # Rule Evaluation
            if 'rule_evaluation' in example:
                md.append("#### Rule Evaluation")
                md.append("")
                md.append(example['rule_evaluation'])
                md.append("")
            
            # Outcome
            if 'outcome' in example and example['outcome']:
                md.append("#### Outcome")
                md.append("")
                for key, value in example['outcome'].items():
                    md.append(f"- **{key}:** {value}")
                md.append("")
            
            # Business Rationale
            if 'business_rationale' in example:
                md.append("#### Business Rationale")
                md.append("")
                md.append(example['business_rationale'])
                md.append("")
        
        md.append("---")
        md.append("")
        
        # Related Rules
        if doc.related_rules:
            md.append("## Related Rules")
            md.append("")
            for related in doc.related_rules:
                md.append(f"- {related}")
            md.append("")
            md.append("---")
            md.append("")
        
        # Dependencies
        md.append("## Dependencies")
        md.append("")
        
        if doc.upstream_dependencies:
            md.append("### Upstream Dependencies")
            md.append("")
            md.append("*What must happen before this rule:*")
            md.append("")
            for dep in doc.upstream_dependencies:
                md.append(f"- {dep}")
            md.append("")
        
        if doc.downstream_impacts:
            md.append("### Downstream Impacts")
            md.append("")
            md.append("*What happens after this rule:*")
            md.append("")
            for impact in doc.downstream_impacts:
                md.append(f"- {impact}")
            md.append("")
        
        md.append("---")
        md.append("")
        
        # Risk and Compliance
        if doc.potential_risks or doc.compliance_considerations:
            md.append("## Risk and Compliance")
            md.append("")
            
            if doc.potential_risks:
                md.append("### Potential Risks")
                md.append("")
                for risk in doc.potential_risks:
                    md.append(f"- {risk}")
                md.append("")
            
            if doc.compliance_considerations:
                md.append("### Compliance Considerations")
                md.append("")
                for compliance in doc.compliance_considerations:
                    md.append(f"- {compliance}")
                md.append("")
            
            md.append("---")
            md.append("")
        
        # Technical Reference
        md.append("## Technical Reference")
        md.append("")
        md.append("*For IT Teams*")
        md.append("")
        md.append(f"- **Flag Name:** {doc.technical_summary.get('flag_name', 'N/A')}")
        md.append(f"- **Setter Functions:** {doc.technical_summary.get('setter_functions', 0)}")
        md.append(f"- **Checker Functions:** {doc.technical_summary.get('checker_functions', 0)}")
        md.append("")
        
        if doc.implementation_locations:
            md.append("### Implementation Locations")
            md.append("")
            for location in doc.implementation_locations:
                md.append(f"- `{location}`")
            md.append("")
        
        md.append("---")
        md.append("")
        
        # Footer
        md.append("*Documentation generated by LLM-Powered Business Rule Extraction System*")
        md.append("")
        md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md))
        
        print(f"   ✅ Created: {output_file}")
    
    def generate_all_documentation(self, output_dir: str = 'output/docs'):
        """Generate documentation for all rules using LLM."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("LLM-Powered Business Rule Documentation Generation")
        print("Pure Python - Markdown Output")
        print("="*70)
        
        all_docs = []
        
        for rule in self.modernized_rules:
            doc = self.generate_full_documentation(rule)
            all_docs.append(doc)
            
            # Create Markdown document
            output_file = f"{output_dir}/{doc.rule_id}_{doc.rule_name.replace(' ', '_')}.md"
            self.create_markdown_document(doc, output_file)
            
            # Save JSON version
            json_file = output_file.replace('.md', '.json')
            with open(json_file, 'w') as f:
                json.dump(asdict(doc), f, indent=2)
            print(f"   ✅ JSON: {json_file}")
        
        print(f"\n✅ Generated documentation for {len(all_docs)} rules")
        print(f"📁 Output directory: {output_dir}")
        print(f"\n📄 Markdown files are ready to:")
        print(f"   - View in any text editor")
        print(f"   - Convert to HTML/PDF with pandoc")
        print(f"   - Commit to version control (Git)")
        print(f"   - View on GitHub/GitLab with formatting")
        
        return all_docs


def main():
    """Main function."""
    import sys
    
    # Get input files
    modernized_file = sys.argv[1] if len(sys.argv) > 1 else 'output/modernized_emergency_rule.json'
    graph_file = sys.argv[2] if len(sys.argv) > 2 else 'output/graph_analysis_for_llm.json'
    
    print("\n" + "="*70)
    print("LLM-Powered Business Rule Documentation Generator")
    print("Pure Python - Markdown Output (No npm required!)")
    print("="*70)
    
    print(f"\n📂 Loading modernized rules from: {modernized_file}")
    print(f"📂 Loading graph analysis from: {graph_file}")
    
    # Create generator
    try:
        generator = MarkdownDocGenerator(modernized_file, graph_file)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Generate all documentation
    docs = generator.generate_all_documentation()
    
    print("\n" + "="*70)
    print("✅ LLM Documentation Generation Complete!")
    print("="*70)
    print("\n💡 All content was generated by AI:")
    print("  ✅ Executive summaries")
    print("  ✅ Business value statements")
    print("  ✅ Plain English conditions/actions")
    print("  ✅ Business examples with realistic data")
    print("  ✅ Related rules")
    print("  ✅ Risk and compliance analysis")
    print("\n📝 Output format: Professional Markdown documents")
    print("   No npm, Node.js, or docx dependencies needed!")


if __name__ == '__main__':
    main()
