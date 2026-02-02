#!/usr/bin/env python3
"""
Rule Modernizer - Convert technical analysis to business-readable rules
Proof of Concept: Emergency Rule
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import re


@dataclass
class ConditionWithSource:
    """Condition with its source function."""
    condition: str
    source_function: str
    relationship: str  # "OR", "AND", "INDEPENDENT"


@dataclass
class ModernizedRule:
    """Business-readable rule structure."""
    rule_id: str
    rule_name: str
    business_description: str
    
    # Core logic
    conditions_if: List[ConditionWithSource]  # Structured conditions with sources
    actions_then: List[str]                   # THEN actions
    
    # Context
    used_by: List[str]        # Which functions/processes use this
    triggers: List[str]       # What triggers this rule
    impacts: List[str]        # What this rule affects
    
    # Technical mapping
    flag_name: str
    setter_functions: List[Dict]
    checker_functions: List[Dict]
    data_inputs: List[str]
    data_outputs: List[str]
    
    # Code locations
    implementation_files: List[str]
    
    # Business metadata
    rule_category: str              # "validation", "calculation", "classification", "audit"
    business_domain: List[str]      # Can span multiple domains
    execution_phase: List[str]      # "Edit", "Audit", "Release", etc.
    rule_criticality: str           # "high", "medium", "low"
    rule_type: str                  # "flag-driven", "threshold", "lookup", etc.


class RuleModernizer:
    """Convert technical analysis to business-readable rules."""
    
    def __init__(self, graph_analysis_file: str):
        """Load graph analysis."""
        with open(graph_analysis_file) as f:
            self.data = json.load(f)
        
        self.business_rules = self.data.get('business_rules', [])
        self.flag_traces = self.data.get('flag_traces', {})
    
    def clean_condition(self, condition: str) -> str:
        """Convert technical condition to business-readable format."""
        if not condition or condition == 'unconditional':
            return None
        
        # Remove C syntax noise
        cleaned = condition.replace('c ->', 'claim.')
        cleaned = cleaned.replace('provider ->', 'provider.')
        cleaned = cleaned.replace('patient ->', 'patient.')
        
        # Remove function call parentheses with arguments
        cleaned = re.sub(r'\(\s*c\s*\)', '', cleaned)
        cleaned = re.sub(r'\(\s*claim\s*\)', '', cleaned)
        
        # Convert operators
        cleaned = cleaned.replace('==', 'equals')
        cleaned = cleaned.replace('!=', 'not equals')
        cleaned = cleaned.replace('&&', 'AND')
        cleaned = cleaned.replace('||', 'OR')
        
        # Handle NOT operator more carefully
        cleaned = cleaned.replace('!', 'NOT ')
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Make specific constants more readable
        cleaned = cleaned.replace('FACILITY_ER', 'Emergency Room')
        cleaned = cleaned.replace('NETWORK_IN', 'In-Network')
        cleaned = cleaned.replace('NETWORK_OUT', 'Out-of-Network')
        cleaned = cleaned.replace('STATUS_PAID', 'Paid')
        
        # Remove extra dots
        cleaned = cleaned.replace('. ', '.')
        
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned
    
    def extract_business_conditions(self, rule: Dict) -> List[ConditionWithSource]:
        """Extract IF conditions with their sources and relationships."""
        conditions = []
        
        flag_name = rule.get('flags_involved', [None])[0]
        if not flag_name or flag_name not in self.flag_traces:
            return conditions
        
        trace = self.flag_traces[flag_name]
        
        # Get conditions from setter functions
        # Each setter is an independent trigger (OR relationship)
        for setter in trace.get('setters', []):
            cond_text = setter.get('condition', '')
            func_name = setter.get('function', '')
            
            if cond_text and cond_text != 'unconditional':
                cleaned = self.clean_condition(cond_text)
                if cleaned:
                    conditions.append(ConditionWithSource(
                        condition=cleaned,
                        source_function=func_name,
                        relationship="OR"  # Multiple setters = OR relationship
                    ))
        
        return conditions
    
    def extract_business_actions(self, rule: Dict) -> List[str]:
        """Extract THEN actions in business language."""
        actions = []
        
        # First, add flag setting action from setters
        flag_name = rule.get('flags_involved', [None])[0]
        if flag_name:
            flag_readable = flag_name.replace('FLAG_', '').replace('_', ' ').lower()
            actions.append(f"Mark claim as {flag_readable}")
        
        # Then get specific actions from checker functions
        if flag_name and flag_name in self.flag_traces:
            trace = self.flag_traces[flag_name]
            
            for checker in trace.get('checkers', []):
                func_name = checker.get('function', '')
                
                # Extract real actions - be conservative, don't invent constants
                if 'apply' in func_name and 'premium' in func_name:
                    # Don't hardcode "1.5x" unless verified in source
                    actions.append("Apply emergency premium adjustment to base rate")
                elif 'apply' in func_name and 'adjustment' in func_name:
                    actions.append("Apply network rate adjustment")
                elif 'calculate' in func_name and 'cos' in func_name:
                    actions.append("Calculate final reimbursement amount")
                elif 'check' in func_name and 'high_value' in func_name:
                    actions.append("Flag for high-value review")
                elif 'check' in func_name and 'duplicate' in func_name:
                    actions.append("Reject as duplicate claim")
                elif 'assess' in func_name and 'fraud' in func_name:
                    actions.append("Flag for fraud investigation")
                elif 'verify' in func_name and 'auth' in func_name:
                    actions.append("Verify prior authorization")
                elif 'review' in func_name and 'necessity' in func_name:
                    actions.append("Trigger medical necessity review")
        
        # Also check raw actions from graph
        raw_actions = rule.get('actions', [])
        for action in raw_actions:
            if action.startswith('Update'):
                # Extract field name
                field = action.replace('Update ', '')
                if 'amount' in field.lower():
                    actions.append("Update claim final amount")
                elif 'status' in field.lower():
                    actions.append("Update claim status")
                elif 'flags' not in field.lower():  # Skip flags (already added)
                    actions.append(action)
            elif action.startswith('Calculate'):
                # Keep calculation actions but be conservative
                if 'multiplier' not in action and '*' not in action:
                    if action not in actions:
                        actions.append(action)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for action in actions:
            if action.lower() not in seen:
                seen.add(action.lower())
                unique_actions.append(action)
        
        return unique_actions
    
    def extract_used_by(self, rule: Dict) -> List[str]:
        """Extract which functions/processes use this rule."""
        used_by = []
        
        flag_name = rule.get('flags_involved', [None])[0]
        if flag_name and flag_name in self.flag_traces:
            trace = self.flag_traces[flag_name]
            
            # Get checker functions (they use this rule)
            for checker in trace.get('checkers', []):
                func_name = checker.get('function', '')
                
                # Convert function name to business description
                business_name = func_name.replace('_', ' ').title()
                
                # Add context
                if 'apply' in func_name.lower():
                    business_name = f"Pricing: {business_name}"
                elif 'verify' in func_name.lower() or 'validate' in func_name.lower():
                    business_name = f"Validation: {business_name}"
                elif 'check' in func_name.lower():
                    business_name = f"Verification: {business_name}"
                elif 'calculate' in func_name.lower():
                    business_name = f"Calculation: {business_name}"
                
                used_by.append(business_name)
        
        return used_by
    
    def extract_triggers(self, rule: Dict) -> List[str]:
        """Extract what triggers this rule."""
        triggers = []
        
        flag_name = rule.get('flags_involved', [None])[0]
        if flag_name and flag_name in self.flag_traces:
            trace = self.flag_traces[flag_name]
            
            # Get setter functions (they trigger this rule)
            for setter in trace.get('setters', []):
                func_name = setter.get('function', '')
                business_name = func_name.replace('_', ' ').title()
                triggers.append(business_name)
        
        return triggers
    
    def extract_impacts(self, rule: Dict) -> List[str]:
        """Extract what this rule impacts."""
        impacts = []
        
        # Look at data fields that are written
        data_fields = rule.get('data_fields_used', [])
        
        for field in data_fields:
            if 'amount' in field.lower() or 'cos' in field.lower():
                impacts.append("Claim reimbursement amount")
            elif 'status' in field.lower():
                impacts.append("Claim processing status")
            elif 'flag' in field.lower():
                impacts.append("Claim state/classification")
            elif 'responsibility' in field.lower():
                impacts.append("Patient financial responsibility")
        
        # Deduplicate
        impacts = list(set(impacts))
        
        return impacts
    
    def categorize_rule(self, rule: Dict) -> str:
        """Determine rule category."""
        rule_name = rule.get('rule_name', '').lower()
        flag_name = rule.get('flags_involved', [None])[0] or ''
        
        # Rules that SET flags are typically classification rules
        if flag_name and flag_name in self.flag_traces:
            trace = self.flag_traces[flag_name]
            has_setters = len(trace.get('setters', [])) > 0
            has_checkers = len(trace.get('checkers', [])) > 0
            
            # If it sets a flag and others use it, it's classification
            if has_setters and has_checkers:
                # Emergency and high-value are classifications that affect pricing
                if 'emergency' in flag_name.lower() or 'high_value' in flag_name.lower():
                    return "classification"
                # Fraud/duplicate are audit classifications
                elif 'fraud' in flag_name.lower() or 'duplicate' in flag_name.lower():
                    return "audit"
                # Auth is validation
                elif 'auth' in flag_name.lower():
                    return "validation"
        
        # Fallback to name-based
        if any(word in rule_name for word in ['fraud', 'duplicate', 'audit', 'assess']):
            return "audit"
        elif any(word in rule_name for word in ['validate', 'verify', 'check', 'auth']):
            return "validation"
        elif any(word in rule_name for word in ['calculate', 'premium', 'adjustment', 'cos']):
            return "calculation"
        else:
            return "classification"
    
    def determine_domains(self, rule: Dict) -> List[str]:
        """Determine business domains (can be multiple)."""
        domains = []
        rule_name = rule.get('rule_name', '').lower()
        flag_name = rule.get('flags_involved', [None])[0] or ''
        
        # Check which processes use this rule
        if flag_name and flag_name in self.flag_traces:
            trace = self.flag_traces[flag_name]
            checker_functions = [c.get('function', '') for c in trace.get('checkers', [])]
            
            # Determine domains from checker functions
            for func in checker_functions:
                if 'validate' in func or 'verify' in func:
                    if 'claim_validation' not in domains:
                        domains.append('claim_validation')
                if 'calculate' in func or 'apply' in func and 'premium' in func:
                    if 'pricing' not in domains:
                        domains.append('pricing')
                if 'fraud' in func or 'duplicate' in func or 'audit' in func:
                    if 'fraud_detection' not in domains:
                        domains.append('fraud_detection')
                if 'network' in func or 'provider' in func:
                    if 'network' not in domains:
                        domains.append('network')
        
        # Fallback to rule name analysis
        if not domains:
            if 'fraud' in rule_name or 'duplicate' in rule_name:
                domains.append('fraud_detection')
            if 'emergency' in rule_name or 'premium' in rule_name or 'cos' in rule_name:
                if 'claim_validation' not in domains:
                    domains.append('claim_validation')
                if 'pricing' not in domains:
                    domains.append('pricing')
            if 'network' in rule_name or 'provider' in rule_name:
                domains.append('network')
            if 'auth' in rule_name:
                domains.append('authorization')
        
        return domains if domains else ['claims_processing']
    
    def determine_execution_phase(self, rule: Dict) -> List[str]:
        """Determine which processing phases this rule executes in."""
        phases = []
        
        # Get setter and checker functions
        flag_name = rule.get('flags_involved', [None])[0]
        if flag_name and flag_name in self.flag_traces:
            trace = self.flag_traces[flag_name]
            
            all_functions = []
            all_functions.extend([s.get('function', '') for s in trace.get('setters', [])])
            all_functions.extend([c.get('function', '') for c in trace.get('checkers', [])])
            
            # Map function names to phases
            for func in all_functions:
                if 'validate' in func or 'verify' in func:
                    if 'Edit' not in phases:
                        phases.append('Edit')
                if 'audit' in func or 'check_duplicate' in func or 'assess_fraud' in func:
                    if 'Audit' not in phases:
                        phases.append('Audit')
                if 'calculate' in func or 'apply' in func:
                    if 'Release' not in phases:
                        phases.append('Release')
        
        return phases if phases else ['Edit']  # Default to Edit phase
    
    def determine_criticality(self, rule: Dict) -> str:
        """Determine rule criticality level."""
        flag_name = rule.get('flags_involved', [None])[0] or ''
        
        # High criticality: Affects payment or prevents fraud
        if any(keyword in flag_name.lower() for keyword in ['emergency', 'fraud', 'duplicate']):
            return 'high'
        
        # Medium criticality: Affects processing flow
        if any(keyword in flag_name.lower() for keyword in ['high_value', 'auth']):
            return 'medium'
        
        # Default to medium
        return 'medium'
    
    def determine_rule_type(self, rule: Dict) -> str:
        """Determine the rule implementation type."""
        flag_name = rule.get('flags_involved', [None])[0]
        
        # All rules in this system are flag-driven
        if flag_name:
            return 'flag-driven'
        
        # Could add other types for future rules
        conditions = rule.get('conditions', [])
        if any('>' in str(c) or '<' in str(c) for c in conditions):
            return 'threshold'
        
        return 'flag-driven'
    
    def generate_business_description(self, rule: Dict, conditions: List[str], 
                                     actions: List[str]) -> str:
        """Generate human-readable business description."""
        rule_name = rule.get('rule_name', '').replace(' Rule', '')
        
        # Build description based on rule type
        if 'Emergency' in rule_name:
            return "When a claim is for an emergency procedure at an Emergency Room facility, the system marks it as emergency and applies premium pricing."
        elif 'High Value' in rule_name:
            return "When a claim amount exceeds the high-value threshold, the system flags it for additional review and fraud assessment."
        elif 'Duplicate' in rule_name:
            return "When a claim matches a previously paid claim for the same patient, procedure, and date, the system flags it as a duplicate."
        elif 'Fraud Risk' in rule_name:
            return "When a claim exhibits fraud risk indicators, the system flags it for fraud investigation."
        elif 'Requires Auth' in rule_name or 'Auth' in rule_name:
            return "When a procedure requires prior authorization, the system verifies authorization before processing."
        
        # Generic fallback
        if conditions and actions:
            # Try to make it readable
            primary_condition = conditions[0] if conditions else "conditions are met"
            primary_action = actions[0] if actions else "triggers system actions"
            return f"When {primary_condition.lower()}, the system {primary_action.lower()}."
        else:
            return f"Business rule for {rule_name.lower()} processing."
    
    def modernize_rule(self, rule: Dict) -> ModernizedRule:
        """Convert one technical rule to business-readable format."""
        
        # Extract business-readable components
        conditions = self.extract_business_conditions(rule)
        actions = self.extract_business_actions(rule)
        used_by = self.extract_used_by(rule)
        triggers = self.extract_triggers(rule)
        impacts = self.extract_impacts(rule)
        
        # Generate description
        description = self.generate_business_description(rule, conditions, actions)
        
        # Get technical details
        flag_name = rule.get('flags_involved', [None])[0] or ''
        
        setter_functions = []
        checker_functions = []
        
        if flag_name in self.flag_traces:
            trace = self.flag_traces[flag_name]
            setter_functions = trace.get('setters', [])
            checker_functions = trace.get('checkers', [])
        
        # Extract data inputs/outputs
        data_fields = rule.get('data_fields_used', [])
        
        # Inputs: fields used in conditions (facility_type, procedure_code, etc.)
        data_inputs = []
        for field in data_fields:
            if any(keyword in field.lower() for keyword in ['facility', 'procedure', 'network', 'patient_id', 'date', 'provider']):
                if field not in data_inputs:
                    data_inputs.append(field)
        
        # Outputs: fields modified by rule (flags, amount, status)
        data_outputs = []
        
        # Always include the flag itself
        if flag_name:
            data_outputs.append('c->flags')
        
        # Add other output fields
        for field in data_fields:
            if any(keyword in field.lower() for keyword in ['amount', 'status', 'cos', 'responsibility']):
                if field not in data_outputs:
                    data_outputs.append(field)
        
        # If no outputs found, infer from actions
        for action in actions:
            if 'amount' in action.lower() and 'c->final_amount' not in data_outputs:
                data_outputs.append('c->final_amount')
            if 'status' in action.lower() and 'c->status' not in data_outputs:
                data_outputs.append('c->status')
        
        # Get implementation files
        impl_files = list(set([loc.split(':')[0] for loc in rule.get('code_locations', [])]))
        
        # Enterprise metadata
        rule_category = self.categorize_rule(rule)
        business_domains = self.determine_domains(rule)
        execution_phases = self.determine_execution_phase(rule)
        criticality = self.determine_criticality(rule)
        rule_type = self.determine_rule_type(rule)
        
        # Create modernized rule
        return ModernizedRule(
            rule_id=f"BR-{flag_name.replace('FLAG_', '')}",
            rule_name=rule.get('rule_name', 'Unknown Rule'),
            business_description=description,
            conditions_if=conditions,
            actions_then=actions,
            used_by=used_by,
            triggers=triggers,
            impacts=impacts,
            flag_name=flag_name,
            setter_functions=setter_functions,
            checker_functions=checker_functions,
            data_inputs=data_inputs,
            data_outputs=data_outputs,
            implementation_files=impl_files,
            rule_category=rule_category,
            business_domain=business_domains,
            execution_phase=execution_phases,
            rule_criticality=criticality,
            rule_type=rule_type
        )
    
    def format_rule_for_display(self, modern_rule: ModernizedRule) -> str:
        """Format modernized rule for human reading."""
        
        output = []
        output.append("=" * 70)
        output.append(f"BUSINESS RULE: {modern_rule.rule_name}")
        output.append("=" * 70)
        
        # Header
        output.append(f"\nRule ID: {modern_rule.rule_id}")
        output.append(f"Category: {modern_rule.rule_category.upper()}")
        output.append(f"Domains: {', '.join(modern_rule.business_domain)}")
        output.append(f"Execution Phase: {', '.join(modern_rule.execution_phase)}")
        output.append(f"Criticality: {modern_rule.rule_criticality.upper()}")
        output.append(f"Type: {modern_rule.rule_type}")
        
        # Description
        output.append(f"\n📋 Description:")
        output.append(f"  {modern_rule.business_description}")
        
        # Core Logic - Structured Conditions
        output.append(f"\n🔍 IF (Conditions):")
        if modern_rule.conditions_if:
            for i, cond in enumerate(modern_rule.conditions_if, 1):
                output.append(f"  {i}. {cond.condition}")
                output.append(f"     Source: {cond.source_function}")
                output.append(f"     Relationship: {cond.relationship}")
        else:
            output.append("  (Always applies)")
        
        output.append(f"\n✓ THEN (Actions):")
        if modern_rule.actions_then:
            for i, action in enumerate(modern_rule.actions_then, 1):
                output.append(f"  {i}. {action}")
        else:
            output.append("  (No explicit actions)")
        
        # Context
        if modern_rule.triggers:
            output.append(f"\n⚡ Triggered By:")
            for trigger in modern_rule.triggers:
                output.append(f"  • {trigger}")
        
        if modern_rule.used_by:
            output.append(f"\n🎯 Used By:")
            for usage in modern_rule.used_by:
                output.append(f"  • {usage}")
        
        if modern_rule.impacts:
            output.append(f"\n💰 Impacts:")
            for impact in modern_rule.impacts:
                output.append(f"  • {impact}")
        
        # Data Dependencies
        if modern_rule.data_inputs:
            output.append(f"\n📥 Data Inputs:")
            for inp in modern_rule.data_inputs:
                output.append(f"  • {inp}")
        
        if modern_rule.data_outputs:
            output.append(f"\n📤 Data Outputs:")
            for out in modern_rule.data_outputs:
                output.append(f"  • {out}")
        
        # Technical Details
        output.append(f"\n🔧 Technical Details:")
        output.append(f"  Flag: {modern_rule.flag_name}")
        output.append(f"  Implementation Files: {', '.join(modern_rule.implementation_files)}")
        output.append(f"  Setter Functions: {len(modern_rule.setter_functions)}")
        output.append(f"  Checker Functions: {len(modern_rule.checker_functions)}")
        
        output.append("\n" + "=" * 70)
        
        return "\n".join(output)


def main():
    """Demo: Modernize the Emergency Rule."""
    import sys
    
    analysis_file = sys.argv[1] if len(sys.argv) > 1 else 'output/graph_analysis_for_llm.json'
    
    print("\n" + "="*70)
    print("Rule Modernizer - Proof of Concept")
    print("Focus: Emergency Rule")
    print("="*70)
    
    # Load and modernize
    modernizer = RuleModernizer(analysis_file)
    
    # Find Emergency Rule
    emergency_rule = None
    for rule in modernizer.business_rules:
        if 'Emergency' in rule.get('rule_name', ''):
            emergency_rule = rule
            break
    
    if not emergency_rule:
        print("\n❌ Emergency Rule not found!")
        return
    
    print("\n📊 Original Technical Rule:")
    print(json.dumps(emergency_rule, indent=2))
    
    # Modernize it
    print("\n🔄 Modernizing...")
    modern_rule = modernizer.modernize_rule(emergency_rule)
    
    # Display
    print("\n✨ MODERNIZED BUSINESS RULE:")
    print(modernizer.format_rule_for_display(modern_rule))
    
    # Export
    output_file = 'output/modernized_emergency_rule.json'
    with open(output_file, 'w') as f:
        json.dump(asdict(modern_rule), f, indent=2)
    
    print(f"\n💾 Exported to: {output_file}")
    
    # Also create a markdown version
    markdown_file = 'output/modernized_emergency_rule.md'
    with open(markdown_file, 'w') as f:
        f.write(modernizer.format_rule_for_display(modern_rule))
    
    print(f"💾 Markdown version: {markdown_file}")
    
    print("\n✅ Modernization complete!")
    print("\n💡 Next Steps:")
    print("  1. Review the modernized rule above")
    print("  2. Verify IF conditions make business sense")
    print("  3. Verify THEN actions are clear")
    print("  4. Check 'Used By' section for context")
    print("  5. Once satisfied, apply to all 5 rules!")


if __name__ == '__main__':
    main()