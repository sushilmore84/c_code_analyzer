#!/usr/bin/env python3
"""
LLM Rule Modernizer - Convert technical analysis to business-readable rules using LLM
Replaces heuristic-based rule_modernizer.py with LLM intelligence for any rule type.
"""

import json
import os
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from claims_analyzer.llm_client import LLMClient
from claims_analyzer.rule_modernizer import ConditionWithSource, ModernizedRule


class LLMRuleModernizer:
    """Convert technical analysis to business-readable rules using LLM intelligence.

    Unlike the heuristic-based RuleModernizer, this class uses an LLM to interpret
    technical C code analysis and produce business-readable rules for any rule type,
    not just predefined templates.

    Args:
        graph_analysis_file: Path to graph_analysis_for_llm.json from Stage 2.
        provider: LLM provider - "anthropic" or "openai".
        model: Model identifier for the chosen provider.
    """

    def __init__(
        self,
        graph_analysis_file: str,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-5-20250929",
    ):
        """Load graph analysis and initialize LLM client."""
        with open(graph_analysis_file) as f:
            self.data = json.load(f)

        self.business_rules = self.data.get("business_rules", [])
        self.flag_traces = self.data.get("flag_traces", {})
        self.provider = provider
        self.model = model
        self.llm = LLMClient()

    def _build_modernization_prompt(self, rule: Dict, flag_trace: Dict) -> str:
        """Build a detailed prompt for the LLM to modernize a technical rule.

        Args:
            rule: Technical business rule dict from graph analysis.
            flag_trace: Flag trace data showing setter/checker function details.

        Returns:
            Formatted prompt string for the LLM.
        """
        prompt = f"""You are a business rules expert analyzing a legacy C healthcare claims processing system.
Your task is to convert a technical business rule extracted from C code into a business-readable modernized rule.

## TECHNICAL RULE DATA

### Rule Information
{json.dumps(rule, indent=2)}

### Flag Trace (how this flag flows through the system)
{json.dumps(flag_trace, indent=2)}

## YOUR TASK

Convert the technical rule above into a business-readable modernized rule. You must produce a JSON object with EXACTLY these fields:

### Field Specifications

1. **rule_id** (string): Generate as "BR-" followed by the flag name without "FLAG_" prefix. Example: FLAG_EMERGENCY -> "BR-EMERGENCY". If no flag, use "BR-" + sanitized rule_name.

2. **rule_name** (string): Keep the original rule_name from the input.

3. **business_description** (string): Write 1-3 sentences describing what this rule does in plain business English. Explain the business purpose, not the technical implementation. A business analyst should understand this without knowing C code.

4. **conditions_if** (array of objects): Convert the raw C conditions into business-readable conditions. Each object must have:
   - "condition" (string): The condition in plain business English. Convert C syntax like "c->facility_type == FACILITY_ER" to "Claim facility type is Emergency Room". Remove pointer syntax, convert constants to readable names, convert operators to English words.
   - "source_function" (string): The function name where this condition originates (from the setter functions in the flag trace).
   - "relationship" (string): One of "OR", "AND", or "INDEPENDENT". Multiple setter functions typically indicate OR relationships. Conditions within the same function are typically AND.

5. **actions_then** (array of strings): Convert raw technical actions to business-readable actions. Each should be a clear statement like "Mark claim as emergency", "Apply emergency premium adjustment to base rate", "Calculate final reimbursement amount". Derive these from:
   - The flag being set (e.g., "Mark claim as [flag meaning]")
   - Checker functions that respond to this flag (their function names indicate what actions they take)
   - Raw actions from the rule data

6. **used_by** (array of strings): List the processes/functions that consume this rule's output. Format as "Context: Function Name In Title Case". Derive from checker functions. Example: "Pricing: Apply Emergency Premium", "Calculation: Calculate Cos Calculation".

7. **triggers** (array of strings): List what triggers this rule. Derive from setter function names. Format as title case. Example: "Validate Emergency Facility".

8. **impacts** (array of strings): List what this rule affects. Analyze the data fields and actions to determine impacts like "Claim reimbursement amount", "Claim processing status", "Patient financial responsibility", "Claim state/classification".

9. **flag_name** (string): The primary flag name, e.g., "FLAG_EMERGENCY". Use the first entry from flags_involved, or empty string if none.

10. **setter_functions** (array of objects): Copy directly from the flag trace's "setters" array. Each object has: function, file, line, condition.

11. **checker_functions** (array of objects): Copy directly from the flag trace's "checkers" array. Each object has: function, file, line, condition.

12. **data_inputs** (array of strings): From data_fields_used, identify fields that are READ as inputs (typically facility_type, procedure_code, network_status, patient_id, date fields, provider fields).

13. **data_outputs** (array of strings): Identify fields that are WRITTEN/modified by this rule. Always include "c->flags" if a flag is set. Include amount fields, status fields, etc.

14. **implementation_files** (array of strings): Extract unique file names from code_locations. Strip line numbers, keep just the filename.

15. **rule_category** (string): One of: "classification", "validation", "calculation", "audit". Determine based on rule purpose:
    - "classification": Rule classifies or categorizes claims (emergency, high-value)
    - "validation": Rule validates data or prerequisites (authorization checks)
    - "calculation": Rule computes values (pricing, amounts)
    - "audit": Rule detects anomalies or fraud (duplicate detection, fraud assessment)

16. **business_domain** (array of strings): One or more of: "claim_validation", "pricing", "fraud_detection", "network", "authorization", "claims_processing". Determine from what the rule does and what functions use it.

17. **execution_phase** (array of strings): One or more of: "Edit", "Audit", "Release". Determine from function names:
    - "Edit": validation, verification functions
    - "Audit": audit, duplicate check, fraud assessment functions
    - "Release": calculation, payment, apply functions

18. **rule_criticality** (string): One of: "high", "medium", "low". Determine based on:
    - "high": Affects payment amounts, prevents fraud, emergency handling
    - "medium": Affects processing flow, requires review
    - "low": Informational or logging

19. **rule_type** (string): One of: "flag-driven", "threshold", "lookup", "calculation". Most rules in flag-based systems are "flag-driven". Use "threshold" if conditions involve numeric comparisons.

## OUTPUT FORMAT

Return ONLY a JSON object (no markdown fencing, no explanation) with exactly the fields specified above. Example structure:

{{
  "rule_id": "BR-EMERGENCY",
  "rule_name": "Emergency Rule",
  "business_description": "When a claim is for an emergency procedure at an Emergency Room facility, the system marks it as emergency and applies premium pricing.",
  "conditions_if": [
    {{
      "condition": "Claim facility type is Emergency Room",
      "source_function": "validate_emergency_facility",
      "relationship": "OR"
    }}
  ],
  "actions_then": ["Mark claim as emergency", "Apply emergency premium adjustment to base rate"],
  "used_by": ["Pricing: Apply Emergency Premium"],
  "triggers": ["Validate Emergency Facility"],
  "impacts": ["Claim reimbursement amount"],
  "flag_name": "FLAG_EMERGENCY",
  "setter_functions": [],
  "checker_functions": [],
  "data_inputs": ["c->facility_type"],
  "data_outputs": ["c->flags"],
  "implementation_files": ["claim_validation.c"],
  "rule_category": "classification",
  "business_domain": ["claim_validation", "pricing"],
  "execution_phase": ["Edit", "Release"],
  "rule_criticality": "high",
  "rule_type": "flag-driven"
}}

IMPORTANT:
- Return ONLY the JSON object, no additional text
- All fields are required
- Arrays can be empty [] but must be present
- Strings can be empty "" but must be present
- For setter_functions and checker_functions: copy directly from the flag trace data provided above
"""
        return prompt

    def _parse_llm_response(self, response_text: str) -> Dict:
        """Parse LLM response to extract ModernizedRule JSON.

        Handles multiple response formats: raw JSON, fenced code blocks,
        and JSON embedded in explanatory text.

        Args:
            response_text: Raw text response from LLM.

        Returns:
            Parsed dictionary matching ModernizedRule structure.

        Raises:
            ValueError: If JSON cannot be extracted or parsed.
        """
        # Strategy 1: Try parsing the entire response as JSON
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Look for ```json ... ``` fenced block
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Look for ``` ... ``` fenced block (no json tag)
        json_match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 4: Find the outermost { ... } in the response
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Failed to parse JSON from LLM response. "
            f"Response preview: {response_text[:300]}..."
        )

    def _validate_modernized_rule(self, parsed: Dict, rule: Dict) -> Dict:
        """Validate and normalize parsed LLM output to match ModernizedRule schema.

        Fills in missing fields with defaults derived from the original rule data.
        Coerces types where possible (e.g., string to single-element list).

        Args:
            parsed: Raw parsed JSON from LLM.
            rule: Original technical rule data for fallback values.

        Returns:
            Normalized dictionary ready to construct ModernizedRule.
        """
        flag_name = (rule.get("flags_involved") or [None])[0] or ""

        schema = {
            "rule_id": (
                str,
                f"BR-{flag_name.replace('FLAG_', '')}" if flag_name else "BR-UNKNOWN",
            ),
            "rule_name": (str, rule.get("rule_name", "Unknown Rule")),
            "business_description": (str, ""),
            "conditions_if": (list, []),
            "actions_then": (list, []),
            "used_by": (list, []),
            "triggers": (list, []),
            "impacts": (list, []),
            "flag_name": (str, flag_name),
            "setter_functions": (list, []),
            "checker_functions": (list, []),
            "data_inputs": (list, []),
            "data_outputs": (list, []),
            "implementation_files": (list, []),
            "rule_category": (str, "classification"),
            "business_domain": (list, ["claims_processing"]),
            "execution_phase": (list, ["Edit"]),
            "rule_criticality": (str, "medium"),
            "rule_type": (str, "flag-driven"),
        }

        validated = {}
        for field, (expected_type, default) in schema.items():
            value = parsed.get(field, default)

            # Type coercion
            if expected_type == list and isinstance(value, str):
                value = [value]
            elif expected_type == str and isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            elif value is None:
                value = default

            validated[field] = value

        # Validate conditions_if entries have correct structure
        validated_conditions = []
        for cond in validated.get("conditions_if", []):
            if isinstance(cond, dict):
                validated_conditions.append(
                    {
                        "condition": cond.get("condition", ""),
                        "source_function": cond.get("source_function", ""),
                        "relationship": cond.get("relationship", "OR"),
                    }
                )
            elif isinstance(cond, str):
                validated_conditions.append(
                    {
                        "condition": cond,
                        "source_function": "",
                        "relationship": "OR",
                    }
                )
        validated["conditions_if"] = validated_conditions

        # Validate rule_category is one of allowed values
        allowed_categories = {"classification", "validation", "calculation", "audit"}
        if validated["rule_category"] not in allowed_categories:
            validated["rule_category"] = "classification"

        # Validate rule_criticality
        allowed_criticalities = {"high", "medium", "low"}
        if validated["rule_criticality"] not in allowed_criticalities:
            validated["rule_criticality"] = "medium"

        # Validate rule_type
        allowed_types = {"flag-driven", "threshold", "lookup", "calculation"}
        if validated["rule_type"] not in allowed_types:
            validated["rule_type"] = "flag-driven"

        return validated

    def _build_modernized_rule(
        self, validated: Dict, rule: Dict, flag_trace: Dict
    ) -> ModernizedRule:
        """Construct a ModernizedRule dataclass from validated LLM output.

        Merges LLM-generated intelligence with technical data from the original
        rule and flag trace.

        Args:
            validated: Validated and normalized dict from _validate_modernized_rule.
            rule: Original technical rule data.
            flag_trace: Flag trace data for this rule's flag.

        Returns:
            ModernizedRule instance.
        """
        conditions = [
            ConditionWithSource(
                condition=c["condition"],
                source_function=c["source_function"],
                relationship=c["relationship"],
            )
            for c in validated["conditions_if"]
        ]

        # Prefer flag trace data for setter/checker functions over LLM output
        setter_functions = validated.get("setter_functions", [])
        checker_functions = validated.get("checker_functions", [])

        if not setter_functions and flag_trace:
            setter_functions = flag_trace.get("setters", [])
        if not checker_functions and flag_trace:
            checker_functions = flag_trace.get("checkers", [])

        return ModernizedRule(
            rule_id=validated["rule_id"],
            rule_name=validated["rule_name"],
            business_description=validated["business_description"],
            conditions_if=conditions,
            actions_then=validated["actions_then"],
            used_by=validated["used_by"],
            triggers=validated["triggers"],
            impacts=validated["impacts"],
            flag_name=validated["flag_name"],
            setter_functions=setter_functions,
            checker_functions=checker_functions,
            data_inputs=validated["data_inputs"],
            data_outputs=validated["data_outputs"],
            implementation_files=validated["implementation_files"],
            rule_category=validated["rule_category"],
            business_domain=validated["business_domain"],
            execution_phase=validated["execution_phase"],
            rule_criticality=validated["rule_criticality"],
            rule_type=validated["rule_type"],
        )

    def _fallback_modernize(self, rule: Dict, flag_trace: Dict) -> ModernizedRule:
        """Fallback to heuristic-based modernization when LLM fails.

        Delegates to the original RuleModernizer class for graceful degradation.

        Args:
            rule: Original technical rule data.
            flag_trace: Flag trace data.

        Returns:
            ModernizedRule from heuristic-based modernizer.
        """
        from claims_analyzer.rule_modernizer import RuleModernizer

        flag_name = (rule.get("flags_involved") or [None])[0] or ""
        fake_analysis = {
            "business_rules": [rule],
            "flag_traces": {flag_name: flag_trace} if flag_trace else {},
        }

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(tmp_fd, "w") as tmp:
                json.dump(fake_analysis, tmp)

            heuristic = RuleModernizer(tmp_path)
            return heuristic.modernize_rule(rule)
        finally:
            os.unlink(tmp_path)

    def modernize_rule(self, rule: Dict) -> ModernizedRule:
        """Modernize a single rule using LLM intelligence.

        Sends the rule's technical data to the LLM for conversion to
        business-readable format. Falls back to heuristic modernization
        on failure.

        Args:
            rule: Technical rule dict from graph_analysis_for_llm.json.

        Returns:
            ModernizedRule with business-readable content.
        """
        flag_name = (rule.get("flags_involved") or [None])[0]
        flag_trace = self.flag_traces.get(flag_name, {}) if flag_name else {}

        try:
            prompt = self._build_modernization_prompt(rule, flag_trace)

            print(f"  Calling {self.provider}/{self.model}...")
            response = self.llm.call_llm(
                provider=self.provider,
                model=self.model,
                prompt=prompt,
                max_tokens=4000,
                temperature=0.2,
            )

            response_text = response["text"]

            parsed = self._parse_llm_response(response_text)
            validated = self._validate_modernized_rule(parsed, rule)
            modernized = self._build_modernized_rule(validated, rule, flag_trace)

            print(f"  Successfully modernized: {modernized.rule_name}")
            return modernized

        except Exception as e:
            print(f"  LLM modernization failed: {e}")
            print(f"  Falling back to heuristic modernization...")
            return self._fallback_modernize(rule, flag_trace)

    def modernize_all_rules(self) -> List[ModernizedRule]:
        """Modernize all rules from the graph analysis.

        Processes every business rule found in the graph analysis data,
        not just specific known rule types.

        Returns:
            List of ModernizedRule instances.
        """
        modernized_rules = []
        total = len(self.business_rules)

        for i, rule in enumerate(self.business_rules, 1):
            rule_name = rule.get("rule_name", "Unknown")
            print(f"\n[{i}/{total}] Modernizing: {rule_name}")

            modernized = self.modernize_rule(rule)
            modernized_rules.append(modernized)

        return modernized_rules

    def format_rule_for_display(self, modern_rule: ModernizedRule) -> str:
        """Format modernized rule for human reading.

        Args:
            modern_rule: The modernized rule to display.

        Returns:
            Formatted string for console output.
        """
        output = []
        output.append("=" * 70)
        output.append(f"BUSINESS RULE: {modern_rule.rule_name}")
        output.append("=" * 70)

        output.append(f"\nRule ID: {modern_rule.rule_id}")
        output.append(f"Category: {modern_rule.rule_category.upper()}")
        output.append(f"Domains: {', '.join(modern_rule.business_domain)}")
        output.append(f"Execution Phase: {', '.join(modern_rule.execution_phase)}")
        output.append(f"Criticality: {modern_rule.rule_criticality.upper()}")
        output.append(f"Type: {modern_rule.rule_type}")

        output.append(f"\nDescription:")
        output.append(f"  {modern_rule.business_description}")

        output.append(f"\nIF (Conditions):")
        if modern_rule.conditions_if:
            for i, cond in enumerate(modern_rule.conditions_if, 1):
                output.append(f"  {i}. {cond.condition}")
                output.append(f"     Source: {cond.source_function}")
                output.append(f"     Relationship: {cond.relationship}")
        else:
            output.append("  (Always applies)")

        output.append(f"\nTHEN (Actions):")
        if modern_rule.actions_then:
            for i, action in enumerate(modern_rule.actions_then, 1):
                output.append(f"  {i}. {action}")
        else:
            output.append("  (No explicit actions)")

        if modern_rule.triggers:
            output.append(f"\nTriggered By:")
            for trigger in modern_rule.triggers:
                output.append(f"  - {trigger}")

        if modern_rule.used_by:
            output.append(f"\nUsed By:")
            for usage in modern_rule.used_by:
                output.append(f"  - {usage}")

        if modern_rule.impacts:
            output.append(f"\nImpacts:")
            for impact in modern_rule.impacts:
                output.append(f"  - {impact}")

        if modern_rule.data_inputs:
            output.append(f"\nData Inputs:")
            for inp in modern_rule.data_inputs:
                output.append(f"  - {inp}")

        if modern_rule.data_outputs:
            output.append(f"\nData Outputs:")
            for out in modern_rule.data_outputs:
                output.append(f"  - {out}")

        output.append(f"\nTechnical Details:")
        output.append(f"  Flag: {modern_rule.flag_name}")
        output.append(
            f"  Implementation Files: {', '.join(modern_rule.implementation_files)}"
        )
        output.append(f"  Setter Functions: {len(modern_rule.setter_functions)}")
        output.append(f"  Checker Functions: {len(modern_rule.checker_functions)}")

        output.append("\n" + "=" * 70)

        return "\n".join(output)


def main():
    """Modernize all business rules using LLM intelligence."""
    analysis_file = (
        sys.argv[1] if len(sys.argv) > 1 else "output/graph_analysis_for_llm.json"
    )
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output/modernized_rules.json"

    print("\n" + "=" * 70)
    print("LLM Rule Modernizer - Business Rule Conversion")
    print("Using LLM intelligence for ALL rule types")
    print("=" * 70)

    print(f"\nLoading graph analysis from: {analysis_file}")

    modernizer = LLMRuleModernizer(analysis_file)

    print(f"Found {len(modernizer.business_rules)} rules to modernize")
    modernized_rules = modernizer.modernize_all_rules()

    # Display results
    print("\n" + "=" * 70)
    print("MODERNIZED BUSINESS RULES")
    print("=" * 70)

    for rule in modernized_rules:
        print(modernizer.format_rule_for_display(rule))

    # Export all rules as JSON array
    rules_data = [asdict(rule) for rule in modernized_rules]

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(rules_data, f, indent=2)

    print(f"\nExported {len(modernized_rules)} rules to: {output_file}")

    # Export individual rule files for backward compatibility with Stage 4
    for rule in modernized_rules:
        individual_file = (
            f"output/modernized_{rule.rule_id.lower().replace('-', '_')}.json"
        )
        with open(individual_file, "w") as f:
            json.dump(asdict(rule), f, indent=2)
        print(f"Exported individual rule: {individual_file}")

    print("\nModernization complete!")
    print("\nNext Steps:")
    print("  1. Review the modernized rules")
    print("  2. Run llm_synthesizer.py to generate DMN tables")
    print("  3. Run llm_markdown_doc_generator.py for documentation")


if __name__ == "__main__":
    main()
