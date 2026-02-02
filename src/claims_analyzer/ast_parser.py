#!/usr/bin/env python3
"""
Enhanced Business Rule Extractor - Addresses all 5 gaps
Extracts: conditions, data fields, return codes, flag flow, complete lineage
"""

import clang.cindex as clang
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import json
import re


@dataclass
class ConditionInfo:
    """Detailed condition information from if/while statements."""
    expression: str  # The actual condition text
    line: int
    variables: List[str]  # Variables referenced in condition
    operators: List[str]  # Comparison operators (>, <, ==, etc.)
    constants: List[str]  # Literal values
    

@dataclass
class DataFieldAccess:
    """Track which struct fields are read/written."""
    field_name: str  # e.g., "claim->amount"
    access_type: str  # "read" or "write"
    line: int
    in_condition: bool  # Is this in an if condition?


@dataclass
class ReturnStatement:
    """Information about return statements."""
    value: str  # Return value expression
    line: int
    is_literal: bool  # True if it's a literal (0, 1, -1)
    context: str  # What led to this return (simplified)


@dataclass
class FlagFlow:
    """Track complete flag lifecycle."""
    flag_name: str
    set_by: List[str]  # Functions that set it
    checked_by: List[str]  # Functions that check it
    set_conditions: List[str]  # Conditions under which it's set
    check_conditions: List[str]  # Conditions under which it's checked


@dataclass
class EnhancedFunctionInfo:
    """Complete function information for business rule extraction."""
    # Basic info (required fields first)
    name: str
    file: str
    line: int
    returns_type: str
    parameters: List[str]
    calls_functions: List[str]
    sets_flags: List[str]
    checks_flags: List[str]
    
    # Optional fields (with defaults) - must come after required fields
    called_by: List[str] = field(default_factory=list)
    flag_set_conditions: Dict[str, str] = field(default_factory=dict)
    flag_check_conditions: Dict[str, str] = field(default_factory=dict)
    conditions: List[ConditionInfo] = field(default_factory=list)
    has_conditionals: int = 0
    data_fields_read: List[DataFieldAccess] = field(default_factory=list)
    data_fields_written: List[DataFieldAccess] = field(default_factory=list)
    return_statements: List[ReturnStatement] = field(default_factory=list)
    complexity: int = 0
    is_business_logic: bool = False
    business_reason: str = ""


class CompilationDatabase:
    """Handle compilation database (compile_commands.json)."""
    
    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        self.commands = {}
        self._load()
    
    def _load(self):
        """Load compile_commands.json if it exists."""
        compile_db_path = self.source_dir / 'compile_commands.json'
        
        if compile_db_path.exists():
            print("✓ Found compile_commands.json")
            with open(compile_db_path) as f:
                data = json.load(f)
            
            for entry in data:
                try:
                    file_path = Path(entry['file'])
                    filename = file_path.name
                    
                    # Handle both 'command' (string) and 'arguments' (array) formats
                    if 'command' in entry:
                        command = entry['command']
                        flags = self._extract_flags_from_command(command)
                    elif 'arguments' in entry:
                        arguments = entry['arguments']
                        flags = self._extract_flags_from_arguments(arguments)
                    else:
                        continue
                    
                    self.commands[filename] = {
                        'flags': flags,
                        'directory': entry.get('directory', str(self.source_dir)),
                        'file': entry['file']
                    }
                except KeyError:
                    continue
            
            print(f"✓ Loaded compilation info for {len(self.commands)} files")
        else:
            print("⚠️  No compile_commands.json found")
    
    def _extract_flags_from_command(self, command: str) -> List[str]:
        """Extract compiler flags from compilation command string."""
        flags = []
        parts = command.split()
        
        i = 0
        while i < len(parts):
            part = parts[i]
            
            if part in ['gcc', 'clang', 'cc'] or part.endswith('/gcc') or part.endswith('/clang'):
                i += 1
                continue
            
            if part == '-o':
                i += 2
                continue
            
            if part.endswith('.c') or part.endswith('.o') or part == '-c':
                i += 1
                continue
            
            if part.startswith('-'):
                flags.append(part)
                if part in ['-I', '-D', '-isystem', '-include']:
                    if i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                        i += 1
                        flags.append(parts[i])
            
            i += 1
        
        return flags
    
    def _extract_flags_from_arguments(self, arguments: List[str]) -> List[str]:
        """Extract compiler flags from arguments array."""
        flags = []
        
        i = 0
        while i < len(arguments):
            arg = arguments[i]
            
            if i == 0 or arg in ['gcc', 'clang', 'cc']:
                i += 1
                continue
            
            if arg == '-o':
                i += 2
                continue
            
            if arg.endswith('.c') or arg.endswith('.o') or arg == '-c':
                i += 1
                continue
            
            if arg.startswith('-'):
                flags.append(arg)
                if arg in ['-I', '-D', '-isystem', '-include']:
                    if i + 1 < len(arguments) and not arguments[i + 1].startswith('-'):
                        i += 1
                        flags.append(arguments[i])
            
            i += 1
        
        return flags
    
    def get_flags_for_file(self, filename: str) -> Optional[List[str]]:
        """Get compilation flags for a specific file."""
        if filename in self.commands:
            return self.commands[filename]['flags']
        return None
    
    def get_all_files(self) -> List[str]:
        """Get list of all files in compilation database."""
        return list(self.commands.keys())


class EnhancedClaimsParser:
    """Enhanced parser that captures complete business rule information."""
    
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.index = clang.Index.create()
        self.functions: List[EnhancedFunctionInfo] = []
        self.all_flags: Set[str] = set()
        self.flag_flows: Dict[str, FlagFlow] = {}
        
        # For building reverse call graph
        self.call_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Load compilation database
        self.comp_db = CompilationDatabase(self.source_dir)
    
    def extract_condition_expression(self, node: clang.Cursor) -> ConditionInfo:
        """Extract the actual condition expression text and components."""
        # Get all tokens in the condition
        tokens = list(node.get_tokens())
        
        if not tokens:
            return ConditionInfo("", node.location.line, [], [], [])
        
        # Build expression text
        expression = ' '.join(t.spelling for t in tokens)
        
        # Extract variables (identifiers that aren't keywords)
        keywords = {'if', 'while', 'for', 'return', 'int', 'double', 'char', 'void'}
        variables = []
        operators = []
        constants = []
        
        for token in tokens:
            if token.kind == clang.TokenKind.IDENTIFIER and token.spelling not in keywords:
                if '->' in ' '.join(t.spelling for t in tokens):
                    # Try to capture struct field access
                    idx = tokens.index(token)
                    if idx > 0 and tokens[idx-1].spelling == '->':
                        if idx > 1:
                            variables.append(f"{tokens[idx-2].spelling}->{token.spelling}")
                    elif idx < len(tokens)-2 and tokens[idx+1].spelling == '->':
                        pass  # Will be captured with next token
                    else:
                        variables.append(token.spelling)
                else:
                    variables.append(token.spelling)
            
            elif token.spelling in ['>', '<', '==', '!=', '>=', '<=', '&&', '||']:
                operators.append(token.spelling)
            
            elif token.kind == clang.TokenKind.LITERAL:
                constants.append(token.spelling)
        
        return ConditionInfo(
            expression=expression,
            line=node.location.line,
            variables=list(set(variables)),
            operators=operators,
            constants=constants
        )
    
    def extract_conditions(self, node: clang.Cursor) -> List[ConditionInfo]:
        """Extract all condition expressions from a function."""
        conditions = []
        
        for child in node.walk_preorder():
            if child.kind == clang.CursorKind.IF_STMT:
                # First child is usually the condition
                for subchild in child.get_children():
                    if subchild.kind != clang.CursorKind.COMPOUND_STMT:
                        condition = self.extract_condition_expression(subchild)
                        conditions.append(condition)
                        break
            
            elif child.kind == clang.CursorKind.WHILE_STMT:
                # First child is the condition
                for subchild in child.get_children():
                    condition = self.extract_condition_expression(subchild)
                    conditions.append(condition)
                    break
        
        return conditions
    
    def extract_data_field_accesses(self, node: clang.Cursor) -> Tuple[List[DataFieldAccess], List[DataFieldAccess]]:
        """Extract all struct field reads and writes."""
        reads = []
        writes = []
        
        # Track if we're in a condition
        in_condition = False
        
        for child in node.walk_preorder():
            # Detect if we're in a condition
            if child.kind in [clang.CursorKind.IF_STMT, clang.CursorKind.WHILE_STMT]:
                in_condition = True
            
            # Look for member access (e.g., c->amount, provider->status)
            if child.kind == clang.CursorKind.MEMBER_REF_EXPR:
                tokens = list(child.get_tokens())
                if len(tokens) >= 3:  # object, ->, field
                    field_access = f"{tokens[0].spelling}->{tokens[2].spelling}"
                    
                    # Determine if it's a read or write
                    # Simple heuristic: if it's on LHS of assignment, it's a write
                    parent = child.semantic_parent
                    is_write = False
                    
                    # Check if this is part of an assignment (only if parent exists)
                    if parent is not None:
                        try:
                            for sibling in parent.get_children():
                                if sibling.kind in [clang.CursorKind.BINARY_OPERATOR, 
                                                   clang.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR]:
                                    # Check if our node is on the left
                                    children = list(sibling.get_children())
                                    if children and children[0] == child:
                                        is_write = True
                                        break
                        except:
                            # If we can't determine, assume it's a read
                            pass
                    
                    access = DataFieldAccess(
                        field_name=field_access,
                        access_type="write" if is_write else "read",
                        line=child.location.line,
                        in_condition=in_condition
                    )
                    
                    if is_write:
                        writes.append(access)
                    else:
                        reads.append(access)
        
        return reads, writes
    
    def extract_return_statements(self, node: clang.Cursor) -> List[ReturnStatement]:
        """Extract all return statements and their context."""
        returns = []
        
        for child in node.walk_preorder():
            if child.kind == clang.CursorKind.RETURN_STMT:
                tokens = list(child.get_tokens())
                
                if len(tokens) > 1:  # Has a return value
                    value = ' '.join(t.spelling for t in tokens[1:])  # Skip 'return' keyword
                    is_literal = value.strip() in ['0', '1', '-1', 'RC_OK', 'RC_REJECT', 'RC_PEND']
                    
                    # Try to get context (simplified - just the line before)
                    context = "unconditional"
                    
                    returns.append(ReturnStatement(
                        value=value,
                        line=child.location.line,
                        is_literal=is_literal,
                        context=context
                    ))
        
        return returns
    
    def find_flag_operations_enhanced(self, node: clang.Cursor) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Find flag operations WITH their conditions."""
        sets_with_conditions = {}
        checks_with_conditions = {}
        
        current_condition = None
        
        for child in node.walk_preorder():
            # Track current condition context
            if child.kind == clang.CursorKind.IF_STMT:
                # Get condition
                for subchild in child.get_children():
                    if subchild.kind != clang.CursorKind.COMPOUND_STMT:
                        tokens = list(subchild.get_tokens())
                        current_condition = ' '.join(t.spelling for t in tokens)
                        break
            
            # Look for flag setting
            if child.kind == clang.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
                tokens = list(child.get_tokens())
                for token in tokens:
                    if 'FLAG_' in token.spelling:
                        sets_with_conditions[token.spelling] = current_condition or "unconditional"
                        self.all_flags.add(token.spelling)
            
            # Look for flag checking
            if child.kind == clang.CursorKind.BINARY_OPERATOR:
                tokens = list(child.get_tokens())
                for token in tokens:
                    if 'FLAG_' in token.spelling:
                        # The condition itself involves the flag
                        condition_expr = ' '.join(t.spelling for t in tokens)
                        checks_with_conditions[token.spelling] = condition_expr
                        self.all_flags.add(token.spelling)
        
        return sets_with_conditions, checks_with_conditions
    
    def analyze_function_enhanced(self, node: clang.Cursor) -> EnhancedFunctionInfo:
        """Analyze function with complete information."""
        # Basic info
        params = []
        for child in node.get_children():
            if child.kind == clang.CursorKind.PARM_DECL:
                params.append(f"{child.type.spelling} {child.spelling}")
        
        # Flag operations with conditions
        sets_with_cond, checks_with_cond = self.find_flag_operations_enhanced(node)
        
        # Conditions (MISSING #1)
        conditions = self.extract_conditions(node)
        
        # Data fields (MISSING #4)
        data_reads, data_writes = self.extract_data_field_accesses(node)
        
        # Return statements (MISSING #3)
        return_stmts = self.extract_return_statements(node)
        
        # Function calls
        calls = []
        for child in node.walk_preorder():
            if child.kind == clang.CursorKind.CALL_EXPR:
                for token in child.get_tokens():
                    if token.kind == clang.TokenKind.IDENTIFIER:
                        func_name = token.spelling
                        calls.append(func_name)
                        # Build call graph
                        self.call_graph[func_name].append(node.spelling)
                        break
        
        # Complexity
        complexity = len(conditions) + len(calls)
        
        # Business logic classification (MISSING #5)
        is_business, reason = self.classify_business_logic(
            node.spelling, conditions, sets_with_cond, checks_with_cond, 
            data_reads, data_writes, return_stmts
        )
        
        return EnhancedFunctionInfo(
            # Required fields first
            name=node.spelling,
            file=node.location.file.name if node.location.file else "unknown",
            line=node.location.line,
            returns_type=node.result_type.spelling,
            parameters=params,
            calls_functions=calls,
            sets_flags=list(sets_with_cond.keys()),
            checks_flags=list(checks_with_cond.keys()),
            # Optional fields
            called_by=[],  # Will be filled later
            flag_set_conditions=sets_with_cond,
            flag_check_conditions=checks_with_cond,
            conditions=conditions,
            has_conditionals=len(conditions),
            data_fields_read=data_reads,
            data_fields_written=data_writes,
            return_statements=return_stmts,
            complexity=complexity,
            is_business_logic=is_business,
            business_reason=reason
        )
    
    def classify_business_logic(self, name: str, conditions: List[ConditionInfo],
                               sets_flags: Dict, checks_flags: Dict,
                               data_reads: List, data_writes: List,
                               returns: List) -> Tuple[bool, str]:
        """
        Classify if function contains business logic with explanation.
        MISSING #5 - Now we explain WHY it's business logic.
        """
        reasons = []
        
        # Check 1: Business keywords in name
        business_keywords = ['validate', 'verify', 'check', 'calculate', 'apply',
                           'audit', 'assess', 'review', 'process', 'determine',
                           'approve', 'reject', 'evaluate']
        
        name_lower = name.lower()
        if any(kw in name_lower for kw in business_keywords):
            reasons.append(f"name contains business keyword '{name_lower}'")
        
        # Check 2: Multiple conditional paths (business decisions)
        if len(conditions) >= 2:
            reasons.append(f"contains {len(conditions)} conditional branches (decision logic)")
        
        # Check 3: Works with business flags
        if sets_flags or checks_flags:
            flag_names = list(sets_flags.keys()) + list(checks_flags.keys())
            reasons.append(f"manages business state flags: {', '.join(flag_names)}")
        
        # Check 4: Reads/writes business data
        if len(data_reads) >= 3:
            fields = set(d.field_name for d in data_reads)
            reasons.append(f"reads {len(fields)} business data fields")
        
        # Check 5: Has meaningful return codes (business outcomes)
        if any(r.is_literal and r.value in ['0', '1', '-1'] for r in returns):
            reasons.append("returns business decision codes (0/1/-1)")
        
        # Check 6: High complexity (complex business logic)
        if len(conditions) + len(data_reads) >= 5:
            reasons.append(f"high complexity (conditions + data access)")
        
        is_business = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "not business logic"
        
        return is_business, reason
    
    def build_flag_flows(self):
        """
        Build complete flag flow lineage (MISSING #2).
        Shows: who sets flag, under what condition, who checks it, and why.
        """
        for flag in self.all_flags:
            flow = FlagFlow(
                flag_name=flag,
                set_by=[],
                checked_by=[],
                set_conditions=[],
                check_conditions=[]
            )
            
            for func in self.functions:
                if flag in func.sets_flags:
                    flow.set_by.append(func.name)
                    condition = func.flag_set_conditions.get(flag, "unconditional")
                    flow.set_conditions.append(f"{func.name}: {condition}")
                
                if flag in func.checks_flags:
                    flow.checked_by.append(func.name)
                    condition = func.flag_check_conditions.get(flag, "unconditional")
                    flow.check_conditions.append(f"{func.name}: {condition}")
            
            self.flag_flows[flag] = flow
    
    def parse_file(self, filename: str) -> Optional[clang.TranslationUnit]:
        """Parse a C file."""
        filepath = self.source_dir / filename
        
        if not filepath.exists():
            return None
        
        flags = self.comp_db.get_flags_for_file(filename)
        if flags is None:
            flags = ['-std=c99', f'-I{self.source_dir}']
        
        tu = self.index.parse(str(filepath), args=flags)
        return tu
    
    def parse_all_files(self) -> List[EnhancedFunctionInfo]:
        """Parse all files and extract enhanced information."""
        c_files = self.comp_db.get_all_files()
        
        if not c_files:
            c_files = ['main.c', 'reference_data.c', 'claim_validation.c',
                      'network_verification.c', 'cos_calculation.c', 'audit.c']
        
        for c_file in c_files:
            tu = self.parse_file(c_file)
            if tu is None:
                continue
            
            for node in tu.cursor.walk_preorder():
                if node.kind == clang.CursorKind.FUNCTION_DECL and node.is_definition():
                    func_info = self.analyze_function_enhanced(node)
                    self.functions.append(func_info)
        
        # Build reverse call graph
        for func in self.functions:
            if func.name in self.call_graph:
                func.called_by = self.call_graph[func.name]
        
        # Build flag flows
        self.build_flag_flows()
        
        return self.functions
    
    def export_to_json(self, filename: str = "enhanced_analysis.json"):
        """Export complete analysis."""
        # Convert dataclasses to dicts, handling nested objects
        def convert_to_dict(obj):
            if isinstance(obj, (ConditionInfo, DataFieldAccess, ReturnStatement)):
                return asdict(obj)
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        functions_data = []
        for func in self.functions:
            func_dict = asdict(func)
            # Convert nested dataclasses
            func_dict['conditions'] = [asdict(c) for c in func.conditions]
            func_dict['data_fields_read'] = [asdict(d) for d in func.data_fields_read]
            func_dict['data_fields_written'] = [asdict(d) for d in func.data_fields_written]
            func_dict['return_statements'] = [asdict(r) for r in func.return_statements]
            functions_data.append(func_dict)
        
        data = {
            'metadata': {
                'source_dir': str(self.source_dir),
                'total_functions': len(self.functions),
                'business_functions': sum(1 for f in self.functions if f.is_business_logic),
                'total_flags': len(self.all_flags),
            },
            'functions': functions_data,
            'flag_flows': {k: asdict(v) for k, v in self.flag_flows.items()},
            'flags': list(self.all_flags),
        }
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Exported enhanced analysis to: {filename}")
        return data


def main():
    import sys
    
    source_dir = sys.argv[1] if len(sys.argv) > 1 else "../claims_system"
    
    print("\n" + "="*70)
    print("ENHANCED Business Rule Extractor")
    print("Addresses ALL 5 Missing Pieces")
    print("="*70)
    
    parser = EnhancedClaimsParser(source_dir)
    functions = parser.parse_all_files()
    
    print(f"\n✓ Parsed {len(functions)} functions")
    print(f"✓ Found {sum(1 for f in functions if f.is_business_logic)} business logic functions")
    print(f"✓ Tracked {len(parser.all_flags)} flags with complete flow")
    
    # Show examples of enhanced data
    print("\n" + "="*70)
    print("EXAMPLE: Enhanced Analysis")
    print("="*70)
    
    # Find a function with rich information
    for func in functions:
        if func.name == "apply_emergency_premium":
            print(f"\nFunction: {func.name}")
            print(f"Business Logic: {func.is_business_logic}")
            print(f"Reason: {func.business_reason}")
            print(f"\nConditions:")
            for cond in func.conditions:
                print(f"  - {cond.expression}")
                print(f"    Variables: {cond.variables}")
                print(f"    Operators: {cond.operators}")
            print(f"\nData Fields Read:")
            for field in func.data_fields_read[:5]:
                print(f"  - {field.field_name} (line {field.line})")
            print(f"\nReturn Statements:")
            for ret in func.return_statements:
                print(f"  - Returns {ret.value} at line {ret.line}")
            break
    
    # Show flag flow
    print("\n" + "="*70)
    print("FLAG FLOW: FLAG_EMERGENCY")
    print("="*70)
    if 'FLAG_EMERGENCY' in parser.flag_flows:
        flow = parser.flag_flows['FLAG_EMERGENCY']
        print(f"\nSet by: {', '.join(flow.set_by)}")
        print("Set conditions:")
        for cond in flow.set_conditions:
            print(f"  - {cond}")
        print(f"\nChecked by: {', '.join(flow.checked_by)}")
        print("Check conditions:")
        for cond in flow.check_conditions:
            print(f"  - {cond}")
    
    # Export
    parser.export_to_json('output/enhanced_analysis.json')
    
    print("\n✅ Enhanced analysis complete!")


if __name__ == '__main__':
    main()
