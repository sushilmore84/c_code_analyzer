#!/usr/bin/env python3
"""
AST parser for C code using libclang with compile_commands.json support.
Handles both 'command' (string) and 'arguments' (array) formats.
"""

import clang.cindex as clang
from pathlib import Path
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class FunctionInfo:
    """Information about a C function."""
    name: str
    file: str
    line: int
    returns_type: str
    parameters: List[str]
    calls_functions: List[str]
    sets_flags: List[str]
    checks_flags: List[str]
    has_conditionals: int
    complexity: int


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
            
            # Build a map of filename -> compilation command
            for entry in data:
                try:
                    file_path = Path(entry['file'])
                    filename = file_path.name
                    
                    # Handle both 'command' (string) and 'arguments' (array) formats
                    if 'command' in entry:
                        # Format 1: "command": "gcc -std=c99 -c file.c"
                        command = entry['command']
                        flags = self._extract_flags_from_command(command)
                    elif 'arguments' in entry:
                        # Format 2: "arguments": ["gcc", "-std=c99", "-c", "file.c"]
                        arguments = entry['arguments']
                        flags = self._extract_flags_from_arguments(arguments)
                    else:
                        print(f"⚠️  Entry for {filename} has neither 'command' nor 'arguments', skipping")
                        continue
                    
                    self.commands[filename] = {
                        'flags': flags,
                        'directory': entry.get('directory', str(self.source_dir)),
                        'file': entry['file']
                    }
                except KeyError as e:
                    print(f"⚠️  Malformed entry in compile_commands.json: missing {e}")
                    continue
            
            print(f"✓ Loaded compilation info for {len(self.commands)} files")
        else:
            print("⚠️  No compile_commands.json found")
            print("   Run: cd claims_system && bear -- make clean all")
    
    def _extract_flags_from_command(self, command: str) -> List[str]:
        """Extract compiler flags from compilation command string."""
        flags = []
        parts = command.split()
        
        i = 0
        while i < len(parts):
            part = parts[i]
            
            # Skip the compiler itself
            if part in ['gcc', 'clang', 'cc', '/usr/bin/gcc', '/usr/bin/clang'] or part.endswith('/gcc') or part.endswith('/clang'):
                i += 1
                continue
            
            # Skip output file
            if part == '-o':
                i += 2  # Skip -o and the filename
                continue
            
            # Skip input file (ends with .c)
            if part.endswith('.c') or part.endswith('.o'):
                i += 1
                continue
            
            # Skip -c flag (compile only)
            if part == '-c':
                i += 1
                continue
            
            # Keep all flags
            if part.startswith('-'):
                flags.append(part)
                
                # Handle flags with separate arguments
                if part in ['-I', '-D', '-isystem', '-include', '-iquote', '-isysroot']:
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
            
            # Skip the compiler
            if i == 0 or arg in ['gcc', 'clang', 'cc']:
                i += 1
                continue
            
            # Skip output file
            if arg == '-o':
                i += 2
                continue
            
            # Skip input files
            if arg.endswith('.c') or arg.endswith('.o'):
                i += 1
                continue
            
            # Skip -c flag
            if arg == '-c':
                i += 1
                continue
            
            # Keep all flags
            if arg.startswith('-'):
                flags.append(arg)
                
                # Handle flags with separate arguments
                if arg in ['-I', '-D', '-isystem', '-include', '-iquote', '-isysroot']:
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
    
    def has_file(self, filename: str) -> bool:
        """Check if file is in compilation database."""
        return filename in self.commands


class ClaimsSystemParser:
    """Parser for claims processing C code using compile_commands.json."""
    
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.index = clang.Index.create()
        self.functions = []
        self.all_flags = set()
        
        # Load compilation database
        print("\n" + "="*70)
        print("STEP 1: Loading Compilation Database")
        print("="*70)
        self.comp_db = CompilationDatabase(self.source_dir)
        print()
    
    def get_flags_for_file(self, filename: str) -> List[str]:
        """Get compilation flags for a specific file."""
        # First try compilation database
        flags = self.comp_db.get_flags_for_file(filename)
        if flags:
            return flags
        
        # Fallback: parse Makefile
        print(f"⚠️  No compilation info for {filename}, using Makefile fallback")
        return self._parse_makefile_flags()
    
    def _parse_makefile_flags(self) -> List[str]:
        """Fallback: Parse Makefile for flags."""
        makefile = self.source_dir / 'Makefile'
        
        if not makefile.exists():
            print("⚠️  No Makefile found, using defaults")
            return ['-std=c99', f'-I{self.source_dir}']
        
        with open(makefile) as f:
            for line in f:
                if line.strip().startswith('CFLAGS'):
                    flags = line.split('=')[1].strip().split()
                    # Make relative paths absolute
                    flags = [f if not f.startswith('-I.') 
                            else f'-I{self.source_dir}' 
                            for f in flags]
                    return flags
        
        return ['-std=c99', f'-I{self.source_dir}']
    
    def parse_file(self, filename: str) -> Optional[clang.TranslationUnit]:
        """Parse a C file with correct compilation flags."""
        filepath = self.source_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  File not found: {filename}")
            return None
        
        print(f"📄 Parsing: {filename}")
        
        # Get flags from compilation database
        args = self.get_flags_for_file(filename)
        print(f"   Flags: {' '.join(args)}")
        
        # Parse with libclang
        tu = self.index.parse(str(filepath), args=args)
        
        # Check for parsing errors
        has_errors = False
        for diag in tu.diagnostics:
            if diag.severity >= clang.Diagnostic.Error:
                print(f"  ⚠️  Error: {diag.spelling}")
                has_errors = True
        
        if not has_errors:
            print(f"  ✓ Parsed successfully")
        
        return tu
    
    def find_flag_operations(self, node: clang.Cursor) -> tuple:
        """Find flag set and check operations."""
        sets_flags = []
        checks_flags = []
        
        for child in node.walk_preorder():
            # Look for flag setting: c->flags |= FLAG_EMERGENCY
            if child.kind == clang.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
                tokens = list(child.get_tokens())
                for token in tokens:
                    if 'FLAG_' in token.spelling:
                        sets_flags.append(token.spelling)
                        self.all_flags.add(token.spelling)
            
            # Look for flag checking: c->flags & FLAG_EMERGENCY
            if child.kind == clang.CursorKind.BINARY_OPERATOR:
                tokens = list(child.get_tokens())
                for token in tokens:
                    if 'FLAG_' in token.spelling:
                        checks_flags.append(token.spelling)
                        self.all_flags.add(token.spelling)
        
        return sets_flags, checks_flags
    
    def count_conditionals(self, node: clang.Cursor) -> int:
        """Count if statements in a function."""
        count = 0
        for child in node.walk_preorder():
            if child.kind == clang.CursorKind.IF_STMT:
                count += 1
        return count
    
    def find_function_calls(self, node: clang.Cursor) -> List[str]:
        """Find all function calls in a function."""
        calls = []
        for child in node.walk_preorder():
            if child.kind == clang.CursorKind.CALL_EXPR:
                for token in child.get_tokens():
                    if token.kind == clang.TokenKind.IDENTIFIER:
                        calls.append(token.spelling)
                        break
        return calls
    
    def analyze_function(self, node: clang.Cursor) -> FunctionInfo:
        """Analyze a single function."""
        # Get parameters
        params = []
        for child in node.get_children():
            if child.kind == clang.CursorKind.PARM_DECL:
                params.append(f"{child.type.spelling} {child.spelling}")
        
        # Find flag operations
        sets_flags, checks_flags = self.find_flag_operations(node)
        
        # Count conditionals (business logic indicator)
        num_conditionals = self.count_conditionals(node)
        
        # Find function calls
        function_calls = self.find_function_calls(node)
        
        # Calculate complexity
        complexity = num_conditionals + len(function_calls)
        
        return FunctionInfo(
            name=node.spelling,
            file=node.location.file.name if node.location.file else "unknown",
            line=node.location.line,
            returns_type=node.result_type.spelling,
            parameters=params,
            calls_functions=function_calls,
            sets_flags=sets_flags,
            checks_flags=checks_flags,
            has_conditionals=num_conditionals,
            complexity=complexity
        )
    
    def parse_all_files(self, c_files: Optional[List[str]] = None) -> List[FunctionInfo]:
        """Parse all C files."""
        print("\n" + "="*70)
        print("STEP 2: Parsing C Files")
        print("="*70)
        
        # If no files specified, use files from compilation database
        if c_files is None:
            if self.comp_db.get_all_files():
                c_files = self.comp_db.get_all_files()
                print(f"Using {len(c_files)} files from compile_commands.json")
            else:
                # Fallback to common files
                c_files = [
                    'reference_data.c',
                    'claim_validation.c',
                    'network_verification.c',
                    'cos_calculation.c',
                    'audit.c',
                    'main.c'
                ]
                print(f"Using default file list: {len(c_files)} files")
        
        print()
        
        # Parse each file
        for c_file in c_files:
            tu = self.parse_file(c_file)
            
            if tu is None:
                continue
            
            # Find all function definitions
            for node in tu.cursor.walk_preorder():
                if node.kind == clang.CursorKind.FUNCTION_DECL:
                    # Only analyze function definitions (not declarations)
                    if node.is_definition():
                        func_info = self.analyze_function(node)
                        self.functions.append(func_info)
                        
                        # Print interesting findings (flags)
                        if func_info.sets_flags or func_info.checks_flags:
                            print(f"  🚩 {func_info.name}:")
                            if func_info.sets_flags:
                                print(f"     Sets: {', '.join(func_info.sets_flags)}")
                            if func_info.checks_flags:
                                print(f"     Checks: {', '.join(func_info.checks_flags)}")
        
        print("\n" + "="*70)
        print(f"✓ Analysis Complete")
        print(f"  Found {len(self.functions)} functions")
        print(f"  Found {len(self.all_flags)} unique flags")
        print("="*70)
        print()
        
        return self.functions
    
    def identify_business_logic_functions(self) -> List[FunctionInfo]:
        """Identify functions likely to contain business rules."""
        business_keywords = [
            'validate', 'verify', 'check', 'calculate', 'apply',
            'audit', 'assess', 'review', 'process', 'determine'
        ]
        
        business_functions = []
        
        for func in self.functions:
            # Check if function name contains business keywords
            name_lower = func.name.lower()
            has_business_keyword = any(keyword in name_lower for keyword in business_keywords)
            
            # Check if function has complexity indicators
            has_complexity = func.has_conditionals >= 2 or func.complexity >= 5
            
            # Check if function works with flags (business state)
            works_with_flags = len(func.sets_flags) > 0 or len(func.checks_flags) > 0
            
            if has_business_keyword or has_complexity or works_with_flags:
                business_functions.append(func)
        
        return business_functions
    
    def find_related_functions(self, target_flag: str) -> Dict[str, List[FunctionInfo]]:
        """Find all functions that set or check a specific flag."""
        setters = []
        checkers = []
        
        for func in self.functions:
            if target_flag in func.sets_flags:
                setters.append(func)
            if target_flag in func.checks_flags:
                checkers.append(func)
        
        return {
            'setters': setters,
            'checkers': checkers
        }
    
    def export_to_json(self, filename: str = "claims_analysis.json"):
        """Export analysis results to JSON."""
        data = {
            'metadata': {
                'source_dir': str(self.source_dir),
                'total_functions': len(self.functions),
                'total_flags': len(self.all_flags),
                'used_compilation_database': len(self.comp_db.commands) > 0
            },
            'functions': [asdict(f) for f in self.functions],
            'flags': list(self.all_flags),
            'business_functions': [asdict(f) for f in self.identify_business_logic_functions()],
            'compilation_database_files': list(self.comp_db.commands.keys())
        }
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Exported results to: {filename}")
        return data


def main():
    """Main analysis function."""
    import sys
    
    # Get source directory from command line or use default
    source_dir = sys.argv[1] if len(sys.argv) > 1 else "../claims_system"
    
    print("\n" + "="*70)
    print("Claims System Business Rule Analyzer")
    print("WITH compile_commands.json SUPPORT")
    print("="*70)
    
    # Parse the claims system
    parser = ClaimsSystemParser(source_dir)
    
    # Parse all files (uses compile_commands.json if available)
    functions = parser.parse_all_files()
    
    # Identify business logic functions
    print("\n🎯 Business Logic Functions:")
    print("="*70)
    business_funcs = parser.identify_business_logic_functions()
    for func in business_funcs:
        print(f"  • {func.name} ({Path(func.file).name}:{func.line})")
        print(f"    Conditionals: {func.has_conditionals}, Complexity: {func.complexity}")
    print()
    
    # Analyze FLAG_EMERGENCY flow
    print("🔍 Tracing FLAG_EMERGENCY:")
    print("="*70)
    emergency_flow = parser.find_related_functions('FLAG_EMERGENCY')
    
    print("  Functions that SET FLAG_EMERGENCY:")
    for func in emergency_flow['setters']:
        print(f"    • {func.name} ({Path(func.file).name}:{func.line})")
    
    print("\n  Functions that CHECK FLAG_EMERGENCY:")
    for func in emergency_flow['checkers']:
        print(f"    • {func.name} ({Path(func.file).name}:{func.line})")
    print()
    
    # Export results
    parser.export_to_json('output/claims_analysis.json')
    
    print("\n✅ Analysis complete!")
    print("\n💡 Key Takeaway:")
    print("   Used compile_commands.json for accurate parsing!")


if __name__ == '__main__':
    main()
