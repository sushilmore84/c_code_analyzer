#!/usr/bin/env python3
"""
Graph Builder for Business Rule Extraction
Builds: Call Graph, Data Flow Graph, Control Flow Graph
Output: Visual diagrams + structured data for LLM
"""

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from collections import defaultdict


@dataclass
class BusinessRule:
    """Structured business rule extracted from graphs."""
    rule_name: str
    rule_type: str  # "calculation", "validation", "audit", "decision"
    conditions: List[str]
    actions: List[str]
    code_locations: List[str]
    dependencies: List[str]
    execution_flow: List[str]
    involved_functions: List[str]
    data_fields_used: List[str]
    flags_involved: List[str]


class GraphBuilder:
    """Build various graphs from enhanced AST analysis."""
    
    def __init__(self, analysis_data: dict):
        """Initialize with output from EnhancedClaimsParser."""
        self.functions = analysis_data['functions']
        self.flag_flows = analysis_data.get('flag_flows', {})
        self.metadata = analysis_data.get('metadata', {})
        
        # Initialize graphs
        self.call_graph = nx.DiGraph()
        self.data_flow_graph = nx.DiGraph()
        self.control_flow_graph = nx.DiGraph()
        self.flag_flow_graph = nx.DiGraph()
        
        # Build all graphs
        self.build_all_graphs()
    
    def build_call_graph(self):
        """Build function call graph."""
        print("\n🔗 Building Call Graph...")
        
        for func in self.functions:
            func_name = func['name']
            
            # Add node with metadata
            self.call_graph.add_node(
                func_name,
                file=Path(func['file']).name,
                line=func['line'],
                is_business=func.get('is_business_logic', False),
                complexity=func.get('complexity', 0),
                has_conditions=func.get('has_conditionals', 0)
            )
            
            # Add edges for function calls
            for called_func in func['calls_functions']:
                # Filter out standard library functions
                if called_func not in ['printf', 'strcpy', 'strcmp', 'malloc', 'free']:
                    self.call_graph.add_edge(
                        func_name, 
                        called_func,
                        relationship='calls'
                    )
        
        print(f"  ✓ {self.call_graph.number_of_nodes()} nodes")
        print(f"  ✓ {self.call_graph.number_of_edges()} edges")
    
    def build_data_flow_graph(self):
        """Build data flow graph showing how data moves through functions."""
        print("\n📊 Building Data Flow Graph...")
        
        for func in self.functions:
            func_name = func['name']
            
            # Add function as node
            self.data_flow_graph.add_node(
                func_name,
                type='function'
            )
            
            # Add data fields as nodes and connect them
            for field_read in func.get('data_fields_read', []):
                field_name = field_read['field_name']
                
                # Add field node if not exists
                if field_name not in self.data_flow_graph:
                    self.data_flow_graph.add_node(
                        field_name,
                        type='data_field'
                    )
                
                # Field flows INTO function (function reads it)
                self.data_flow_graph.add_edge(
                    field_name,
                    func_name,
                    relationship='reads',
                    line=field_read['line']
                )
            
            for field_write in func.get('data_fields_written', []):
                field_name = field_write['field_name']
                
                # Add field node if not exists
                if field_name not in self.data_flow_graph:
                    self.data_flow_graph.add_node(
                        field_name,
                        type='data_field'
                    )
                
                # Function flows INTO field (function writes it)
                self.data_flow_graph.add_edge(
                    func_name,
                    field_name,
                    relationship='writes',
                    line=field_write['line']
                )
        
        print(f"  ✓ {self.data_flow_graph.number_of_nodes()} nodes")
        print(f"  ✓ {self.data_flow_graph.number_of_edges()} edges")
    
    def build_flag_flow_graph(self):
        """Build flag flow graph showing how flags propagate."""
        print("\n🚩 Building Flag Flow Graph...")
        
        for flag_name, flow in self.flag_flows.items():
            # Add flag as central node
            self.flag_flow_graph.add_node(
                flag_name,
                type='flag'
            )
            
            # Connect setters to flag
            for setter in flow['set_by']:
                self.flag_flow_graph.add_node(setter, type='function')
                self.flag_flow_graph.add_edge(
                    setter,
                    flag_name,
                    relationship='sets'
                )
            
            # Connect flag to checkers
            for checker in flow['checked_by']:
                self.flag_flow_graph.add_node(checker, type='function')
                self.flag_flow_graph.add_edge(
                    flag_name,
                    checker,
                    relationship='checked_by'
                )
        
        print(f"  ✓ {self.flag_flow_graph.number_of_nodes()} nodes")
        print(f"  ✓ {self.flag_flow_graph.number_of_edges()} edges")
    
    def build_control_flow_graph(self):
        """Build control flow graph showing execution paths."""
        print("\n⚡ Building Control Flow Graph...")
        
        # For each function with conditions, create control flow
        for func in self.functions:
            if func.get('has_conditionals', 0) > 0:
                func_name = func['name']
                
                # Add function entry
                entry_node = f"{func_name}:entry"
                self.control_flow_graph.add_node(entry_node, type='entry')
                
                # Add conditions as decision nodes
                for i, condition in enumerate(func.get('conditions', [])):
                    cond_node = f"{func_name}:condition_{i+1}"
                    self.control_flow_graph.add_node(
                        cond_node,
                        type='condition',
                        expression=condition['expression'],
                        line=condition['line']
                    )
                    
                    # Connect entry to first condition, or previous condition to next
                    if i == 0:
                        self.control_flow_graph.add_edge(entry_node, cond_node)
                    else:
                        prev_cond = f"{func_name}:condition_{i}"
                        self.control_flow_graph.add_edge(prev_cond, cond_node, label='next')
                
                # Add return nodes
                for i, ret in enumerate(func.get('return_statements', [])):
                    ret_node = f"{func_name}:return_{i+1}"
                    self.control_flow_graph.add_node(
                        ret_node,
                        type='return',
                        value=ret['value'],
                        line=ret['line']
                    )
        
        print(f"  ✓ {self.control_flow_graph.number_of_nodes()} nodes")
        print(f"  ✓ {self.control_flow_graph.number_of_edges()} edges")
    
    def build_all_graphs(self):
        """Build all graph types."""
        self.build_call_graph()
        self.build_data_flow_graph()
        self.build_flag_flow_graph()
        self.build_control_flow_graph()
    
    def visualize_call_graph(self, output_file: str = 'output/call_graph.png'):
        """Visualize call graph with business functions highlighted."""
        print(f"\n📈 Generating call graph visualization...")
        
        plt.figure(figsize=(24, 18))
        
        # Use hierarchical layout for better structure
        # Try to arrange nodes in layers based on call depth
        try:
            # Compute node positions using a force-directed algorithm with more space
            pos = nx.spring_layout(self.call_graph, k=3.5, iterations=100, seed=42)
        except:
            pos = nx.spring_layout(self.call_graph, k=2.5, iterations=50)
        
        # Separate business and non-business functions
        business_nodes = [n for n in self.call_graph.nodes() 
                         if self.call_graph.nodes[n].get('is_business', False)]
        other_nodes = [n for n in self.call_graph.nodes() 
                      if not self.call_graph.nodes[n].get('is_business', False)]
        
        # Draw business functions in red (larger, more prominent)
        nx.draw_networkx_nodes(
            self.call_graph, pos,
            nodelist=business_nodes,
            node_color='#e74c3c',
            node_size=2500,
            node_shape='s',
            alpha=0.95,
            linewidths=3,
            edgecolors='#c0392b'
        )
        
        # Draw other functions in blue (smaller)
        nx.draw_networkx_nodes(
            self.call_graph, pos,
            nodelist=other_nodes,
            node_color='#4ecdc4',
            node_size=1500,
            alpha=0.7,
            linewidths=2,
            edgecolors='#3ba89e'
        )
        
        # Draw edges with subtle curves to reduce overlap
        nx.draw_networkx_edges(
            self.call_graph, pos,
            edge_color='#95a5a6',
            arrows=True,
            arrowsize=22,
            width=2,
            alpha=0.5,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1'
        )
        
        # Draw labels with background for better readability
        labels = {node: node for node in self.call_graph.nodes()}
        nx.draw_networkx_labels(
            self.call_graph, pos,
            labels,
            font_size=9,
            font_weight='bold',
            font_color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.75)
        )
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', edgecolor='#c0392b', label='Business Logic Functions'),
            Patch(facecolor='#4ecdc4', edgecolor='#3ba89e', label='Utility Functions')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=14, framealpha=0.9)
        
        plt.title("Call Graph - Function Dependencies", 
                 fontsize=20, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to {output_file}")
        plt.close()
    
    def visualize_flag_flow_graph(self, output_file: str = 'output/flag_flow_graph.png'):
        """Visualize how flags flow through the system."""
        print(f"\n🚩 Generating flag flow visualization...")
        
        if self.flag_flow_graph.number_of_nodes() == 0:
            print("  ⚠️  No flags found")
            return
        
        plt.figure(figsize=(16, 12))
        
        # Position nodes
        pos = nx.spring_layout(self.flag_flow_graph, k=3, iterations=50)
        
        # Separate flags and functions
        flag_nodes = [n for n in self.flag_flow_graph.nodes() 
                     if self.flag_flow_graph.nodes[n].get('type') == 'flag']
        func_nodes = [n for n in self.flag_flow_graph.nodes() 
                     if self.flag_flow_graph.nodes[n].get('type') == 'function']
        
        # Draw flags as large red circles
        nx.draw_networkx_nodes(
            self.flag_flow_graph, pos,
            nodelist=flag_nodes,
            node_color='#e74c3c',
            node_size=3000,
            node_shape='o',
            alpha=0.9
        )
        
        # Draw functions as smaller blue squares
        nx.draw_networkx_nodes(
            self.flag_flow_graph, pos,
            nodelist=func_nodes,
            node_color='#3498db',
            node_size=2000,
            node_shape='s',
            alpha=0.7
        )
        
        # Draw edges with different colors for sets vs checks
        set_edges = [(u, v) for u, v, d in self.flag_flow_graph.edges(data=True)
                    if d.get('relationship') == 'sets']
        check_edges = [(u, v) for u, v, d in self.flag_flow_graph.edges(data=True)
                      if d.get('relationship') == 'checked_by']
        
        set_edge_collection = nx.draw_networkx_edges(
            self.flag_flow_graph, pos,
            edgelist=set_edges,
            edge_color='#27ae60',  # Green for sets
            arrows=True,
            arrowsize=25,
            width=3,
            alpha=0.8,
            label='Sets flag'
        )
        
        check_edge_collection = nx.draw_networkx_edges(
            self.flag_flow_graph, pos,
            edgelist=check_edges,
            edge_color='#f39c12',  # Orange for checks
            arrows=True,
            arrowsize=25,
            width=3,
            alpha=0.8,
            label='Checks flag'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.flag_flow_graph, pos,
            font_size=9,
            font_weight='bold'
        )
        
        # Add legend with patches instead of relying on edge labels
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label='Flags'),
            Patch(facecolor='#3498db', label='Functions'),
            Patch(facecolor='#27ae60', label='Sets flag (green arrows)'),
            Patch(facecolor='#f39c12', label='Checks flag (orange arrows)')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.title("Flag Flow Graph (Flags in Red, Functions in Blue)", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_file}")
        plt.close()
    
    def visualize_data_flow_graph(self, output_file: str = 'output/data_flow_graph.png'):
        """Visualize data flow between functions and data fields - filtered and readable."""
        print(f"\n📊 Generating data flow visualization...")
        
        # Create a filtered graph with only business-relevant nodes
        filtered_graph = nx.DiGraph()
        
        # Only include business logic functions
        business_func_names = [f['name'] for f in self.functions if f.get('is_business_logic', False)]
        
        # Key data fields we care about (filter out noise)
        important_field_patterns = ['claim', 'provider', 'patient', 'flags', 'cos', 'amount', 'status']
        
        # Build filtered graph
        for func_name in business_func_names:
            filtered_graph.add_node(func_name, type='function')
            
            # Add only important data field connections
            for u, v, data in self.data_flow_graph.edges(data=True):
                # Check if this edge involves our business function
                if u == func_name or v == func_name:
                    # Check if the other node (data field) is important
                    other_node = v if u == func_name else u
                    
                    if other_node in business_func_names:
                        # Function to function (keep it)
                        filtered_graph.add_node(other_node, type='function')
                        filtered_graph.add_edge(u, v, **data)
                    elif any(pattern in other_node.lower() for pattern in important_field_patterns):
                        # Important data field
                        filtered_graph.add_node(other_node, type='data_field')
                        filtered_graph.add_edge(u, v, **data)
        
        if filtered_graph.number_of_nodes() == 0:
            print("  ⚠️  No business data flow found")
            return
        
        print(f"  Filtered to {filtered_graph.number_of_nodes()} nodes (from {self.data_flow_graph.number_of_nodes()})")
        
        plt.figure(figsize=(24, 16))
        
        # Use hierarchical layout for better readability
        try:
            # Try to create a hierarchical layout
            pos = nx.spring_layout(filtered_graph, k=3, iterations=100, seed=42)
        except:
            pos = nx.spring_layout(filtered_graph, k=2, iterations=50)
        
        # Separate functions and data fields
        func_nodes = [n for n in filtered_graph.nodes() 
                     if filtered_graph.nodes[n].get('type') == 'function']
        data_nodes = [n for n in filtered_graph.nodes() 
                     if filtered_graph.nodes[n].get('type') == 'data_field']
        
        # Draw functions as blue squares (larger)
        nx.draw_networkx_nodes(
            filtered_graph, pos,
            nodelist=func_nodes,
            node_color='#3498db',
            node_size=2500,
            node_shape='s',
            alpha=0.9,
            linewidths=2,
            edgecolors='#2c3e50'
        )
        
        # Draw data fields as green circles (larger)
        nx.draw_networkx_nodes(
            filtered_graph, pos,
            nodelist=data_nodes,
            node_color='#2ecc71',
            node_size=2000,
            node_shape='o',
            alpha=0.9,
            linewidths=2,
            edgecolors='#27ae60'
        )
        
        # Draw edges with different styles for reads vs writes
        read_edges = [(u, v) for u, v, d in filtered_graph.edges(data=True)
                     if d.get('relationship') == 'reads']
        write_edges = [(u, v) for u, v, d in filtered_graph.edges(data=True)
                      if d.get('relationship') == 'writes']
        
        # Reads: solid purple arrows
        nx.draw_networkx_edges(
            filtered_graph, pos,
            edgelist=read_edges,
            edge_color='#9b59b6',
            arrows=True,
            arrowsize=25,
            width=2.5,
            alpha=0.7,
            style='solid',
            connectionstyle='arc3,rad=0.1'
        )
        
        # Writes: dashed orange arrows
        nx.draw_networkx_edges(
            filtered_graph, pos,
            edgelist=write_edges,
            edge_color='#e67e22',
            arrows=True,
            arrowsize=25,
            width=2.5,
            alpha=0.7,
            style='dashed',
            connectionstyle='arc3,rad=0.1'
        )
        
        # Draw labels with better visibility
        nx.draw_networkx_labels(
            filtered_graph, pos,
            font_size=9,
            font_weight='bold',
            font_color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
        )
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', edgecolor='#2c3e50', label='Functions'),
            Patch(facecolor='#2ecc71', edgecolor='#27ae60', label='Data Fields'),
            Patch(facecolor='#9b59b6', label='Reads (solid)'),
            Patch(facecolor='#e67e22', label='Writes (dashed)')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.title("Data Flow Graph - Business Functions & Key Data Fields", 
                 fontsize=18, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to {output_file}")
        plt.close()
    
    def trace_flag_path(self, flag_name: str) -> Dict:
        """Trace complete path for a specific flag."""
        if flag_name not in self.flag_flows:
            return {}
        
        flow = self.flag_flows[flag_name]
        
        # Build execution path
        path = {
            'flag': flag_name,
            'setters': [],
            'checkers': [],
            'complete_flow': []
        }
        
        # Get setter details
        for setter_name in flow['set_by']:
            setter_func = next((f for f in self.functions if f['name'] == setter_name), None)
            if setter_func:
                path['setters'].append({
                    'function': setter_name,
                    'file': Path(setter_func['file']).name,
                    'line': setter_func['line'],
                    'condition': setter_func.get('flag_set_conditions', {}).get(flag_name, 'unconditional')
                })
        
        # Get checker details
        for checker_name in flow['checked_by']:
            checker_func = next((f for f in self.functions if f['name'] == checker_name), None)
            if checker_func:
                path['checkers'].append({
                    'function': checker_name,
                    'file': Path(checker_func['file']).name,
                    'line': checker_func['line'],
                    'condition': checker_func.get('flag_check_conditions', {}).get(flag_name, 'unconditional')
                })
        
        return path
    
    def extract_actions(self, rule: BusinessRule, checkers: List[Dict]) -> List[str]:
        """Extract explicit actions from checker functions."""
        actions = []
        
        for checker in checkers:
            checker_func = next((f for f in self.functions 
                                if f['name'] == checker['function']), None)
            if not checker_func:
                continue
            
            # Action 1: Function name implies action
            func_name = checker['function']
            if 'apply' in func_name:
                action = f"Apply {func_name.replace('apply_', '').replace('_', ' ')}"
                if action not in actions:
                    actions.append(action)
            elif 'calculate' in func_name:
                action = f"Calculate {func_name.replace('calculate_', '').replace('_', ' ')}"
                if action not in actions:
                    actions.append(action)
            elif 'check' in func_name:
                action = f"Check {func_name.replace('check_', '').replace('_', ' ')}"
                if action not in actions:
                    actions.append(action)
            elif 'verify' in func_name:
                action = f"Verify {func_name.replace('verify_', '').replace('_', ' ')}"
                if action not in actions:
                    actions.append(action)
            elif 'assess' in func_name:
                action = f"Assess {func_name.replace('assess_', '').replace('_', ' ')}"
                if action not in actions:
                    actions.append(action)
            elif 'review' in func_name:
                action = f"Review {func_name.replace('review_', '').replace('_', ' ')}"
                if action not in actions:
                    actions.append(action)
            
            # Action 2: Data field writes
            for write in checker_func.get('data_fields_written', []):
                field_name = write['field_name']
                action = f"Update {field_name}"
                if action not in actions:
                    actions.append(action)
            
            # Action 3: Return value calculations
            for ret in checker_func.get('return_statements', []):
                ret_value = ret['value']
                
                # Look for calculations (multipliers, additions, etc.)
                if '*' in ret_value and any(x in ret_value for x in ['1.5', '2', '0.8']):
                    action = f"Calculate: {ret_value}"
                    if action not in actions:
                        actions.append(action)
                elif 'multiplier' in ret_value.lower():
                    action = "Apply rate multiplier"
                    if action not in actions:
                        actions.append(action)
                elif 'premium' in ret_value.lower():
                    action = "Apply premium adjustment"
                    if action not in actions:
                        actions.append(action)
        
        # If no actions found, add generic action based on flag
        if not actions and len(rule.flags_involved) > 0:
            flag = rule.flags_involved[0]
            action = f"Process {flag.replace('FLAG_', '').replace('_', ' ').lower()}"
            actions.append(action)
        
        return actions
    
    def extract_business_rules(self) -> List[BusinessRule]:
        """Extract business rules by analyzing graph patterns."""
        print("\n💼 Extracting Business Rules from Graphs...")
        
        rules = []
        
        # Extract rules based on flag flows
        for flag_name, flow in self.flag_flows.items():
            # This is a potential business rule
            rule_name = flag_name.replace('FLAG_', '').replace('_', ' ').title()
            
            # Trace the complete path
            path = self.trace_flag_path(flag_name)
            
            if not path:
                continue
            
            # Build rule
            rule = BusinessRule(
                rule_name=f"{rule_name} Rule",
                rule_type="decision",  # Could be enhanced with better detection
                conditions=[],
                actions=[],
                code_locations=[],
                dependencies=[flag_name],
                execution_flow=[],
                involved_functions=[],
                data_fields_used=[],
                flags_involved=[flag_name]
            )
            
            # Extract conditions from setters
            for setter in path['setters']:
                if setter['condition'] != 'unconditional':
                    rule.conditions.append(setter['condition'])
                rule.code_locations.append(f"{setter['file']}:{setter['function']} (line {setter['line']})")
                rule.involved_functions.append(setter['function'])
                rule.execution_flow.append(f"{setter['function']} sets {flag_name}")
                
                # Add action for setting the flag
                flag_action = f"Set {flag_name}"
                if flag_action not in rule.actions:
                    rule.actions.append(flag_action)
            
            # Extract actions from checkers
            for checker in path['checkers']:
                rule.code_locations.append(f"{checker['file']}:{checker['function']} (line {checker['line']})")
                rule.involved_functions.append(checker['function'])
                rule.execution_flow.append(f"{checker['function']} checks {flag_name}")
                
                # Get data fields used by checker
                checker_func = next((f for f in self.functions if f['name'] == checker['function']), None)
                if checker_func:
                    for field in checker_func.get('data_fields_read', []):
                        if field['field_name'] not in rule.data_fields_used:
                            rule.data_fields_used.append(field['field_name'])
            
            # NEW: Extract explicit actions
            rule.actions = self.extract_actions(rule, path['checkers'])
            
            rules.append(rule)
        
        print(f"  ✓ Extracted {len(rules)} business rules")
        return rules
    
    def export_for_llm(self, output_file: str = 'output/graph_analysis_for_llm.json'):
        """Export graph analysis in LLM-friendly format."""
        print(f"\n🤖 Exporting analysis for LLM...")
        
        # Extract business rules
        rules = self.extract_business_rules()
        
        # Prepare structured data for LLM
        llm_data = {
            'summary': {
                'total_functions': len(self.functions),
                'business_functions': sum(1 for f in self.functions if f.get('is_business_logic')),
                'flags_tracked': len(self.flag_flows),
                'rules_extracted': len(rules)
            },
            'graphs': {
                'call_graph': {
                    'nodes': self.call_graph.number_of_nodes(),
                    'edges': self.call_graph.number_of_edges(),
                    'business_functions': [n for n in self.call_graph.nodes() 
                                         if self.call_graph.nodes[n].get('is_business', False)]
                },
                'data_flow_graph': {
                    'nodes': self.data_flow_graph.number_of_nodes(),
                    'edges': self.data_flow_graph.number_of_edges()
                },
                'flag_flow_graph': {
                    'nodes': self.flag_flow_graph.number_of_nodes(),
                    'edges': self.flag_flow_graph.number_of_edges()
                }
            },
            'business_rules': [asdict(rule) for rule in rules],
            'flag_traces': {
                flag_name: self.trace_flag_path(flag_name)
                for flag_name in self.flag_flows.keys()
            }
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(llm_data, f, indent=2)
        
        print(f"  ✓ Exported to {output_file}")
        return llm_data


def main():
    """Main function to demonstrate graph building."""
    import sys
    
    # Load enhanced analysis
    analysis_file = sys.argv[1] if len(sys.argv) > 1 else 'output/enhanced_analysis.json'
    
    print("\n" + "="*70)
    print("Graph Builder for Business Rule Extraction")
    print("="*70)
    
    print(f"\n📂 Loading analysis from: {analysis_file}")
    
    with open(analysis_file) as f:
        analysis_data = json.load(f)
    
    # Build graphs
    builder = GraphBuilder(analysis_data)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)
    
    builder.visualize_call_graph()
    builder.visualize_flag_flow_graph()
    builder.visualize_data_flow_graph()
    
    # Export for LLM
    print("\n" + "="*70)
    print("Preparing Data for LLM")
    print("="*70)
    
    llm_data = builder.export_for_llm()
    
    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"\n✓ Generated 3 graph visualizations")
    print(f"✓ Extracted {len(llm_data['business_rules'])} business rules")
    print(f"✓ Traced {len(llm_data['flag_traces'])} flag flows")
    print(f"\n📁 Output files:")
    print(f"  - output/call_graph.png")
    print(f"  - output/flag_flow_graph.png")
    print(f"  - output/data_flow_graph.png")
    print(f"  - output/graph_analysis_for_llm.json")
    
    print("\n✅ Graph analysis complete!")
    print("\n💡 Next: Feed 'graph_analysis_for_llm.json' to LLM for final rule synthesis")


if __name__ == '__main__':
    main()