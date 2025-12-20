#!/usr/bin/env python3
"""
Quick reference utility for looking up music theory rules.

Usage:
    python rule_reference.py "parallel fifths"
    python rule_reference.py --context jazz
    python rule_reference.py --severity strict
    python rule_reference.py --species first
"""

import sys
import argparse
from typing import Dict, Any

from penta_core.teachers import (
    VoiceLeadingRules,
    HarmonyRules,
    CounterpointRules,
    RuleSeverity,
    Species
)


def search_rules(query: str) -> Dict[str, Any]:
    """Search for rules matching the query string."""
    results = {}
    query_lower = query.lower()
    
    # Search voice leading rules
    vl_rules = VoiceLeadingRules.get_all_rules()
    for category, rules in vl_rules.items():
        for rule_name, rule_data in rules.items():
            if (query_lower in rule_name.lower() or 
                query_lower in rule_data.get('name', '').lower() or
                query_lower in rule_data.get('description', '').lower()):
                results[f"voice_leading.{category}.{rule_name}"] = rule_data
    
    # Search harmony rules
    harm_rules = HarmonyRules.get_all_rules()
    for category, rules in harm_rules.items():
        for rule_name, rule_data in rules.items():
            rule_str = str(rule_data).lower()
            if query_lower in rule_name.lower() or query_lower in rule_str:
                results[f"harmony.{category}.{rule_name}"] = rule_data
    
    # Search counterpoint rules
    cp_rules = CounterpointRules.get_all_rules()
    for category, rules in cp_rules.items():
        for rule_name, rule_data in rules.items():
            if (query_lower in rule_name.lower() or
                query_lower in rule_data.get('name', '').lower() or
                query_lower in rule_data.get('description', '').lower()):
                results[f"counterpoint.{category}.{rule_name}"] = rule_data
    
    return results


def format_rule(rule_path: str, rule_data: Dict[str, Any]) -> str:
    """Format a rule for display."""
    output = []
    output.append("=" * 70)
    output.append(f"RULE: {rule_path}")
    output.append("=" * 70)
    
    if isinstance(rule_data, dict):
        if 'name' in rule_data:
            output.append(f"\nName: {rule_data['name']}")
        
        if 'description' in rule_data:
            output.append(f"Description: {rule_data['description']}")
        
        if 'severity' in rule_data:
            severity = rule_data['severity']
            if hasattr(severity, 'value'):
                output.append(f"Severity: {severity.value.upper()}")
            else:
                output.append(f"Severity: {severity}")
        
        if 'context' in rule_data:
            output.append(f"Context: {rule_data['context']}")
        
        if 'reason' in rule_data:
            output.append(f"\nReason: {rule_data['reason']}")
        
        if 'exception' in rule_data:
            output.append(f"\nExceptions: {rule_data['exception']}")
        
        if 'example_violation' in rule_data:
            output.append(f"\nViolation Example:")
            for key, value in rule_data['example_violation'].items():
                output.append(f"  {key}: {value}")
        
        if 'example_correct' in rule_data:
            output.append(f"\nCorrect Example:")
            for key, value in rule_data['example_correct'].items():
                output.append(f"  {key}: {value}")
        
        # Add any other relevant fields
        skip_keys = {'name', 'description', 'severity', 'context', 'reason', 
                     'exception', 'example_violation', 'example_correct'}
        for key, value in rule_data.items():
            if key not in skip_keys and not key.startswith('_'):
                if isinstance(value, (str, int, float, bool)):
                    output.append(f"{key.replace('_', ' ').title()}: {value}")
    
    output.append("")
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Quick reference for music theory rules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "parallel fifths"
  %(prog)s "leading tone"
  %(prog)s --context jazz
  %(prog)s --severity strict
  %(prog)s --species second
  %(prog)s --list-all
        """
    )
    
    parser.add_argument('query', nargs='?', help='Search query for rules')
    parser.add_argument('--context', choices=['classical', 'jazz', 'contemporary', 'all'],
                       help='Filter by musical context')
    parser.add_argument('--severity', 
                       choices=['strict', 'high', 'medium', 'low', 'guideline'],
                       help='Filter by minimum severity')
    parser.add_argument('--species', 
                       choices=['first', 'second', 'third', 'fourth', 'fifth'],
                       help='Show specific species counterpoint rules')
    parser.add_argument('--list-all', action='store_true',
                       help='List all rule categories')
    
    args = parser.parse_args()
    
    # List all categories
    if args.list_all:
        print("\n" + "=" * 70)
        print("MUSIC THEORY RULE CATEGORIES")
        print("=" * 70)
        
        print("\nVOICE LEADING RULES:")
        vl_rules = VoiceLeadingRules.get_all_rules()
        for category in vl_rules.keys():
            print(f"  • {category.replace('_', ' ').title()}")
        
        print("\nHARMONY RULES:")
        harm_rules = HarmonyRules.get_all_rules()
        for category in harm_rules.keys():
            print(f"  • {category.replace('_', ' ').title()}")
        
        print("\nCOUNTERPOINT RULES:")
        cp_rules = CounterpointRules.get_all_rules()
        for category in cp_rules.keys():
            print(f"  • {category.replace('_', ' ').title()}")
        
        print("\nUse --context, --severity, or --species to filter rules")
        print("Use a search query to find specific rules\n")
        return
    
    # Filter by context
    if args.context:
        print(f"\nRules for {args.context.upper()} context:\n")
        rules = VoiceLeadingRules.get_rules_by_context(args.context)
        
        for category, category_rules in rules.items():
            if category_rules:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rule_name, rule_data in category_rules.items():
                    print(f"  • {rule_data.get('name', rule_name)}")
        return
    
    # Filter by severity
    if args.severity:
        severity_map = {
            'strict': RuleSeverity.STRICT,
            'high': RuleSeverity.HIGH,
            'medium': RuleSeverity.MEDIUM,
            'low': RuleSeverity.LOW,
            'guideline': RuleSeverity.GUIDELINE,
        }
        
        print(f"\nRules with {args.severity.upper()} severity or higher:\n")
        rules = VoiceLeadingRules.get_rules_by_severity(severity_map[args.severity])
        
        for category, category_rules in rules.items():
            if category_rules:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rule_name, rule_data in category_rules.items():
                    severity = rule_data.get('severity', '')
                    if hasattr(severity, 'value'):
                        severity = severity.value
                    print(f"  • {rule_data.get('name', rule_name)} [{severity}]")
        return
    
    # Show species counterpoint rules
    if args.species:
        species_map = {
            'first': Species.FIRST,
            'second': Species.SECOND,
            'third': Species.THIRD,
            'fourth': Species.FOURTH,
            'fifth': Species.FIFTH,
        }
        
        print(f"\n{args.species.upper()} SPECIES COUNTERPOINT RULES")
        print("=" * 70 + "\n")
        
        rules = CounterpointRules.get_species_rules(species_map[args.species])
        
        for rule_name, rule_data in rules.items():
            print(format_rule(f"counterpoint.{args.species}_species.{rule_name}", rule_data))
        return
    
    # Search for rules
    if args.query:
        results = search_rules(args.query)
        
        if not results:
            print(f"\nNo rules found matching: '{args.query}'")
            print("Try a different search term or use --list-all to see categories\n")
            return
        
        print(f"\nFound {len(results)} rule(s) matching '{args.query}':\n")
        
        for rule_path, rule_data in results.items():
            print(format_rule(rule_path, rule_data))
        
        return
    
    # No arguments - show usage
    parser.print_help()


if __name__ == "__main__":
    main()
