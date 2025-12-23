#!/usr/bin/env python3
"""
Master Merger Execution Script

Executes the complete merger process with progress tracking.

Steps:
1. Run deduplication analysis
2. Create monorepo structure
3. Migrate modules with history
4. Standardize names
5. Validate migration
6. Generate reports
7. Create deprecation PR for DAiW

Usage:
    python execute_merger.py --source /path/to/daiw --target /path/to/kelly
    python execute_merger.py --dry-run --skip-validation
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


class MergerExecutor:
    """Execute the complete repository merger process."""
    
    def __init__(self, source_repo: str, target_repo: str, dry_run: bool = False):
        self.source_repo = Path(source_repo)
        self.target_repo = Path(target_repo)
        self.dry_run = dry_run
        self.scripts_dir = self.target_repo / 'tools' / 'scripts'
        self.reports_dir = self.target_repo / 'migration_reports'
        self.reports_dir.mkdir(exist_ok=True)
        
        self.steps = [
            ('Deduplication Analysis', self.run_deduplication),
            ('Create Monorepo Structure', self.create_monorepo),
            ('Migrate Modules', self.migrate_modules),
            ('Standardize Names', self.standardize_names),
            ('Validate Migration', self.validate_migration),
            ('Generate Final Report', self.generate_final_report),
        ]
        
        self.step_results = []
    
    def run_script(self, script_name: str, args: List[str]) -> bool:
        """Run a migration script."""
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            print(f"  ❌ Script not found: {script_name}")
            return False
        
        cmd = ['python', str(script_path)] + args
        
        if self.dry_run:
            print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
            return True
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ❌ Failed with exit code {result.returncode}")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
        
        return True
    
    def run_deduplication(self) -> bool:
        """Step 1: Run deduplication analysis."""
        print("\n" + "="*60)
        print("STEP 1: Deduplication Analysis")
        print("="*60)
        
        report_file = self.reports_dir / f'deduplication_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        
        args = [
            '--scan', str(self.target_repo),
            '--output', str(report_file),
            '--format', 'markdown'
        ]
        
        success = self.run_script('deduplicate.py', args)
        
        if success:
            print(f"  ✅ Deduplication report: {report_file}")
        
        return success
    
    def create_monorepo(self) -> bool:
        """Step 2: Create monorepo structure."""
        print("\n" + "="*60)
        print("STEP 2: Create Monorepo Structure")
        print("="*60)
        
        args = ['--output', str(self.target_repo)]
        
        if self.dry_run:
            args.append('--dry-run')
        
        return self.run_script('create_monorepo.py', args)
    
    def migrate_modules(self) -> bool:
        """Step 3: Migrate modules with git history."""
        print("\n" + "="*60)
        print("STEP 3: Migrate Modules")
        print("="*60)
        
        report_file = self.reports_dir / f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        
        args = [
            '--source', str(self.source_repo),
            '--target', str(self.target_repo),
            '--module', 'all',
            '--report', str(report_file)
        ]
        
        if self.dry_run:
            args.append('--dry-run')
        
        success = self.run_script('migrate_modules.py', args)
        
        if success:
            print(f"  ✅ Migration report: {report_file}")
        
        return success
    
    def standardize_names(self) -> bool:
        """Step 4: Standardize naming conventions."""
        print("\n" + "="*60)
        print("STEP 4: Standardize Names")
        print("="*60)
        
        args = [
            '--scan', str(self.target_repo),
            '--fix'
        ]
        
        if self.dry_run:
            args.append('--dry-run')
        
        return self.run_script('standardize_names.py', args)
    
    def validate_migration(self) -> bool:
        """Step 5: Validate the migration."""
        print("\n" + "="*60)
        print("STEP 5: Validate Migration")
        print("="*60)
        
        report_file = self.reports_dir / f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        
        args = [
            '--repo', str(self.target_repo),
            '--test', 'all',
            '--report', str(report_file)
        ]
        
        success = self.run_script('validate_migration.py', args)
        
        if success:
            print(f"  ✅ Validation report: {report_file}")
        
        return success
    
    def generate_final_report(self) -> bool:
        """Step 6: Generate final merger report."""
        print("\n" + "="*60)
        print("STEP 6: Generate Final Report")
        print("="*60)
        
        report_file = self.reports_dir / f'merger_complete_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        
        lines = [
            "# Kelly Music Brain 2.0 Merger Report",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Source Repository:** {self.source_repo}",
            f"**Target Repository:** {self.target_repo}",
            "",
            "## Execution Summary",
            ""
        ]
        
        for step_name, step_result in self.step_results:
            status = "✅" if step_result else "❌"
            lines.append(f"- {status} {step_name}")
        
        lines.extend([
            "",
            "## Generated Reports",
            ""
        ])
        
        for report in sorted(self.reports_dir.glob('*.md')):
            lines.append(f"- [{report.name}]({report})")
        
        lines.extend([
            "",
            "## Next Steps",
            "",
            "1. Review all generated reports",
            "2. Resolve any merge conflicts",
            "3. Run full test suite: `pytest tests/`",
            "4. Build C++ components: `cmake -B build && cmake --build build`",
            "5. Update documentation",
            "6. Create deprecation PR for DAiW repository",
            "7. Announce migration to users",
            ""
        ])
        
        report_content = '\n'.join(lines)
        
        if not self.dry_run:
            report_file.write_text(report_content, encoding='utf-8')
            print(f"  ✅ Final report: {report_file}")
        else:
            print(f"  [DRY RUN] Would create: {report_file}")
        
        return True
    
    def execute(self, skip_validation: bool = False):
        """Execute all merger steps."""
        print("\n" + "="*70)
        print("KELLY MUSIC BRAIN 2.0 - REPOSITORY MERGER EXECUTION")
        print("="*70)
        print(f"Source: {self.source_repo}")
        print(f"Target: {self.target_repo}")
        print(f"Dry Run: {self.dry_run}")
        print("="*70)
        
        steps_to_run = self.steps
        if skip_validation:
            steps_to_run = [s for s in steps_to_run if 'Validate' not in s[0]]
        
        for step_name, step_func in steps_to_run:
            result = step_func()
            self.step_results.append((step_name, result))
            
            if not result and not self.dry_run:
                print(f"\n❌ Step failed: {step_name}")
                print("Aborting merger execution.")
                return False
        
        print("\n" + "="*70)
        if all(r for _, r in self.step_results):
            print("✅ MERGER EXECUTION COMPLETE!")
        else:
            print("⚠️  MERGER EXECUTION COMPLETED WITH WARNINGS")
        print("="*70)
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Execute the complete DAiW → Kelly 2.0 merger process"
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to DAiW source repository'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='.',
        help='Path to Kelly target repository (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation step'
    )
    
    args = parser.parse_args()
    
    executor = MergerExecutor(args.source, args.target, dry_run=args.dry_run)
    success = executor.execute(skip_validation=args.skip_validation)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
