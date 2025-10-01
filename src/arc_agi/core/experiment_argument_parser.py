#!/usr/bin/env python3
"""
Argument parser for batch experiment runner.

Extracted from BatchExperimentRunner to follow Single Responsibility Principle.
Handles all command-line argument parsing and validation.
"""

import argparse


class ExperimentArgumentParser:
    """Handles command-line argument parsing for batch experiments."""

    @staticmethod
    def parse() -> argparse.Namespace:
        """Parse command line arguments with comprehensive options."""
        parser = argparse.ArgumentParser(
            description="Comprehensive batch experiment runner for ARC-AGI testing",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run all templates and problems
  python run_all_problems.py

  # Run specific templates and problems with 5 iterations
  python run_all_problems.py -t "baseline_justjson_enhanced.j2,reflexion_enhanced.j2" -p "0d3d703e,08ed6ac7" -i 5

  # Dry run with verbose output
  python run_all_problems.py --dry-run --verbose

  # Use different method module
  python run_all_problems.py --method arc_agi.methods.method2_reflexion --model openrouter/deepseek/deepseek-chat-v3-0324

  # Analyze existing experiments and generate summary statistics
  python run_all_problems.py --summarise-experiments --verbose

  # Summarize experiments from custom output directory
  python run_all_problems.py --summarise-experiments -o my_custom_results
            """,
        )

        # Core experiment parameters
        parser.add_argument(
            "-m",
            "--method",
            type=str,
            default="arc_agi.methods.method1_text_prompt",
            help="Method module to use (default: %(default)s)",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="openrouter/deepseek/deepseek-chat-v3-0324",
            help="Model name to use (default: %(default)s)",
        )
        parser.add_argument(
            "-i",
            "--iterations",
            type=int,
            default=1,
            help="Number of iterations per test (default: %(default)s)",
        )

        # Template and problem selection
        parser.add_argument(
            "-t",
            "--templates",
            type=str,
            default="baseline_justjson_enhanced.j2,baseline_wquotedgridcsv_excel_enhanced.j2,baseline_wplaingrid_enhanced.j2,reflexion_enhanced.j2",
            help="Comma-separated list of template names or indices",
        )
        parser.add_argument(
            "-p",
            "--problems",
            type=str,
            help="Comma-separated list of problem IDs or indices",
        )

        # Output and behavior
        parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            default="batch_results",
            help="Output directory (default: %(default)s)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be run without executing",
        )
        parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
        parser.add_argument(
            "--summarise-experiments",
            action="store_true",
            help="Analyze existing experiment results and generate/update summary statistics",
        )
        parser.add_argument(
            "--fail-fast",
            action="store_true",
            help="Stop execution on first error (default: continue through all experiments)",
        )

        # Checkpoint and resume
        parser.add_argument(
            "--no-checkpoint",
            action="store_true",
            help="Disable automatic checkpoint saving (checkpoints enabled by default)",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Force resume from last checkpoint (auto-prompts by default)",
        )
        parser.add_argument(
            "--no-resume",
            action="store_true",
            help="Start fresh, ignoring any existing checkpoints",
        )

        return parser.parse_args()
