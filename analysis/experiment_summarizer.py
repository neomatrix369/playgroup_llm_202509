"""
Experiment summarization module for analyzing past experiment results.

Handles the summarization mode which:
- Discovers existing experiments
- Aggregates data across runs
- Displays summary insights
- Generates persistent summary files

Following SOLID principles:
- Single Responsibility: Only handles summarization mode operations
- Dependency Inversion: Depends on abstractions (aggregator, output generator)

Extracted from BatchExperimentRunner to separate operational modes.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

from analysis.experiment_aggregator import ExperimentAggregator


class ExperimentSummarizer:
    """Handles experiment summarization mode operations.
    
    Centralizes the complete summarization workflow that was in
    BatchExperimentRunner, enabling independent testing and cleaner
    separation between running experiments and analyzing past results.
    """
    
    def __init__(
        self,
        aggregator: ExperimentAggregator,
        output_generator_callback: Callable[[Path, dict], list],
        log_callback: Callable[[str], None] = None
    ):
        """Initialize experiment summarizer.
        
        Args:
            aggregator: ExperimentAggregator instance for discovering/aggregating data
            output_generator_callback: Callback to generate persistent summary files
            log_callback: Optional callback for logging with timestamps
        """
        self.aggregator = aggregator
        self.generate_persistent_summary = output_generator_callback
        self.log = log_callback if log_callback else lambda msg: print(msg)
    
    def run(self, args: argparse.Namespace) -> None:
        """Run the complete summarization workflow.
        
        Args:
            args: Command line arguments containing:
                - output_dir: Directory to search for experiments
                - verbose: Whether to show detailed output
        """
        print("\n" + "=" * 80)
        print("📊  EXPERIMENT SUMMARIZATION MODE  📊")
        print("=" * 80)
        
        base_output_dir = Path(args.output_dir)
        self.log(f"🔍 Scanning for existing experiments in {base_output_dir}")
        
        # Discover existing experiments
        experiments = self.aggregator.discover_experiments(base_output_dir)
        
        if not experiments:
            print(f"❌ No experiment results found in {base_output_dir}")
            print("   💡 Run some experiments first with: python run_all_problems.py")
            return
        
        print(f"✅ Found {len(experiments)} experiment run(s)")
        
        if args.verbose:
            for i, exp in enumerate(experiments[:5], 1):
                date_str = datetime.fromtimestamp(exp['date']).strftime('%Y-%m-%d %H:%M')
                print(f"   {i}. {exp['timestamp']} ({date_str})")
            if len(experiments) > 5:
                print(f"   ... and {len(experiments) - 5} more")
        
        self.log("📈 Aggregating experiment data...")
        
        # Aggregate all experiment data
        aggregated_data = self.aggregator.aggregate(experiments)
        
        if not aggregated_data:
            print("❌ No valid experiment data could be loaded")
            return
        
        # Display summary insights
        self._display_insights(aggregated_data)
        
        # Generate persistent summary files
        self.log("💾 Generating persistent summary files...")
        generated_files = self.generate_persistent_summary(base_output_dir, aggregated_data)
        
        # Display generated files
        self._display_generated_files(base_output_dir, generated_files)
        
        self.log("✅ Experiment summarization completed!")
        self._display_usage_tips()
    
    def _display_insights(self, aggregated_data: dict) -> None:
        """Display aggregated insights to console.
        
        Args:
            aggregated_data: Aggregated experiment data dictionary
        """
        print(f"\n🏆 " + "=" * 70)
        print(f"📊 AGGREGATED INSIGHTS FROM ALL EXPERIMENTS")
        print("=" * 75)
        
        # Best template insights
        if aggregated_data.get('analysis', {}).get('template_ranking'):
            best_template = aggregated_data['analysis']['template_ranking'][0]
            print(f"🥇 Best Overall Template: {best_template['template']}")
            print(f"   📊 Average success: {best_template['avg_all_correct_rate']:.1%}")
            print(f"   🎯 Tested on {best_template['total_problems']} problems")
        
        # Template statistics
        if aggregated_data.get('template_stats'):
            print(f"\n📝 Template Statistics:")
            template_count = len(aggregated_data['template_stats'])
            print(f"   🔢 Total templates tested: {template_count}")
            
            most_tested = max(aggregated_data['template_stats'].items(),
                            key=lambda x: x[1]['experiments'])
            print(f"   🧪 Most tested: {most_tested[0][:40]:40s} ({most_tested[1]['experiments']} times)")
        
        # Problem statistics
        if aggregated_data.get('problem_stats'):
            print(f"\n🎯 Problem Statistics:")
            problem_count = len(aggregated_data['problem_stats'])
            print(f"   🔢 Total problems tested: {problem_count}")
            
            hardest_problem = min(aggregated_data['problem_stats'].items(),
                                key=lambda x: x[1]['avg_success_rate'])
            print(f"   🔴 Hardest problem: {hardest_problem[0]} ({hardest_problem[1]['avg_success_rate']:.1%} avg success)")
        
        print("=" * 75)
    
    def _display_generated_files(self, base_output_dir: Path, generated_files: list) -> None:
        """Display list of generated summary files.
        
        Args:
            base_output_dir: Base output directory
            generated_files: List of generated file paths
        """
        print(f"\n💾 " + "=" * 60)
        print(f"📁 SUMMARY FILES GENERATED")
        print("=" * 65)
        print(f"    📂 Directory: {base_output_dir}")
        for file_path in generated_files:
            file_name = Path(file_path).name
            print(f"    📄 {file_name}")
        print("=" * 65)
    
    def _display_usage_tips(self) -> None:
        """Display usage tips for the generated summary files."""
        print("\n💡 Use these files to:")
        print("   📊 Track template performance over time")
        print("   🎯 Identify problem difficulty patterns")
        print("   🔍 Find optimal template selections")
        print("   📈 Monitor experiment trends")
