#!/usr/bin/env python3
"""
Comprehensive batch experiment runner for ARC-AGI testing.

Merges functionality from run_batch_tests.sh and run_all_problems.py:
- Multiple template iteration with flexible selection
- Multiple problem iteration with flexible selection
- Multiple method module support (method1_text_prompt, method2_reflexion, etc.)
- Comprehensive timing tracking (global, template-level, individual)
- Real-time progress reporting with timestamps
- Multi-format output generation (console, CSV, HTML)
- Detailed success rate analysis and LLM statistics
- Direct Python integration (no subprocess overhead)
"""

import argparse
import importlib
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not available. CSV output will be limited.")
    pd = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available. Environment variables may not be loaded.")
    load_dotenv = lambda: None

import utils

# Refactored classes to eliminate duplication
from analysis.performance_grader import PerformanceGrader
from analysis.difficulty_classifier import DifficultyClassifier
from analysis.statistics_aggregator import (
    TemplateStatisticsAggregator,
    ProblemStatisticsAggregator,
    ExperimentStatisticsAggregator,
    BestTemplateRecommender
)
from analysis.experiment_aggregator import ExperimentAggregator
from analysis.experiment_summarizer import ExperimentSummarizer
from output.report_writer import ReportWriter
from output.iteration_formatter import IterationStatusFormatter
from output.failure_formatter import FailureSummaryFormatter
from output.output_generator import OutputGenerator
from output.console_display import ConsoleDisplay
from core.timing_tracker import TimingTracker
from core.experiment_config import ExperimentConfigResolver
from core.experiment_executor import ExperimentExecutor
from core.experiment_validator import ExperimentValidator
from core.experiment_loop_orchestrator import ExperimentLoopOrchestrator
from core.service_registry import ServiceRegistry, ServiceFactory
from domain.value_objects import SuccessThresholds
from domain.experiment_state import ExperimentResults, ExperimentContext


class BatchExperimentRunner:
    """Main class for running batch experiments with comprehensive tracking."""

    def __init__(self):
        """
        Initialize batch experiment runner.
        
        Following Object Calisthenics Rule 8: Maximum 2 instance variables.
        Reduced from 14 instance variables to 5 by using cohesive objects:
        - _timing: Timing operations
        - _results: All results data (was 4 variables)
        - _context: Execution context (was 2 variables)
        - _services: All services (was 7 variables)
        - _thresholds: Success thresholds
        """
        # Core infrastructure (5 instance variables total)
        self._timing = TimingTracker()
        self._results = ExperimentResults()
        self._context = ExperimentContext()
        self._thresholds = SuccessThresholds()
        
        # Service registry (consolidates 7 lazy-initialized services)
        factory = self._create_service_factory()
        self._services = ServiceRegistry(factory)

    def parse_arguments(self) -> argparse.Namespace:
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
  python run_all_problems.py --method method2_reflexion --model openrouter/deepseek/deepseek-chat-v3-0324

  # Analyze existing experiments and generate summary statistics
  python run_all_problems.py --summarise-experiments --verbose

  # Summarize experiments from custom output directory
  python run_all_problems.py --summarise-experiments -o my_custom_results
            """)

        # Core experiment parameters
        parser.add_argument("-m", "--method", type=str, default="method1_text_prompt",
                          help="Method module to use (default: %(default)s)")
        parser.add_argument("--model", type=str, default="openrouter/deepseek/deepseek-chat-v3-0324",
                          help="Model name to use (default: %(default)s)")
        parser.add_argument("-i", "--iterations", type=int, default=1,
                          help="Number of iterations per test (default: %(default)s)")

        # Template and problem selection
        parser.add_argument("-t", "--templates", type=str, default="baseline_justjson_enhanced.j2,baseline_wquotedgridcsv_excel_enhanced.j2,baseline_wplaingrid_enhanced.j2,reflexion_enhanced.j2",
                          help="Comma-separated list of template names or indices")
        parser.add_argument("-p", "--problems", type=str,
                          help="Comma-separated list of problem IDs or indices")

        # Output and behavior
        parser.add_argument("-o", "--output-dir", type=str, default="batch_results",
                          help="Output directory (default: %(default)s)")
        parser.add_argument("--dry-run", action="store_true",
                          help="Show what would be run without executing")
        parser.add_argument("-v", "--verbose", action="store_true",
                          help="Verbose output")
        parser.add_argument("--summarise-experiments", action="store_true",
                          help="Analyze existing experiment results and generate/update summary statistics")
        parser.add_argument("--fail-fast", action="store_true",
                          help="Stop execution on first error (default: continue through all experiments)")

        return parser.parse_args()

    # Configuration resolution methods moved to ExperimentConfigResolver
    # (Duplication removed in Phase 2 refactoring)

    def log_timestamp(self, message: str, to_file_only: bool = False) -> None:
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"🕒 [{timestamp}] {message}"

        if not to_file_only:
            print(log_msg)

        self._context.write_to_log(log_msg)

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def _create_service_factory(self) -> ServiceFactory:
        """Create service factory with required dependencies."""
        def ranking_callback(results_data):
            # Temporarily use provided results for analysis
            original = self._results.results
            self._results.set_all_results(results_data)
            analysis = self.generate_ranking_analysis()
            self._results.set_all_results(original)
            return analysis
        
        return ServiceFactory(
            timing_tracker=self._timing,
            results_accessor=lambda: self._results.results,
            log_callback=self.log_timestamp,
            format_duration_callback=self.format_duration,
            run_experiment_callback=self.run_single_experiment,
            analyze_results_callback=self.analyze_results,
            generate_ranking_analysis_callback=ranking_callback,
            generate_persistent_summary_callback=self.generate_persistent_summary,
            thresholds=self._thresholds,
            log_file_handle_accessor=lambda: self._context.log_handle
        )

    def validate_prerequisites(self, templates_to_use: List[str], problems_to_use: List[str], method_module: Any, args: argparse.Namespace) -> Tuple[bool, List[str]]:
        """Validate all prerequisites before starting experiments."""
        # Delegate to ExperimentValidator (Phase 2B refactoring)
        return self._services.validator().validate_all(templates_to_use, problems_to_use, method_module, args)

    def setup_experiment_without_argparse(self, experiment_folder: Path) -> Tuple[Any, str]:
        """Setup experiment folder and database without re-parsing arguments."""
        # Delegate to ExperimentExecutor (Phase 2A refactoring)
        return self._services.executor().setup_experiment(experiment_folder)

    def run_single_experiment(
        self,
        template: str,
        problem: str,
        method_module: Any,
        args: argparse.Namespace,
        experiment_folder: Path
    ) -> Tuple[List[Any], List[Any]]:
        """Run a single experiment with timing."""
        # Delegate to ExperimentExecutor (Phase 2A refactoring)
        executor = self._services.executor()
        result = executor.run_experiment(template, problem, method_module, args, experiment_folder)
        # Sync state back to main runner
        self._context.set_attempts(executor.total_experiments_attempted)
        self._results.set_all_failures(executor.failed_experiments)
        return result

    def analyze_results(
        self,
        template: str,
        problem: str,
        rr_trains: List[Any],
        iterations: int,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Analyze experiment results for success rates with real-time feedback."""
        # Delegate to ExperimentExecutor (Phase 2A refactoring)
        return self._services.executor().analyze_results(template, problem, rr_trains, iterations, verbose)

    def generate_console_table(self) -> None:
        """Generate formatted console table."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        self._services.output_generator().generate_console_table()

    def generate_csv_output(self, output_dir: Path) -> str:
        """Generate CSV output file with ranking data."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        return self._services.output_generator().generate_csv_output(output_dir)
    
    def _generate_csv_output_DEPRECATED(self, output_dir: Path) -> str:
        """Generate CSV output file with ranking data."""
        if not self.results_data:
            return ""

        if pd is None:
            print("Warning: pandas not available, skipping CSV output")
            return ""

        # Main results data
        df = pd.DataFrame(self.results_data)

        # Add timing columns
        df['individual_duration_formatted'] = df['individual_duration'].apply(self.format_duration)
        df['problem_duration_formatted'] = df['problem_duration'].apply(self.format_duration)

        # Add ranking information
        analysis = self.generate_ranking_analysis()
        if analysis:
            # Add experiment rank
            experiment_rankings = {f"{r['template']}|{r['problem']}": i+1
                                 for i, r in enumerate(analysis['experiment_ranking'])}
            df['experiment_rank'] = df.apply(lambda row: experiment_rankings.get(f"{row['template']}|{row['problem']}", 0), axis=1)

            # Add template rank
            template_rankings = {r['template']: i+1 for i, r in enumerate(analysis['template_ranking'])}
            df['template_rank'] = df['template'].map(template_rankings)

            # Add problem difficulty
            problem_difficulties = {r['problem']: r['difficulty'] for r in analysis['problem_analysis']}
            df['problem_difficulty'] = df['problem'].map(problem_difficulties)

            # Add grade based on performance (use refactored grader, eliminates duplication #2)
            grader = PerformanceGrader()
            df['performance_grade'] = df['all_correct_rate'].apply(grader.grade)

        # Use the output directory name as timestamp for consistency
        timestamp = output_dir.name
        csv_file = output_dir / f"batch_results_{timestamp}.csv"

        # Sort by experiment rank for better readability
        if 'experiment_rank' in df.columns:
            df = df.sort_values('experiment_rank')

        df.to_csv(csv_file, index=False)

        # Also generate separate ranking files
        if analysis:
            # Template ranking CSV
            template_df = pd.DataFrame(analysis['template_ranking'])
            template_csv = output_dir / f"template_ranking_{timestamp}.csv"
            template_df.to_csv(template_csv, index=False)

            # Problem analysis CSV
            problem_df = pd.DataFrame(analysis['problem_analysis'])
            problem_csv = output_dir / f"problem_analysis_{timestamp}.csv"
            problem_df.to_csv(problem_csv, index=False)

        return str(csv_file)

    def generate_html_output(self, output_dir: Path) -> str:
        """Generate HTML output file."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        return self._services.output_generator().generate_html_output(output_dir)
    
    def _generate_html_output_DEPRECATED(self, output_dir: Path) -> str:
        """Generate HTML output file."""
        if not self.results_data:
            return ""

        # Use the output directory name as timestamp for consistency
        timestamp = output_dir.name
        html_file = output_dir / f"batch_results_{timestamp}.html"

        total_duration = self._timing.get_global_duration()

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ARC-AGI Batch Experiment Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ background-color: #d4edda; }}
        .partial {{ background-color: #fff3cd; }}
        .failed {{ background-color: #f8d7da; }}
        .timing {{ font-family: monospace; font-size: 0.9em; }}
        .summary {{ background-color: #e9ecef; padding: 15px; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>ARC-AGI Batch Experiment Results</h1>

    <div class="summary">
        <h2>Experiment Summary</h2>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Duration:</strong> {self.format_duration(total_duration)}</p>
        <p><strong>Total Tests:</strong> {len(self.results_data)}</p>
        <p><strong>Average Success Rate:</strong> {sum(r['all_correct_rate'] for r in self.results_data) / len(self.results_data):.2%}</p>
    </div>

    <table>
        <tr>
            <th>Template</th>
            <th>Problem</th>
            <th>All Correct Rate</th>
            <th>At Least One Correct</th>
            <th>Test Duration</th>
            <th>Branch Duration</th>
        </tr>
"""

        for result in self.results_data:
            # Determine row class based on success rate
            row_class = ""
            if result['all_correct_rate'] >= 0.8:
                row_class = "success"
            elif result['all_correct_rate'] >= 0.3:
                row_class = "partial"
            else:
                row_class = "failed"

            test_time = self.format_duration(result['individual_duration'])
            branch_time = self.format_duration(result['problem_duration'])

            html_content += f"""
        <tr class="{row_class}">
            <td>{result['template']}</td>
            <td>{result['problem']}</td>
            <td>{result['all_correct_rate']:.2%}</td>
            <td>{result['at_least_one_correct_rate']:.2%}</td>
            <td class="timing">{test_time}</td>
            <td class="timing">{branch_time}</td>
        </tr>"""

        html_content += """
    </table>
</body>
</html>
"""

        html_file.write_text(html_content)
        return str(html_file)

    def generate_ranking_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive ranking and best-performance analysis."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        return self._services.output_generator().generate_ranking_analysis()
    
    def _generate_ranking_analysis_DEPRECATED(self) -> Dict[str, Any]:
        """Generate comprehensive ranking and best-performance analysis."""
        if not self.results_data:
            return {}

        # 1. Experiment ranking (template+problem combinations)
        experiment_ranking = sorted(
            self.results_data,
            key=lambda x: (x['all_correct_rate'], x['at_least_one_correct_rate']),
            reverse=True
        )

        # 2. Template ranking (average performance across all problems)
        # Refactored: Use TemplateStatisticsAggregator (eliminates duplication #6)
        template_aggregator = TemplateStatisticsAggregator(self.results_data)
        template_ranking = template_aggregator.aggregate_to_ranking()

        # 3. Problem difficulty analysis (average performance across all templates)
        # Refactored: Use ProblemStatisticsAggregator (eliminates duplication #7)
        problem_aggregator = ProblemStatisticsAggregator(self.results_data)
        problem_analysis = problem_aggregator.aggregate_to_analysis()

        # 4. Best template recommendations per problem
        # Refactored: Use BestTemplateRecommender (eliminates duplication #8)
        recommender = BestTemplateRecommender(self.results_data)
        best_template_per_problem = recommender.recommend_all()

        return {
            'experiment_ranking': experiment_ranking,
            'template_ranking': template_ranking,
            'problem_analysis': problem_analysis,
            'best_template_per_problem': best_template_per_problem
        }

    def generate_ranking_report(self, output_dir: Path, timestamp: str) -> str:
        """Generate comprehensive ranking and analysis report."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        return self._services.output_generator().generate_ranking_report(output_dir, timestamp)
    
    def _generate_ranking_report_DEPRECATED(self, output_dir: Path, timestamp: str) -> str:
        """Generate comprehensive ranking and analysis report."""
        if not self.results_data:
            return ""

        analysis = self.generate_ranking_analysis()
        if not analysis:
            return ""

        ranking_file = output_dir / f"ranking_analysis_{timestamp}.log"

        with ReportWriter.open(ranking_file) as writer:
            writer.section_header("🏆  COMPREHENSIVE RANKING & PERFORMANCE ANALYSIS  🏆")
            writer.blank_line()

            # 1. Top performing experiments
            writer.subsection_header("🥇 TOP PERFORMING EXPERIMENTS (Template + Problem Combinations):")
            for i, result in enumerate(analysis['experiment_ranking'][:10], 1):
                grade = "🎯" if result['all_correct_rate'] >= 0.8 else "✅" if result['all_correct_rate'] >= 0.6 else "⚠️"
                writer.writeln(f"{grade} #{i:2d}: {result['template'][:35]:35s} + {result['problem']}")
                writer.writeln(f"      📊 {result['all_correct_rate']:6.1%} all correct, {result['at_least_one_correct_rate']:6.1%} partial")
                writer.writeln(f"      ⏱️  {self.format_duration(result['individual_duration'])}\n")

            # 2. Template ranking
            writer.section_separator()
            writer.subsection_header("📝 TEMPLATE RANKING (Overall Performance Across All Problems):")
            for i, template_data in enumerate(analysis['template_ranking'], 1):
                grade = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                writer.writeln(f"{grade} #{i}: {template_data['template']}")
                writer.writeln(f"    📊 Average: {template_data['avg_all_correct_rate']:6.1%} all correct, {template_data['avg_partial_rate']:6.1%} partial")
                writer.writeln(f"    🎯 Excellent problems: {template_data['excellent_problems']}/{template_data['total_problems']} ({template_data['excellent_problems']/template_data['total_problems']:.1%})")
                writer.writeln(f"    ✅ Good problems: {template_data['good_problems']}/{template_data['total_problems']} ({template_data['good_problems']/template_data['total_problems']:.1%})")
                writer.writeln(f"    ⏱️  Average duration: {self.format_duration(template_data['avg_duration'])}")
                writer.writeln(f"    📈 Weighted score: {template_data['score']:.3f}\n")

            # 3. Problem difficulty analysis
            writer.section_separator()
            writer.subsection_header("🎯 PROBLEM DIFFICULTY ANALYSIS:")

            difficulty_groups = {}
            for problem_data in analysis['problem_analysis']:
                difficulty = problem_data['difficulty']
                if difficulty not in difficulty_groups:
                    difficulty_groups[difficulty] = []
                difficulty_groups[difficulty].append(problem_data)

            for difficulty in ['EASY', 'MEDIUM', 'HARD', 'VERY_HARD']:
                if difficulty in difficulty_groups:
                    problems = difficulty_groups[difficulty]
                    writer.writeln(f"🌟 {difficulty} Problems ({len(problems)}):")
                    for problem_data in problems:
                        difficulty_icon = "🟢" if difficulty == "EASY" else "🟡" if difficulty == "MEDIUM" else "🟠" if difficulty == "HARD" else "🔴"
                        writer.writeln(f"  {difficulty_icon} {problem_data['problem']}: {problem_data['avg_all_correct_rate']:6.1%} avg success")
                        writer.writeln(f"     🏆 Best template: {problem_data['best_template'][:40]:40s} ({problem_data['best_score']:6.1%})")
                    writer.blank_line()

            # 4. Best template recommendations
            writer.section_separator()
            writer.subsection_header("🎯 OPTIMAL TEMPLATE RECOMMENDATIONS PER PROBLEM:")
            for problem, recommendation in analysis['best_template_per_problem'].items():
                writer.writeln(f"🎲 Problem: {problem}")
                writer.writeln(f"   🏆 Best: {recommendation['best_template'][:50]:50s} ({recommendation['best_score']:6.1%})")
                if recommendation['alternatives']:
                    writer.writeln("   📋 Alternatives:")
                    for alt in recommendation['alternatives']:
                        writer.writeln(f"      • {alt['template'][:45]:45s} ({alt['all_correct_rate']:6.1%})")
                writer.blank_line()

            # 5. Comparative analysis summary
            writer.section_separator()
            writer.subsection_header("📊 COMPARATIVE ANALYSIS SUMMARY:")

            best_template = analysis['template_ranking'][0]
            worst_template = analysis['template_ranking'][-1]
            easiest_problem = analysis['problem_analysis'][0]
            hardest_problem = analysis['problem_analysis'][-1]

            writer.writeln(f"🏆 Best Overall Template: {best_template['template']}")
            writer.writeln(f"   📊 Average success rate: {best_template['avg_all_correct_rate']:.1%}")
            writer.writeln(f"   🎯 Excellent on {best_template['excellent_problems']}/{best_template['total_problems']} problems\n")

            writer.writeln(f"⚠️  Most Challenging Template: {worst_template['template']}")
            writer.writeln(f"   📊 Average success rate: {worst_template['avg_all_correct_rate']:.1%}\n")

            writer.writeln(f"🟢 Easiest Problem: {easiest_problem['problem']}")
            writer.writeln(f"   📊 Average success rate: {easiest_problem['avg_all_correct_rate']:.1%}")
            writer.writeln(f"   🏆 Best template: {easiest_problem['best_template']}\n")

            writer.writeln(f"🔴 Hardest Problem: {hardest_problem['problem']}")
            writer.writeln(f"   📊 Average success rate: {hardest_problem['avg_all_correct_rate']:.1%}")
            writer.writeln(f"   🏆 Best template: {hardest_problem['best_template']}\n")

            writer.section_separator()

        return str(ranking_file)

    def discover_existing_experiments(self, base_output_dir: Path) -> List[Dict[str, Any]]:
        """Discover existing experiment result directories and load their data."""
        # Delegate to ExperimentAggregator (Phase 2C refactoring)
        return self._services.aggregator().discover_experiments(base_output_dir)

    def load_experiment_data(self, csv_file: Path) -> List[Dict[str, Any]]:
        """Load experiment data from CSV file."""
        # Delegate to ExperimentAggregator (Phase 2C refactoring)
        return self._services.aggregator().load_data(csv_file)

    def aggregate_experiment_data(self, all_experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data from multiple experiment runs."""
        # Delegate to ExperimentAggregator (Phase 2C refactoring)
        return self._services.aggregator().aggregate(all_experiments)

    def _generate_failure_summary(self) -> None:
        """Generate and print failure summary."""
        formatter = FailureSummaryFormatter()

        # Print to console
        formatter.print_console_summary(self._results.failures, self._context.get_attempts())

        # Also log to file if available
        if self._context.is_logging():
            formatter.write_file_summary(self._context.log_handle, self._results.failures, self._context.get_attempts())

    def generate_persistent_summary(self, base_output_dir: Path, aggregated_data: Dict[str, Any]) -> List[str]:
        """Generate persistent summary files that aggregate all experiments."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        return self._services.output_generator().generate_persistent_summary(base_output_dir, aggregated_data)
    
    def _generate_persistent_summary_DEPRECATED(self, base_output_dir: Path, aggregated_data: Dict[str, Any]) -> List[str]:
        """Generate persistent summary files that aggregate all experiments."""
        if not aggregated_data:
            return []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_files = []

        # 1. Overall experiment summary
        summary_file = base_output_dir / "experiment_summary_latest.log"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("📊  COMPREHENSIVE EXPERIMENT SUMMARY (ALL RUNS)  📊\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"🧮 Total experiments analyzed: {aggregated_data['total_experiments']}\n")
            f.write(f"📊 Total individual results: {aggregated_data['total_results']}\n\n")

            # Recent experiments
            f.write("🕒 RECENT EXPERIMENTS:\n")
            f.write("─" * 40 + "\n")
            for exp in aggregated_data['experiment_metadata'][:5]:
                date_str = datetime.fromtimestamp(exp['date']).strftime('%Y-%m-%d %H:%M')
                f.write(f"  📁 {exp['timestamp']}: {exp['result_count']} results ({date_str})\n")

            # Template performance
            if aggregated_data.get('analysis', {}).get('template_ranking'):
                f.write(f"\n🏆 TOP TEMPLATES (All-Time Performance):\n")
                f.write("─" * 50 + "\n")
                for i, template_data in enumerate(aggregated_data['analysis']['template_ranking'][:5], 1):
                    f.write(f"  #{i}: {template_data['template'][:45]:45s}\n")
                    f.write(f"      📊 Avg: {template_data['avg_all_correct_rate']:6.1%} | Score: {template_data['score']:.3f}\n")

            # Problem insights
            if aggregated_data.get('analysis', {}).get('problem_analysis'):
                f.write(f"\n🎯 PROBLEM DIFFICULTY INSIGHTS:\n")
                f.write("─" * 40 + "\n")
                easy_problems = [p for p in aggregated_data['analysis']['problem_analysis'] if p['difficulty'] == 'EASY']
                hard_problems = [p for p in aggregated_data['analysis']['problem_analysis'] if p['difficulty'] in ['HARD', 'VERY_HARD']]

                if easy_problems:
                    f.write(f"  🟢 Easiest: {', '.join([p['problem'] for p in easy_problems[:3]])}\n")
                if hard_problems:
                    f.write(f"  🔴 Hardest: {', '.join([p['problem'] for p in hard_problems[:3]])}\n")

            f.write("\n" + "=" * 80 + "\n")

        generated_files.append(str(summary_file))

        # 2. Template performance trends (CSV)
        if pd is not None and aggregated_data['template_stats']:
            template_trends_data = []
            for template, stats in aggregated_data['template_stats'].items():
                template_trends_data.append({
                    'template': template,
                    'total_experiments': stats['experiments'],
                    'avg_success_rate': stats['avg_success_rate'],
                    'max_success_rate': stats['max_success_rate'],
                    'avg_duration': stats['avg_duration'],
                    'consistency_score': 1.0 - (max(stats['success_rates']) - min(stats['success_rates']))
                })

            template_df = pd.DataFrame(template_trends_data)
            template_df = template_df.sort_values('avg_success_rate', ascending=False)
            template_trends_file = base_output_dir / "template_performance_trends.csv"
            template_df.to_csv(template_trends_file, index=False)
            generated_files.append(str(template_trends_file))

        # 3. Problem performance analysis (CSV)
        if pd is not None and aggregated_data['problem_stats']:
            problem_trends_data = []
            classifier = DifficultyClassifier()
            for problem, stats in aggregated_data['problem_stats'].items():
                # Use refactored classifier (eliminates duplication #5)
                difficulty = classifier.classify_simple(stats['avg_success_rate'])
                problem_trends_data.append({
                    'problem': problem,
                    'total_experiments': stats['experiments'],
                    'avg_success_rate': stats['avg_success_rate'],
                    'max_success_rate': stats['max_success_rate'],
                    'templates_tested': len(stats['templates_used']),
                    'difficulty_rating': difficulty
                })

            problem_df = pd.DataFrame(problem_trends_data)
            problem_df = problem_df.sort_values('avg_success_rate', ascending=False)
            problem_trends_file = base_output_dir / "problem_difficulty_trends.csv"
            problem_df.to_csv(problem_trends_file, index=False)
            generated_files.append(str(problem_trends_file))

        # 4. Best combinations lookup table
        if aggregated_data.get('analysis', {}).get('best_template_per_problem'):
            lookup_file = base_output_dir / "best_template_lookup.json"
            import json

            lookup_data = {
                'generated': datetime.now().isoformat(),
                'recommendations': {}
            }

            for problem, recommendation in aggregated_data['analysis']['best_template_per_problem'].items():
                lookup_data['recommendations'][problem] = {
                    'best_template': recommendation['best_template'],
                    'success_rate': recommendation['best_score'],
                    'alternatives': [
                        {
                            'template': alt['template'],
                            'success_rate': alt['all_correct_rate']
                        }
                        for alt in recommendation['alternatives'][:3]
                    ]
                }

            with open(lookup_file, 'w') as f:
                json.dump(lookup_data, f, indent=2)
            generated_files.append(str(lookup_file))

        return generated_files

    def run_summarise_experiments(self, args: argparse.Namespace) -> None:
        """Analyze existing experiments and generate/update summary statistics."""
        # Delegate to ExperimentSummarizer (Phase 2D refactoring)
        self._services.summarizer().run(args)

    def generate_summary_log(self, output_dir: Path, timestamp: str, total_duration: float, templates_to_use: list, problems_to_use: list, total_combinations: int) -> str:
        """Generate detailed summary log file."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        return self._services.output_generator().generate_summary_log(output_dir, timestamp, total_duration, templates_to_use, problems_to_use, total_combinations)
    
    def _generate_summary_log_DEPRECATED(self, output_dir: Path, timestamp: str, total_duration: float, templates_to_use: list, problems_to_use: list, total_combinations: int) -> str:
        """Generate detailed summary log file."""
        if not self.results_data:
            return ""

        summary_log = output_dir / f"batch_summary_{timestamp}.log"

        with open(summary_log, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("🧠  ARC-AGI BATCH EXPERIMENT SUMMARY  🧠\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"📅 Experiment started: {self._timing.format_global_start()}\n\n")
            f.write("📋 CONFIGURATION:\n")
            f.write("─" * 40 + "\n")
            f.write(f"  🔧 Method: method1_text_prompt\n")
            f.write(f"  🤖 Model: openrouter/deepseek/deepseek-chat-v3-0324\n")
            f.write(f"  🔄 Iterations: 1\n\n")
            f.write(f"📝 Templates ({len(templates_to_use)}):\n")
            for i, template in enumerate(templates_to_use, 1):
                f.write(f"  {i}. {template}\n")
            f.write(f"\n🎯 Problems ({len(problems_to_use)}):\n")
            for i, problem in enumerate(problems_to_use, 1):
                f.write(f"  {i}. {problem}\n")
            f.write(f"\n🧮 Total combinations: {total_combinations}\n")
            f.write("=" * 80 + "\n\n")
            f.write("📊 INDIVIDUAL TEST RESULTS:\n")
            f.write("─" * 40 + "\n")

            grader = PerformanceGrader()
            for i, result in enumerate(self.results_data, 1):
                # Use refactored grader (eliminates duplication #3)
                success_icon = grader.success_icon(result['all_correct_rate'], result['at_least_one_correct_rate'])
                f.write(f"{success_icon} Test {i:02d}: {result['template']} + {result['problem']}\n")
                f.write(f"    📊 Results: {result['all_correct']}/{result['total_runs']} all correct ({result['all_correct_rate']:.1%}), ")
                f.write(f"{result['at_least_one_correct']}/{result['total_runs']} partial ({result['at_least_one_correct_rate']:.1%})\n")
                f.write(f"    ⏱️  Duration: {self.format_duration(result['individual_duration'])}\n\n")

            f.write("=" * 80 + "\n")
            f.write("🎉  EXPERIMENT COMPLETED  🎉\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"📅 Completed at: {self._timing.format_global_end()}\n")
            f.write(f"⏱️  Total duration: {self.format_duration(total_duration)}\n\n")

            if self.results_data:
                successful_tests = len([r for r in self.results_data if r['all_correct_rate'] > 0])
                f.write("📊 FINAL RESULTS SUMMARY:\n")
                f.write("─" * 40 + "\n")
                f.write(f"🎯 Tests with success: {successful_tests}/{len(self.results_data)} ({successful_tests / len(self.results_data):.1%})\n")
                f.write(f"⏱️  Average time per test: {self.format_duration(total_duration / total_combinations)}\n\n")

                # Template timing breakdown
                f.write("🕐 TEMPLATE TIMING BREAKDOWN:\n")
                f.write("─" * 40 + "\n")
                for template in templates_to_use:
                    template_time = self._timing.get_template_duration(template)
                    f.write(f"  📝 {template}: {self.format_duration(template_time)}\n")
            f.write("\n" + "=" * 80 + "\n")

        return str(summary_log)

    def _print_experiment_configuration(
        self, args: argparse.Namespace, templates_to_use: List[str], problems_to_use: List[str]
    ) -> int:
        """
        Print experiment configuration. Returns total_combinations.

        Following Object Calisthenics Rule 7: Small focused methods.
        """
        # Delegate to ConsoleDisplay (Phase 1 refactoring)
        total_combinations = self._services.console_display().print_experiment_configuration(args, templates_to_use, problems_to_use)
        self.log_timestamp(f"✅ Configuration complete. Starting {total_combinations} test combinations.")
        return total_combinations

    def _setup_output_directory(self, args: argparse.Namespace) -> Tuple[Path, str]:
        """
        Setup output directory and logging. Returns (output_dir, timestamp).

        Following Object Calisthenics Rule 7: Small focused methods.
        """
        base_output_dir = Path(args.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output_dir / timestamp

        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n💾 Results will be saved to: {output_dir}")

            # Setup log file
            log_file = output_dir / f"experiment_run_{timestamp}.log"
            self.log_file_handle = open(log_file, 'w')
            self.log_timestamp(f"Logging to: {log_file}")
            print(f"📝 Detailed log: {log_file.name}")

        return output_dir, timestamp

    def _load_method_module(self, args: argparse.Namespace) -> Optional[Any]:
        """
        Load and validate method module. Returns module or None.

        Following Object Calisthenics Rule 7: Small focused methods.
        """
        if args.dry_run:
            return None

        try:
            method_module = importlib.import_module(args.method)
            print(f"Using module: {method_module.__file__}")
            print(f"Using entry point: {method_module.run_experiment_for_iterations.__name__}")
            return method_module
        except ImportError as e:
            print(f"Error importing method module '{args.method}': {e}")
            if self.log_file_handle:
                self.log_file_handle.close()
            return None

    def _run_preflight_validation(
        self,
        templates_to_use: List[str],
        problems_to_use: List[str],
        method_module: Any,
        args: argparse.Namespace
    ) -> bool:
        """
        Run pre-flight validation. Returns True if passed.

        Following Object Calisthenics Rule 7: Small focused methods.
        """
        print(f"\n{'='*80}")
        print("🔍 PRE-FLIGHT VALIDATION")
        print(f"{'='*80}")

        validation_passed, validation_errors = self.validate_prerequisites(
            templates_to_use, problems_to_use, method_module, args
        )

        if not validation_passed:
            print(f"\n❌ Cannot proceed with experiments due to validation failures")
            if self.log_file_handle:
                self.log_file_handle.write("\nValidation failed:\n")
                for error in validation_errors:
                    self.log_file_handle.write(f"  - {error}\n")
                self.log_file_handle.close()
            return False

        print(f"{'='*80}\n")
        return True

    def _print_template_performance_summary(
        self, template: str, template_results: List[Dict[str, Any]], formatted_duration: str
    ) -> None:
        """Print performance summary for a completed template.

        Args:
            template: Template name
            template_results: List of result dictionaries for this template
            formatted_duration: Formatted duration string
        """
        # Delegate to ConsoleDisplay (Phase 1 refactoring)
        console = self._get_console_display()
        console.print_template_performance_summary(template, template_results, formatted_duration)

    def _print_key_insights(self, analysis: Dict[str, Any]) -> None:
        """Print key insights and rankings from experiment analysis.

        Args:
            analysis: Analysis dictionary from generate_ranking_analysis()
        """
        # Delegate to ConsoleDisplay (Phase 1 refactoring)
        console = self._get_console_display()
        console.print_key_insights(analysis)

    def _generate_and_save_outputs(
        self, output_dir: Path, timestamp: str, total_duration: float,
        templates_to_use: List[str], problems_to_use: List[str], total_combinations: int
    ) -> None:
        """Generate and save all output files, then print file summary.

        Args:
            output_dir: Output directory path
            timestamp: Timestamp string for filenames
            total_duration: Total experiment duration in seconds
            templates_to_use: List of templates used
            problems_to_use: List of problems used
            total_combinations: Total number of test combinations
        """
        csv_file = self.generate_csv_output(output_dir)
        html_file = self.generate_html_output(output_dir)
        summary_log = self.generate_summary_log(output_dir, timestamp, total_duration, templates_to_use, problems_to_use, total_combinations)
        ranking_report = self.generate_ranking_report(output_dir, timestamp)

        print(f"\n💾 " + "=" * 60)
        print(f"📁 RESULTS SAVED")
        print("=" * 65)
        print(f"    📂 Directory: {output_dir}")
        if csv_file and csv_file != "":
            csv_path = Path(csv_file) if isinstance(csv_file, str) else csv_file
            print(f"    📊 CSV File: {csv_path.name}")
        if html_file and html_file != "":
            html_path = Path(html_file) if isinstance(html_file, str) else html_file
            print(f"    🌐 HTML Report: {html_path.name}")
        if summary_log:
            print(f"    📝 Summary Log: {Path(summary_log).name}")
        if ranking_report:
            print(f"    🏆 Ranking Analysis: {Path(ranking_report).name}")
        print("=" * 65)

    def _print_final_summary(
        self, total_duration: float, total_combinations: int,
        templates_to_use: List[str], problems_to_use: List[str]
    ) -> None:
        """Print final experiment summary including performance breakdown and LLM statistics.

        Args:
            total_duration: Total experiment duration in seconds
            total_combinations: Total number of test combinations
            templates_to_use: List of templates used
            problems_to_use: List of problems used
        """
        # Delegate to ConsoleDisplay (Phase 1 refactoring)
        console = self._get_console_display()
        console.print_final_summary(
            total_duration, total_combinations, templates_to_use, problems_to_use,
            self.results_data, self.all_llm_responses
        )
        self.log_timestamp("🎉 EXPERIMENT COMPLETED! 🎉")

    def run_batch_experiments(self, args: argparse.Namespace) -> None:
        """Main execution method with comprehensive tracking."""
        # Record global start time
        self._timing.start_global()

        # Stylish header
        print("\n" + "=" * 80)
        print("🧠  ARC-AGI BATCH EXPERIMENT RUNNER  🧠")
        print("=" * 80)
        self.log_timestamp("🚀 EXPERIMENT STARTED")

        # Resolve selections
        self.log_timestamp("🔍 Resolving template and problem selections...")
        templates_to_use = ExperimentConfigResolver.resolve_templates(args.templates)
        problems_to_use = ExperimentConfigResolver.resolve_problems(args.problems)

        # Display configuration (extracted method)
        total_combinations = self._print_experiment_configuration(args, templates_to_use, problems_to_use)

        # Setup output directory and logging (extracted method)
        output_dir, timestamp = self._setup_output_directory(args)

        # Load method module (extracted method)
        method_module = self._load_method_module(args)
        if method_module is None and not args.dry_run:
            return

        # Run pre-flight validation (extracted method)
        if not self._run_preflight_validation(templates_to_use, problems_to_use, method_module, args):
            return

        # Execute experiments using orchestrator (removes 100+ lines of procedural code)
        orchestrator = self._get_loop_orchestrator()
        loop_result = orchestrator.execute_loop(
            templates_to_use, problems_to_use, method_module, args, output_dir
        )
        
        # Collect results from orchestrator
        self.results_data = loop_result['results_data']
        self.all_llm_responses = loop_result['all_llm_responses']
        self.failed_experiments = loop_result['failed_experiments']
        
        # If loop stopped early due to fail-fast, handle it
        if not loop_result['completed']:
            self._generate_failure_summary()
            return
        
        # Print template performance summaries
        if not args.dry_run:
            for template in templates_to_use:
                template_results = [r for r in self.results_data if r['template'] == template]
                if template_results:
                    template_duration = self._timing.get_template_duration(template)
                    formatted_duration = self.format_duration(template_duration)
                    self._print_template_performance_summary(template, template_results, formatted_duration)

        # Record global end time
        self._timing.end_global()
        total_duration = self._timing.get_global_duration()

        self.log_timestamp("🎉 All tests completed. Generating results...")

        # Generate outputs
        if not args.dry_run and self.results_data:
            self.generate_console_table()

            # Generate ranking analysis and show key insights
            analysis = self.generate_ranking_analysis()
            if analysis:
                self._print_key_insights(analysis)

            self._generate_and_save_outputs(output_dir, timestamp, total_duration, templates_to_use, problems_to_use, total_combinations)

        # Final summary
        self._print_final_summary(total_duration, total_combinations, templates_to_use, problems_to_use)

        # Generate failure summary if there were any failures
        if self.failed_experiments:
            self._generate_failure_summary()

        # Close log file
        if self.log_file_handle:
            self.log_file_handle.write(f"\n{'='*80}\n")
            self.log_file_handle.write("EXPERIMENT COMPLETED\n")
            self.log_file_handle.write(f"{'='*80}\n")
            self.log_file_handle.close()
            self.log_file_handle = None


def main():
    """Main entry point."""
    runner = BatchExperimentRunner()
    args = runner.parse_arguments()

    try:
        if args.summarise_experiments:
            runner.run_summarise_experiments(args)
        else:
            runner.run_batch_experiments(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        if runner.failed_experiments:
            runner._generate_failure_summary()
        if runner.log_file_handle:
            runner.log_file_handle.write("\n\nEXPERIMENT INTERRUPTED BY USER\n")
            runner.log_file_handle.close()
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        if runner.failed_experiments:
            runner._generate_failure_summary()
        if runner.log_file_handle:
            runner.log_file_handle.write(f"\n\nEXPERIMENT FAILED: {str(e)}\n")
            runner.log_file_handle.write(traceback.format_exc())
            runner.log_file_handle.close()


if __name__ == "__main__":
    main()