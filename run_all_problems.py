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
import time
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
from utils import do_first_setup

# Refactored classes to eliminate duplication
from domain.value_objects import SuccessThresholds, DifficultyThresholds
from analysis.performance_grader import PerformanceGrader
from analysis.difficulty_classifier import DifficultyClassifier
from analysis.statistics_aggregator import (
    TemplateStatisticsAggregator,
    ProblemStatisticsAggregator,
    ExperimentStatisticsAggregator,
    BestTemplateRecommender
)


class BatchExperimentRunner:
    """Main class for running batch experiments with comprehensive tracking."""

    def __init__(self):
        self.timing_data: Dict[str, float] = {}
        self.template_timings: Dict[str, float] = {}
        self.individual_timings: Dict[str, float] = {}
        self.problem_timings: Dict[str, float] = {}
        self.results_data: List[Dict[str, Any]] = []
        self.all_llm_responses: List[Any] = []
        self.global_start_time: float = 0
        self.global_end_time: float = 0
        self.failed_experiments: List[Dict[str, Any]] = []
        self.total_experiments_attempted: int = 0
        self.log_file_handle = None

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

    def discover_templates(self) -> List[str]:
        """Discover available J2 template files."""
        prompts_dir = Path("prompts")
        if not prompts_dir.exists():
            raise FileNotFoundError("prompts directory not found")

        templates = []
        for j2_file in prompts_dir.glob("*.j2"):
            # Skip templates with 'spelke' in name as per original script
            if 'spelke' not in j2_file.name.lower():
                templates.append(j2_file.name)

        # Sort to ensure consistent ordering
        return sorted(templates)

    def get_default_problems(self) -> List[str]:
        """Get default problem IDs from run_all_problems.py."""
        return [
            "0d3d703e",  # fixed colour mapping, 3x3 grid min
            "08ed6ac7",  # coloured in order of height, 9x9 grid min
            "178fcbfb",  # dots form coloured lines, 9x9 grid min
            "9565186b",  # most frequent colour wins, 3x3 grid min
            "0a938d79",  # dots form repeated coloured lines, 9x22 grid min
        ]

    def resolve_templates(self, selection: Optional[str]) -> List[str]:
        """Resolve template selection from string or indices."""
        available_templates = self.discover_templates()

        if not selection:
            return available_templates

        selected = []
        for item in selection.split(","):
            item = item.strip()
            if item.isdigit():
                # Index selection
                idx = int(item)
                if 0 <= idx < len(available_templates):
                    selected.append(available_templates[idx])
                else:
                    print(f"Warning: Template index {idx} out of range")
            else:
                # Name selection
                if item in available_templates:
                    selected.append(item)
                else:
                    print(f"Warning: Template '{item}' not found")

        return selected

    def resolve_problems(self, selection: Optional[str]) -> List[str]:
        """Resolve problem selection from string or indices."""
        default_problems = self.get_default_problems()

        if not selection:
            return default_problems

        selected = []
        for item in selection.split(","):
            item = item.strip()
            if item.isdigit():
                # Check if it's an index or a problem ID
                if len(item) <= 2:  # Assume index if short number
                    idx = int(item)
                    if 0 <= idx < len(default_problems):
                        selected.append(default_problems[idx])
                    else:
                        print(f"Warning: Problem index {idx} out of range")
                else:
                    # Treat as problem ID
                    selected.append(item)
            else:
                # Direct problem ID
                selected.append(item)

        return selected

    def log_timestamp(self, message: str, to_file_only: bool = False) -> None:
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"🕒 [{timestamp}] {message}"

        if not to_file_only:
            print(log_msg)

        if self.log_file_handle:
            self.log_file_handle.write(log_msg + "\n")
            self.log_file_handle.flush()

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

    def validate_prerequisites(self, templates_to_use: List[str], problems_to_use: List[str], method_module: Any, args: argparse.Namespace) -> Tuple[bool, List[str]]:
        """Validate all prerequisites before starting experiments."""
        errors = []

        self.log_timestamp("🔍 Running pre-flight validation...")

        # 1. Validate templates exist
        print("  ✓ Checking templates...")
        available_templates = self.discover_templates()
        for template in templates_to_use:
            if template not in available_templates:
                errors.append(f"Template not found: {template}")
            else:
                print(f"    ✓ {template}")

        # 2. Validate at least one problem can be loaded
        print("  ✓ Checking problem data access...")
        if problems_to_use:
            try:
                test_problem = utils.get_examples(problems_to_use[0])
                if not test_problem:
                    errors.append(f"Problem {problems_to_use[0]} loaded but returned empty data")
                else:
                    print(f"    ✓ Successfully loaded problem: {problems_to_use[0]}")
                    print(f"      - Train examples: {len(test_problem.get('train', []))}")
                    print(f"      - Test examples: {len(test_problem.get('test', []))}")
            except Exception as e:
                errors.append(f"Failed to load problem {problems_to_use[0]}: {str(e)}")

        # 3. Validate method module has required function
        print("  ✓ Checking method module...")
        if not hasattr(method_module, 'run_experiment_for_iterations'):
            errors.append("Method module missing 'run_experiment_for_iterations' function")
        else:
            print(f"    ✓ Method module loaded: {method_module.__file__}")

        # Note: Database and API validation will happen during first experiment setup
        # We skip them here to avoid argument parsing conflicts with do_first_setup()
        print("  ✓ Database and API validation will occur during experiment setup")

        if errors:
            print("\n❌ PRE-FLIGHT VALIDATION FAILED:")
            for error in errors:
                print(f"   ✗ {error}")
            return False, errors
        else:
            print("\n✅ All pre-flight checks passed!")
            return True, []

    def setup_experiment_without_argparse(self, experiment_folder: Path) -> Tuple[Any, str]:
        """Setup experiment folder and database without re-parsing arguments."""
        logger = utils.setup_logging(experiment_folder)
        logger.info("Started experiment")
        db_filename = utils.make_db(experiment_folder)
        logger.info(f"Database created at: {db_filename}")
        return logger, db_filename

    def run_single_experiment(
        self,
        template: str,
        problem: str,
        method_module: Any,
        args: argparse.Namespace,
        experiment_folder: Path
    ) -> Tuple[List[Any], List[Any]]:
        """Run a single experiment with timing."""
        self.log_timestamp(f"Starting individual test: {template} + {problem}")
        self.total_experiments_attempted += 1

        if args.dry_run:
            print(f"[DRY-RUN] Would run {template} with {problem}")
            return [], []

        individual_start_time = time.time()

        try:
            # Setup for this specific experiment (without re-parsing arguments)
            logger, db_filename = self.setup_experiment_without_argparse(experiment_folder)

            # Load the specific problem
            problems = utils.get_examples(problem)

            # Set logger for the method module
            method_module.logger = logger

            # Print iteration progress header
            print(f"    🔄 Running {args.iterations} iteration(s)...")
            iteration_start_time = time.time()

            # Run the experiment
            llm_responses, rr_trains = method_module.run_experiment_for_iterations(
                db_filename,
                model=args.model,
                iterations=args.iterations,
                problems=problems,
                template_name=template,
            )

            iteration_end_time = time.time()
            iteration_duration = iteration_end_time - iteration_start_time

            # Show iteration timing summary
            avg_per_iteration = iteration_duration / args.iterations if args.iterations > 0 else 0
            print(f"    ✓ Completed {args.iterations} iteration(s) in {self.format_duration(iteration_duration)}")
            print(f"      (avg: {self.format_duration(avg_per_iteration)} per iteration)")

            individual_end_time = time.time()
            individual_duration = individual_end_time - individual_start_time

            # Store timing
            self.individual_timings[f"{template}|{problem}"] = individual_duration

            formatted_duration = self.format_duration(individual_duration)
            self.log_timestamp(f"Individual test completed successfully in {formatted_duration}")

            return llm_responses, rr_trains

        except Exception as e:
            individual_end_time = time.time()
            individual_duration = individual_end_time - individual_start_time

            # Store timing even for failures
            self.individual_timings[f"{template}|{problem}"] = individual_duration

            # Record the failure
            self.failed_experiments.append({
                'template': template,
                'problem': problem,
                'error': str(e),
                'duration': individual_duration,
                'traceback': traceback.format_exc() if args.verbose else None
            })

            formatted_duration = self.format_duration(individual_duration)
            self.log_timestamp(f"Individual test failed after {formatted_duration}: {str(e)}")

            # Log full traceback to file
            if self.log_file_handle:
                self.log_file_handle.write(f"\n{'='*60}\n")
                self.log_file_handle.write(f"FAILURE: {template} + {problem}\n")
                self.log_file_handle.write(f"Error: {str(e)}\n")
                self.log_file_handle.write(f"{'='*60}\n")
                self.log_file_handle.write(traceback.format_exc())
                self.log_file_handle.write(f"\n{'='*60}\n\n")
                self.log_file_handle.flush()

            raise e

    def analyze_results(
        self,
        template: str,
        problem: str,
        rr_trains: List[Any],
        iterations: int,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Analyze experiment results for success rates with real-time feedback."""
        all_correct = 0
        at_least_one_correct = 0

        print(f"    📊 Analyzing {iterations} iteration(s) for {problem}:")

        for i, rr_train in enumerate(rr_trains, 1):
            ran_all_train_problems_correctly = rr_train[0].transform_ran_and_matched_for_all_inputs
            ran_at_least_one_train_problem_correctly = rr_train[0].transform_ran_and_matched_at_least_once

            # Get detailed sub-problem results
            execution_outcomes = rr_train[1] if len(rr_train) > 1 else []

            # Real-time feedback for each iteration
            if ran_all_train_problems_correctly:
                all_correct += 1
                status_icon = "✅"
                status_text = "ALL_CORRECT"
            elif ran_at_least_one_train_problem_correctly:
                status_icon = "⚠️"
                status_text = "PARTIAL"
            else:
                status_icon = "❌"
                status_text = "FAILED"

            # Show detailed iteration results
            print(f"      Iteration {i}: {status_icon} {status_text}")

            # Show sub-problem details if we have execution outcomes
            if execution_outcomes and verbose:
                for j, outcome in enumerate(execution_outcomes):
                    sub_icon = "✅" if outcome.was_correct else "❌"
                    print(f"        Sub-problem {j+1}: {sub_icon} {'PASS' if outcome.was_correct else 'FAIL'}")
            elif execution_outcomes:
                # Show summary of sub-problems even without verbose
                correct_count = sum(1 for outcome in execution_outcomes if outcome.was_correct)
                total_count = len(execution_outcomes)
                print(f"        Sub-problems: {correct_count}/{total_count} correct")

            # Count at_least_one_correct properly
            if ran_at_least_one_train_problem_correctly:
                at_least_one_correct += 1

        # Summary for this problem
        all_correct_rate = all_correct / iterations
        at_least_one_rate = at_least_one_correct / iterations

        if all_correct_rate >= 0.8:
            summary_icon = "🎯"
            summary_color = "EXCELLENT"
        elif all_correct_rate >= 0.5:
            summary_icon = "✅"
            summary_color = "GOOD"
        elif at_least_one_rate >= 0.5:
            summary_icon = "⚠️"
            summary_color = "PARTIAL"
        else:
            summary_icon = "❌"
            summary_color = "POOR"

        print(f"    {summary_icon} Problem Results: {all_correct}/{iterations} all correct ({all_correct_rate:.1%}), "
              f"{at_least_one_correct}/{iterations} partial ({at_least_one_rate:.1%}) - {summary_color}")

        return {
            "template": template,
            "problem": problem,
            "total_runs": iterations,
            "all_correct": all_correct,
            "at_least_one_correct": at_least_one_correct,
            "all_correct_rate": all_correct_rate,
            "at_least_one_correct_rate": at_least_one_rate,
            "individual_duration": self.individual_timings.get(f"{template}|{problem}", 0),
            "problem_duration": self.problem_timings.get(f"{template}|{problem}", 0),
        }

    def generate_console_table(self) -> None:
        """Generate formatted console table."""
        if not self.results_data:
            return

        print(f"\n📋 " + "═" * 90)
        print(f"📊 DETAILED RESULTS TABLE 📊")
        print("═" * 95)

        # Header with better formatting
        print(f"{'📝 Template':<40} {'🎲 Problem':<12} {'🎯 All%':<8} {'⚠️ Part%':<9} {'⏱️ Time':<10} {'📊 Grade':<8}")
        print("─" * 95)

        # Data rows with visual indicators
        grader = PerformanceGrader()
        for result in self.results_data:
            template_short = result['template'][:35] + "..." if len(result['template']) > 35 else result['template']
            test_time = self.format_duration(result['individual_duration'])

            # Use refactored grader (eliminates duplication #1)
            grade = grader.grade_with_icon(result['all_correct_rate'])

            print(f"{template_short:<40} {result['problem']:<12} {result['all_correct_rate']:<7.1%} "
                  f"{result['at_least_one_correct_rate']:<8.1%} {test_time:<10} {grade:<8}")

        print("─" * 95)
        print(f"Legend: 🎯A(≥80%) ✅B(60-79%) ⚠️C(40-59%) 🔶D(partial) ❌F(<40%)")
        print("═" * 95)

    def generate_csv_output(self, output_dir: Path) -> str:
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
        if not self.results_data:
            return ""

        # Use the output directory name as timestamp for consistency
        timestamp = output_dir.name
        html_file = output_dir / f"batch_results_{timestamp}.html"

        total_duration = self.global_end_time - self.global_start_time

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
        if not self.results_data:
            return ""

        analysis = self.generate_ranking_analysis()
        if not analysis:
            return ""

        ranking_file = output_dir / f"ranking_analysis_{timestamp}.log"

        with open(ranking_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("🏆  COMPREHENSIVE RANKING & PERFORMANCE ANALYSIS  🏆\n")
            f.write("=" * 80 + "\n\n")

            # 1. Top performing experiments
            f.write("🥇 TOP PERFORMING EXPERIMENTS (Template + Problem Combinations):\n")
            f.write("─" * 70 + "\n")
            for i, result in enumerate(analysis['experiment_ranking'][:10], 1):
                grade = "🎯" if result['all_correct_rate'] >= 0.8 else "✅" if result['all_correct_rate'] >= 0.6 else "⚠️"
                f.write(f"{grade} #{i:2d}: {result['template'][:35]:35s} + {result['problem']}\n")
                f.write(f"      📊 {result['all_correct_rate']:6.1%} all correct, {result['at_least_one_correct_rate']:6.1%} partial\n")
                f.write(f"      ⏱️  {self.format_duration(result['individual_duration'])}\n\n")

            # 2. Template ranking
            f.write("=" * 80 + "\n")
            f.write("📝 TEMPLATE RANKING (Overall Performance Across All Problems):\n")
            f.write("─" * 70 + "\n")
            for i, template_data in enumerate(analysis['template_ranking'], 1):
                grade = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                f.write(f"{grade} #{i}: {template_data['template']}\n")
                f.write(f"    📊 Average: {template_data['avg_all_correct_rate']:6.1%} all correct, {template_data['avg_partial_rate']:6.1%} partial\n")
                f.write(f"    🎯 Excellent problems: {template_data['excellent_problems']}/{template_data['total_problems']} ({template_data['excellent_problems']/template_data['total_problems']:.1%})\n")
                f.write(f"    ✅ Good problems: {template_data['good_problems']}/{template_data['total_problems']} ({template_data['good_problems']/template_data['total_problems']:.1%})\n")
                f.write(f"    ⏱️  Average duration: {self.format_duration(template_data['avg_duration'])}\n")
                f.write(f"    📈 Weighted score: {template_data['score']:.3f}\n\n")

            # 3. Problem difficulty analysis
            f.write("=" * 80 + "\n")
            f.write("🎯 PROBLEM DIFFICULTY ANALYSIS:\n")
            f.write("─" * 70 + "\n")

            difficulty_groups = {}
            for problem_data in analysis['problem_analysis']:
                difficulty = problem_data['difficulty']
                if difficulty not in difficulty_groups:
                    difficulty_groups[difficulty] = []
                difficulty_groups[difficulty].append(problem_data)

            for difficulty in ['EASY', 'MEDIUM', 'HARD', 'VERY_HARD']:
                if difficulty in difficulty_groups:
                    problems = difficulty_groups[difficulty]
                    f.write(f"🌟 {difficulty} Problems ({len(problems)}):\n")
                    for problem_data in problems:
                        difficulty_icon = "🟢" if difficulty == "EASY" else "🟡" if difficulty == "MEDIUM" else "🟠" if difficulty == "HARD" else "🔴"
                        f.write(f"  {difficulty_icon} {problem_data['problem']}: {problem_data['avg_all_correct_rate']:6.1%} avg success\n")
                        f.write(f"     🏆 Best template: {problem_data['best_template'][:40]:40s} ({problem_data['best_score']:6.1%})\n")
                    f.write("\n")

            # 4. Best template recommendations
            f.write("=" * 80 + "\n")
            f.write("🎯 OPTIMAL TEMPLATE RECOMMENDATIONS PER PROBLEM:\n")
            f.write("─" * 70 + "\n")
            for problem, recommendation in analysis['best_template_per_problem'].items():
                f.write(f"🎲 Problem: {problem}\n")
                f.write(f"   🏆 Best: {recommendation['best_template'][:50]:50s} ({recommendation['best_score']:6.1%})\n")
                if recommendation['alternatives']:
                    f.write(f"   📋 Alternatives:\n")
                    for alt in recommendation['alternatives']:
                        f.write(f"      • {alt['template'][:45]:45s} ({alt['all_correct_rate']:6.1%})\n")
                f.write("\n")

            # 5. Comparative analysis summary
            f.write("=" * 80 + "\n")
            f.write("📊 COMPARATIVE ANALYSIS SUMMARY:\n")
            f.write("─" * 70 + "\n")

            best_template = analysis['template_ranking'][0]
            worst_template = analysis['template_ranking'][-1]
            easiest_problem = analysis['problem_analysis'][0]
            hardest_problem = analysis['problem_analysis'][-1]

            f.write(f"🏆 Best Overall Template: {best_template['template']}\n")
            f.write(f"   📊 Average success rate: {best_template['avg_all_correct_rate']:.1%}\n")
            f.write(f"   🎯 Excellent on {best_template['excellent_problems']}/{best_template['total_problems']} problems\n\n")

            f.write(f"⚠️  Most Challenging Template: {worst_template['template']}\n")
            f.write(f"   📊 Average success rate: {worst_template['avg_all_correct_rate']:.1%}\n\n")

            f.write(f"🟢 Easiest Problem: {easiest_problem['problem']}\n")
            f.write(f"   📊 Average success rate: {easiest_problem['avg_all_correct_rate']:.1%}\n")
            f.write(f"   🏆 Best template: {easiest_problem['best_template']}\n\n")

            f.write(f"🔴 Hardest Problem: {hardest_problem['problem']}\n")
            f.write(f"   📊 Average success rate: {hardest_problem['avg_all_correct_rate']:.1%}\n")
            f.write(f"   🏆 Best template: {hardest_problem['best_template']}\n\n")

            f.write("=" * 80 + "\n")

        return str(ranking_file)

    def discover_existing_experiments(self, base_output_dir: Path) -> List[Dict[str, Any]]:
        """Discover existing experiment result directories and load their data."""
        if not base_output_dir.exists():
            print(f"⚠️  Output directory {base_output_dir} does not exist")
            return []

        experiment_dirs = []
        for item in base_output_dir.iterdir():
            if item.is_dir() and item.name.replace('_', '').replace('-', '').isdigit():
                # Look for CSV files in the directory
                csv_files = list(item.glob("batch_results_*.csv"))
                if csv_files:
                    experiment_dirs.append({
                        'directory': item,
                        'timestamp': item.name,
                        'csv_file': csv_files[0],
                        'date': item.stat().st_mtime
                    })

        # Sort by date (most recent first)
        experiment_dirs.sort(key=lambda x: x['date'], reverse=True)
        return experiment_dirs

    def load_experiment_data(self, csv_file: Path) -> List[Dict[str, Any]]:
        """Load experiment data from CSV file."""
        if pd is None:
            print("⚠️  pandas not available, cannot load experiment data")
            return []

        try:
            df = pd.read_csv(csv_file)
            return df.to_dict('records')
        except Exception as e:
            print(f"⚠️  Error loading {csv_file}: {e}")
            return []

    def aggregate_experiment_data(self, all_experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data from multiple experiment runs."""
        if not all_experiments:
            return {}

        # Combine all experiment data
        all_results = []
        experiment_metadata = []

        for exp in all_experiments:
            data = self.load_experiment_data(exp['csv_file'])
            if data:
                # Add metadata to each result
                for result in data:
                    result['experiment_timestamp'] = exp['timestamp']
                    result['experiment_date'] = exp['date']
                all_results.extend(data)

                experiment_metadata.append({
                    'timestamp': exp['timestamp'],
                    'date': exp['date'],
                    'result_count': len(data),
                    'directory': str(exp['directory'])
                })

        if not all_results:
            return {}

        # Create temporary results data for analysis
        original_results = self.results_data
        self.results_data = all_results

        # Generate analysis
        analysis = self.generate_ranking_analysis()

        # Restore original results
        self.results_data = original_results

        # Add aggregated statistics
        # Refactored: Use aggregator classes (eliminates duplication #9)
        template_aggregator = TemplateStatisticsAggregator(all_results)
        template_stats = template_aggregator.aggregate_to_dict()

        problem_aggregator = ProblemStatisticsAggregator(all_results)
        problem_stats = problem_aggregator.aggregate_to_dict()

        experiment_aggregator = ExperimentStatisticsAggregator(all_results)
        experiment_stats = experiment_aggregator.aggregate_to_dict()

        return {
            'analysis': analysis,
            'template_stats': template_stats,
            'problem_stats': problem_stats,
            'experiment_stats': experiment_stats,
            'experiment_metadata': experiment_metadata,
            'total_results': len(all_results),
            'total_experiments': len(all_experiments)
        }

    def _generate_failure_summary(self) -> None:
        """Generate and print failure summary."""
        if not self.failed_experiments:
            return

        print(f"\n{'='*80}")
        print(f"⚠️  FAILURE SUMMARY ⚠️")
        print(f"{'='*80}")
        print(f"Total experiments attempted: {self.total_experiments_attempted}")
        print(f"Failed experiments: {len(self.failed_experiments)}")
        print(f"Success rate: {(self.total_experiments_attempted - len(self.failed_experiments)) / self.total_experiments_attempted:.1%}")
        print(f"\n📋 Failed Experiment Details:")
        print("─" * 70)

        for i, failure in enumerate(self.failed_experiments, 1):
            print(f"\n❌ Failure #{i}:")
            print(f"   Template: {failure['template']}")
            print(f"   Problem: {failure['problem']}")
            print(f"   Error: {failure['error']}")
            print(f"   Duration: {self.format_duration(failure['duration'])}")

        print(f"\n{'='*80}")

        # Also log to file if available
        if self.log_file_handle:
            self.log_file_handle.write(f"\n{'='*80}\n")
            self.log_file_handle.write("FAILURE SUMMARY\n")
            self.log_file_handle.write(f"{'='*80}\n")
            self.log_file_handle.write(f"Total experiments attempted: {self.total_experiments_attempted}\n")
            self.log_file_handle.write(f"Failed experiments: {len(self.failed_experiments)}\n\n")

            for i, failure in enumerate(self.failed_experiments, 1):
                self.log_file_handle.write(f"\nFailure #{i}:\n")
                self.log_file_handle.write(f"  Template: {failure['template']}\n")
                self.log_file_handle.write(f"  Problem: {failure['problem']}\n")
                self.log_file_handle.write(f"  Error: {failure['error']}\n")
                self.log_file_handle.write(f"  Duration: {self.format_duration(failure['duration'])}\n")
                if failure.get('traceback'):
                    self.log_file_handle.write(f"\nTraceback:\n{failure['traceback']}\n")

            self.log_file_handle.write(f"\n{'='*80}\n\n")
            self.log_file_handle.flush()

    def generate_persistent_summary(self, base_output_dir: Path, aggregated_data: Dict[str, Any]) -> List[str]:
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
        print("\n" + "=" * 80)
        print("📊  EXPERIMENT SUMMARIZATION MODE  📊")
        print("=" * 80)

        base_output_dir = Path(args.output_dir)
        self.log_timestamp(f"🔍 Scanning for existing experiments in {base_output_dir}")

        # Discover existing experiments
        experiments = self.discover_existing_experiments(base_output_dir)

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

        self.log_timestamp("📈 Aggregating experiment data...")

        # Aggregate all experiment data
        aggregated_data = self.aggregate_experiment_data(experiments)

        if not aggregated_data:
            print("❌ No valid experiment data could be loaded")
            return

        # Generate summary insights (console output)
        print(f"\n🏆 " + "=" * 70)
        print(f"📊 AGGREGATED INSIGHTS FROM ALL EXPERIMENTS")
        print("=" * 75)

        if aggregated_data.get('analysis', {}).get('template_ranking'):
            best_template = aggregated_data['analysis']['template_ranking'][0]
            print(f"🥇 Best Overall Template: {best_template['template']}")
            print(f"   📊 Average success: {best_template['avg_all_correct_rate']:.1%}")
            print(f"   🎯 Tested on {best_template['total_problems']} problems")

        if aggregated_data.get('template_stats'):
            print(f"\n📝 Template Statistics:")
            template_count = len(aggregated_data['template_stats'])
            print(f"   🔢 Total templates tested: {template_count}")

            most_tested = max(aggregated_data['template_stats'].items(),
                            key=lambda x: x[1]['experiments'])
            print(f"   🧪 Most tested: {most_tested[0][:40]:40s} ({most_tested[1]['experiments']} times)")

        if aggregated_data.get('problem_stats'):
            print(f"\n🎯 Problem Statistics:")
            problem_count = len(aggregated_data['problem_stats'])
            print(f"   🔢 Total problems tested: {problem_count}")

            hardest_problem = min(aggregated_data['problem_stats'].items(),
                                key=lambda x: x[1]['avg_success_rate'])
            print(f"   🔴 Hardest problem: {hardest_problem[0]} ({hardest_problem[1]['avg_success_rate']:.1%} avg success)")

        print("=" * 75)

        self.log_timestamp("💾 Generating persistent summary files...")

        # Generate persistent summary files
        generated_files = self.generate_persistent_summary(base_output_dir, aggregated_data)

        print(f"\n💾 " + "=" * 60)
        print(f"📁 SUMMARY FILES GENERATED")
        print("=" * 65)
        print(f"    📂 Directory: {base_output_dir}")
        for file_path in generated_files:
            file_name = Path(file_path).name
            print(f"    📄 {file_name}")
        print("=" * 65)

        self.log_timestamp("✅ Experiment summarization completed!")
        print("\n💡 Use these files to:")
        print("   📊 Track template performance over time")
        print("   🎯 Identify problem difficulty patterns")
        print("   🔍 Find optimal template selections")
        print("   📈 Monitor experiment trends")

    def generate_summary_log(self, output_dir: Path, timestamp: str, total_duration: float, templates_to_use: list, problems_to_use: list, total_combinations: int) -> str:
        """Generate detailed summary log file."""
        if not self.results_data:
            return ""

        summary_log = output_dir / f"batch_summary_{timestamp}.log"

        with open(summary_log, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("🧠  ARC-AGI BATCH EXPERIMENT SUMMARY  🧠\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"📅 Experiment started: {datetime.fromtimestamp(self.global_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
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
            f.write(f"📅 Completed at: {datetime.fromtimestamp(self.global_end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
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
                    template_time = self.template_timings.get(template, 0)
                    f.write(f"  📝 {template}: {self.format_duration(template_time)}\n")
            f.write("\n" + "=" * 80 + "\n")

        return str(summary_log)

    def run_batch_experiments(self, args: argparse.Namespace) -> None:
        """Main execution method with comprehensive tracking."""
        # Record global start time
        self.global_start_time = time.time()

        # Stylish header
        print("\n" + "=" * 80)
        print("🧠  ARC-AGI BATCH EXPERIMENT RUNNER  🧠")
        print("=" * 80)
        self.log_timestamp("🚀 EXPERIMENT STARTED")

        # Resolve selections
        self.log_timestamp("🔍 Resolving template and problem selections...")
        templates_to_use = self.resolve_templates(args.templates)
        problems_to_use = self.resolve_problems(args.problems)

        # Configuration display with better formatting
        print(f"\n📋 EXPERIMENT CONFIGURATION")
        print("─" * 50)
        print(f"  🔧 Method Module: {args.method}")
        print(f"  🤖 Model: {args.model}")
        print(f"  🔄 Iterations per test: {args.iterations}")

        print(f"\n📝 Templates to test ({len(templates_to_use)}):")
        for i, template in enumerate(templates_to_use):
            print(f"    {i+1}. {template}")

        print(f"\n🎯 Problems to test ({len(problems_to_use)}):")
        for i, problem in enumerate(problems_to_use):
            print(f"    {i+1}. {problem}")

        # Calculate total combinations
        total_combinations = len(templates_to_use) * len(problems_to_use)
        print(f"\n🧮 Total test combinations: {total_combinations}")
        print("─" * 50)
        self.log_timestamp(f"✅ Configuration complete. Starting {total_combinations} test combinations.")

        # Create timestamp-based output directory and setup logging
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

        # Load method module
        if not args.dry_run:
            try:
                method_module = importlib.import_module(args.method)
                print(f"Using module: {method_module.__file__}")
                print(f"Using entry point: {method_module.run_experiment_for_iterations.__name__}")
            except ImportError as e:
                print(f"Error importing method module '{args.method}': {e}")
                if self.log_file_handle:
                    self.log_file_handle.close()
                return
        else:
            method_module = None

        # Run pre-flight validation
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
            return

        print(f"{'='*80}\n")

        # Execute experiments with branching timing
        current_test = 0

        # Template-level timing loop
        for template in templates_to_use:
            template_start_time = time.time()

            print(f"\n🔄 " + "=" * 70)
            print(f"📝 TEMPLATE BRANCH: {template}")
            print("=" * 75)
            self.log_timestamp(f"🌿 Starting template branch: {template} ({len(problems_to_use)} problems)")

            # Problem-level timing loop
            for problem in problems_to_use:
                problem_start_time = time.time()
                current_test += 1

                print(f"\n🎯 TEST [{current_test:02d}/{total_combinations:02d}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"    📝 Template: {template}")
                print(f"    🎲 Problem:  {problem}")
                self.log_timestamp(f"🔍 Branch: {template} → {problem} (Test {current_test}/{total_combinations})")

                try:
                    llm_responses, rr_trains = self.run_single_experiment(
                        template, problem, method_module, args, output_dir
                    )

                    # Collect LLM responses
                    self.all_llm_responses.extend(llm_responses)

                    # Analyze results with real-time feedback
                    if not args.dry_run:
                        result_analysis = self.analyze_results(template, problem, rr_trains, args.iterations, args.verbose)
                        self.results_data.append(result_analysis)

                    # Overall success indicator
                    if not args.dry_run and result_analysis['all_correct_rate'] >= 0.8:
                        print("  🎯 Excellent Results!")
                    elif not args.dry_run and result_analysis['all_correct_rate'] >= 0.5:
                        print("  ✅ Good Results!")
                    elif not args.dry_run and result_analysis['at_least_one_correct_rate'] >= 0.5:
                        print("  ⚠️ Partial Success")
                    else:
                        print("  ✓ Test Completed" if args.dry_run else "  ❌ Poor Results")

                except Exception as e:
                    print(f"\n  ❌ EXPERIMENT FAILED: {str(e)}")
                    print(f"     Template: {template}")
                    print(f"     Problem: {problem}")

                    # Always show traceback in verbose mode or when logging to file
                    if args.verbose:
                        traceback.print_exc()

                    # If fail-fast is enabled, stop execution
                    if args.fail_fast:
                        print(f"\n❌ Stopping execution due to --fail-fast flag")
                        print(f"   Failed on test {current_test}/{total_combinations}")
                        print(f"   Template: {template}, Problem: {problem}")
                        # Generate failure summary before exiting
                        self._generate_failure_summary()
                        return

                # Problem-level timing summary
                problem_end_time = time.time()
                problem_duration = problem_end_time - problem_start_time
                self.problem_timings[f"{template}|{problem}"] = problem_duration

                formatted_duration = self.format_duration(problem_duration)
                self.log_timestamp(f"✅ Problem completed: {problem} in {formatted_duration}")

            # Template-level timing summary
            template_end_time = time.time()
            template_duration = template_end_time - template_start_time
            self.template_timings[template] = template_duration

            formatted_duration = self.format_duration(template_duration)
            self.log_timestamp(f"🏁 Template branch completed: {template} in {formatted_duration} ({len(problems_to_use)} problems)")

            # Template performance summary
            if not args.dry_run:
                template_results = [r for r in self.results_data if r['template'] == template]
                if template_results:
                    excellent_count = len([r for r in template_results if r['all_correct_rate'] >= 0.8])
                    good_count = len([r for r in template_results if 0.5 <= r['all_correct_rate'] < 0.8])
                    partial_count = len([r for r in template_results if r['all_correct_rate'] < 0.5 and r['at_least_one_correct_rate'] >= 0.5])
                    poor_count = len([r for r in template_results if r['at_least_one_correct_rate'] < 0.5])

                    avg_all_correct = sum(r['all_correct_rate'] for r in template_results) / len(template_results)
                    avg_partial = sum(r['at_least_one_correct_rate'] for r in template_results) / len(template_results)

                    print(f"\n📊 " + "─" * 60)
                    print(f"📈 TEMPLATE PERFORMANCE SUMMARY: {template}")
                    print("─" * 65)
                    print(f"    🎯 Excellent (≥80%): {excellent_count:2d} problems")
                    print(f"    ✅ Good (50-79%):   {good_count:2d} problems")
                    print(f"    ⚠️  Partial (<50% all, ≥50% some): {partial_count:2d} problems")
                    print(f"    ❌ Poor (<50% any): {poor_count:2d} problems")
                    print(f"    📊 Average success: {avg_all_correct:.1%} all correct, {avg_partial:.1%} partial")
                    print(f"    ⏱️  Total duration: {formatted_duration}")
                    print("─" * 65)

        # Record global end time
        self.global_end_time = time.time()
        total_duration = self.global_end_time - self.global_start_time

        self.log_timestamp("🎉 All tests completed. Generating results...")

        # Generate outputs
        if not args.dry_run and self.results_data:
            self.generate_console_table()

            # Generate ranking analysis and show key insights
            analysis = self.generate_ranking_analysis()
            if analysis:
                print(f"\n🏆 " + "=" * 70)
                print(f"🎯 KEY INSIGHTS & RANKINGS")
                print("=" * 75)

                # Best template overall
                best_template = analysis['template_ranking'][0]
                print(f"🥇 Best Overall Template: {best_template['template']}")
                print(f"   📊 Average success: {best_template['avg_all_correct_rate']:.1%} all correct")
                print(f"   🎯 Excellent on {best_template['excellent_problems']}/{best_template['total_problems']} problems")

                # Top 3 experiments
                print(f"\n🌟 Top 3 Performing Experiments:")
                for i, result in enumerate(analysis['experiment_ranking'][:3], 1):
                    grade = "🎯" if result['all_correct_rate'] >= 0.8 else "✅" if result['all_correct_rate'] >= 0.6 else "⚠️"
                    template_short = result['template'][:30] + "..." if len(result['template']) > 30 else result['template']
                    print(f"   {grade} #{i}: {template_short} + {result['problem']} ({result['all_correct_rate']:.1%})")

                # Problem difficulty insights
                easy_problems = [p for p in analysis['problem_analysis'] if p['difficulty'] == 'EASY']
                hard_problems = [p for p in analysis['problem_analysis'] if p['difficulty'] in ['HARD', 'VERY_HARD']]

                if easy_problems:
                    print(f"\n🟢 Easiest Problems: {', '.join([p['problem'] for p in easy_problems[:3]])}")
                if hard_problems:
                    print(f"🔴 Hardest Problems: {', '.join([p['problem'] for p in hard_problems[:3]])}")

                print("=" * 75)

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

        # Final summary with enhanced aesthetics
        print(f"\n🏆 " + "=" * 70)
        print(f"🎉 FINAL EXPERIMENT SUMMARY 🎉")
        print("=" * 75)
        print(f"⏰ Start Time:  {datetime.fromtimestamp(self.global_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏁 End Time:    {datetime.fromtimestamp(self.global_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Duration:    {self.format_duration(total_duration)}")
        print(f"🧮 Total tests: {total_combinations}")

        if self.results_data:
            successful_tests = len([r for r in self.results_data if r['all_correct_rate'] > 0])
            excellent_tests = len([r for r in self.results_data if r['all_correct_rate'] >= 0.8])
            good_tests = len([r for r in self.results_data if 0.5 <= r['all_correct_rate'] < 0.8])

            print(f"\n📊 PERFORMANCE BREAKDOWN:")
            print("─" * 40)
            print(f"🎯 Excellent (≥80%): {excellent_tests:2d} tests")
            print(f"✅ Good (50-79%):   {good_tests:2d} tests")
            print(f"⚠️  Some success:    {successful_tests - excellent_tests - good_tests:2d} tests")
            print(f"❌ No success:      {len(self.results_data) - successful_tests:2d} tests")
            print(f"📈 Overall success rate: {successful_tests / len(self.results_data):.1%}")
            print(f"⚡ Average time per test: {self.format_duration(total_duration / total_combinations)}")

            # Template timing breakdown
            print(f"\n🕐 TEMPLATE PERFORMANCE:")
            print("─" * 50)
            for template in templates_to_use:
                template_time = self.template_timings.get(template, 0)
                avg_per_problem = template_time / len(problems_to_use)
                template_results = [r for r in self.results_data if r['template'] == template]
                avg_success = sum(r['all_correct_rate'] for r in template_results) / len(template_results) if template_results else 0

                print(f"📝 {template[:40]:40s}")
                print(f"    ⏱️  {self.format_duration(template_time):>10s} (avg: {self.format_duration(avg_per_problem):>8s}/problem)")
                print(f"    📊 {avg_success:>9.1%} average success rate")

        # LLM usage statistics
        if self.all_llm_responses:
            print(f"\n🤖 LLM USAGE STATISTICS:")
            print("─" * 30)
            provider_counts = Counter([response.provider for response in self.all_llm_responses])
            for provider, count in provider_counts.items():
                print(f"    🔗 {provider}: {count} calls")

            token_usages = [response.usage.total_tokens for response in self.all_llm_responses]
            print(f"    🎯 Max tokens: {max(token_usages):,}")
            print(f"    📊 Median tokens: {sorted(token_usages)[len(token_usages)//2]:,}")
            print(f"    📈 Total tokens: {sum(token_usages):,}")

        print("=" * 75)
        self.log_timestamp("🎉 EXPERIMENT COMPLETED! 🎉")

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