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
        self.experiment_timestamp: str = ""  # Will be set when experiment starts

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
            """)

        # Core experiment parameters
        parser.add_argument("-m", "--method", type=str, default="method1_text_prompt",
                          help="Method module to use (default: %(default)s)")
        parser.add_argument("--model", type=str, default="openrouter/deepseek/deepseek-chat-v3-0324",
                          help="Model name to use (default: %(default)s)")
        parser.add_argument("-i", "--iterations", type=int, default=1,
                          help="Number of iterations per test (default: %(default)s)")

        # Template and problem selection
        parser.add_argument("-t", "--templates", type=str,
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

    def log_timestamp(self, message: str) -> None:
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"🕒 [{timestamp}] {message}")

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

    def run_single_experiment(
        self,
        template: str,
        problem: str,
        method_module: Any,
        args: argparse.Namespace
    ) -> Tuple[List[Any], List[Any]]:
        """Run a single experiment with timing."""
        self.log_timestamp(f"Starting individual test: {template} + {problem}")

        if args.dry_run:
            print(f"[DRY-RUN] Would run {template} with {problem}")
            return [], []

        individual_start_time = time.time()

        try:
            # Setup for this specific experiment
            _, _, logger, _, db_filename, _ = do_first_setup()

            # Load the specific problem
            problems = utils.get_examples(problem)

            # Set logger for the method module
            method_module.logger = logger

            # Run the experiment
            llm_responses, rr_trains = method_module.run_experiment_for_iterations(
                db_filename,
                model=args.model,
                iterations=args.iterations,
                problems=problems,
                template_name=template,
            )

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

            formatted_duration = self.format_duration(individual_duration)
            self.log_timestamp(f"Individual test failed after {formatted_duration}: {str(e)}")

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

            if verbose:
                print(f"      Iteration {i}: {status_icon} {status_text}")

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
        for result in self.results_data:
            template_short = result['template'][:35] + "..." if len(result['template']) > 35 else result['template']
            test_time = self.format_duration(result['individual_duration'])

            # Determine grade emoji
            if result['all_correct_rate'] >= 0.8:
                grade = "🎯 A"
            elif result['all_correct_rate'] >= 0.6:
                grade = "✅ B"
            elif result['all_correct_rate'] >= 0.4:
                grade = "⚠️ C"
            elif result['at_least_one_correct_rate'] >= 0.5:
                grade = "🔶 D"
            else:
                grade = "❌ F"

            print(f"{template_short:<40} {result['problem']:<12} {result['all_correct_rate']:<7.1%} "
                  f"{result['at_least_one_correct_rate']:<8.1%} {test_time:<10} {grade:<8}")

        print("─" * 95)
        print(f"Legend: 🎯A(≥80%) ✅B(60-79%) ⚠️C(40-59%) 🔶D(partial) ❌F(<40%)")
        print("═" * 95)

    def generate_csv_output(self, output_dir: Path) -> str:
        """Generate CSV output file."""
        if not self.results_data:
            return ""

        if pd is None:
            print("Warning: pandas not available, skipping CSV output")
            return ""

        df = pd.DataFrame(self.results_data)

        # Add timing columns
        df['individual_duration_formatted'] = df['individual_duration'].apply(self.format_duration)
        df['problem_duration_formatted'] = df['problem_duration'].apply(self.format_duration)

        csv_file = output_dir / f"batch_results_{self.experiment_timestamp}.csv"

        df.to_csv(csv_file, index=False)
        return str(csv_file)

    def generate_html_output(self, output_dir: Path) -> str:
        """Generate HTML output file."""
        if not self.results_data:
            return ""

        html_file = output_dir / f"batch_results_{self.experiment_timestamp}.html"

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

    def run_batch_experiments(self, args: argparse.Namespace) -> None:
        """Main execution method with comprehensive tracking."""
        # Record global start time and set experiment timestamp
        self.global_start_time = time.time()
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

        # Create timestamp-based output directory
        base_output_dir = Path(args.output_dir)
        output_dir = base_output_dir / self.experiment_timestamp
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n💾 Results will be saved to: {output_dir}")

            # Create experiment summary log with better formatting
            summary_log = output_dir / f"batch_summary_{self.experiment_timestamp}.log"
            with open(summary_log, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("🧠  ARC-AGI BATCH EXPERIMENT SUMMARY  🧠\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"📅 Experiment started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("📋 CONFIGURATION:\n")
                f.write("─" * 40 + "\n")
                f.write(f"  🔧 Method: {args.method}\n")
                f.write(f"  🤖 Model: {args.model}\n")
                f.write(f"  🔄 Iterations: {args.iterations}\n\n")
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

        # Load method module
        if not args.dry_run:
            try:
                method_module = importlib.import_module(args.method)
                print(f"Using module: {method_module.__file__}")
                print(f"Using entry point: {method_module.run_experiment_for_iterations.__name__}")
            except ImportError as e:
                print(f"Error importing method module '{args.method}': {e}")
                return
        else:
            method_module = None

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
                        template, problem, method_module, args
                    )

                    # Collect LLM responses
                    self.all_llm_responses.extend(llm_responses)

                    # Analyze results with real-time feedback
                    if not args.dry_run:
                        result_analysis = self.analyze_results(template, problem, rr_trains, args.iterations, args.verbose)
                        self.results_data.append(result_analysis)

                        # Log individual test result to summary with better formatting
                        summary_log = output_dir / f"batch_summary_{self.experiment_timestamp}.log"
                        with open(summary_log, 'a') as f:
                            success_icon = "🎯" if result_analysis['all_correct_rate'] >= 0.8 else "✅" if result_analysis['all_correct_rate'] >= 0.5 else "⚠️" if result_analysis['at_least_one_correct_rate'] >= 0.5 else "❌"
                            f.write(f"{success_icon} Test {current_test:02d}: {template} + {problem}\n")
                            f.write(f"    📊 Results: {result_analysis['all_correct']}/{args.iterations} all correct ({result_analysis['all_correct_rate']:.1%}), ")
                            f.write(f"{result_analysis['at_least_one_correct']}/{args.iterations} partial ({result_analysis['at_least_one_correct_rate']:.1%})\n")
                            f.write(f"    ⏱️  Duration: {self.format_duration(result_analysis['individual_duration'])}\n\n")

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
                    print(f"  ✗ Failed: {str(e)}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()

                    # Log failure to summary with better formatting
                    if not args.dry_run:
                        summary_log = output_dir / f"batch_summary_{self.experiment_timestamp}.log"
                        with open(summary_log, 'a') as f:
                            f.write(f"💥 Test {current_test:02d}: {template} + {problem}\n")
                            f.write(f"    ❌ Status: FAILED\n")
                            f.write(f"    🚨 Error: {str(e)}\n\n")

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

            csv_file = self.generate_csv_output(output_dir)
            html_file = self.generate_html_output(output_dir)

            # Add final summary to log with enhanced formatting
            summary_log = output_dir / f"batch_summary_{self.experiment_timestamp}.log"
            with open(summary_log, 'a') as f:
                f.write("=" * 80 + "\n")
                f.write("🎉  EXPERIMENT COMPLETED  🎉\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
            print(f"    📝 Summary Log: {summary_log.name}")
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
        self.log_timestamp("🎉 EXPERIMENT COMPLETED SUCCESSFULLY! 🎉")


def main():
    """Main entry point."""
    runner = BatchExperimentRunner()
    args = runner.parse_arguments()

    try:
        runner.run_batch_experiments(args)
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nExperiment failed with error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()