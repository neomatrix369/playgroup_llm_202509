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

    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments with comprehensive options."""
        parser = argparse.ArgumentParser(
            description="Comprehensive batch experiment runner for ARC-AGI testing",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run all templates and problems
  python run_batch_experiments.py

  # Run specific templates and problems with 5 iterations
  python run_batch_experiments.py -t "baseline_justjson_enhanced.j2,reflexion_enhanced.j2" -p "0d3d703e,08ed6ac7" -i 5

  # Dry run with verbose output
  python run_batch_experiments.py --dry-run --verbose

  # Use different method module
  python run_batch_experiments.py --method method2_reflexion --model openrouter/deepseek/deepseek-chat-v3-0324
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
        print(f"[{timestamp}] {message}")

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
        iterations: int
    ) -> Dict[str, Any]:
        """Analyze experiment results for success rates."""
        all_correct = 0
        at_least_one_correct = 0

        for rr_train in rr_trains:
            ran_all_train_problems_correctly = rr_train[0].transform_ran_and_matched_for_all_inputs
            ran_at_least_one_train_problem_correctly = rr_train[0].transform_ran_and_matched_at_least_once

            if ran_all_train_problems_correctly:
                all_correct += 1
            if ran_at_least_one_train_problem_correctly:
                at_least_one_correct += 1

        return {
            "template": template,
            "problem": problem,
            "total_runs": iterations,
            "all_correct": all_correct,
            "at_least_one_correct": at_least_one_correct,
            "all_correct_rate": all_correct / iterations,
            "at_least_one_correct_rate": at_least_one_correct / iterations,
            "individual_duration": self.individual_timings.get(f"{template}|{problem}", 0),
            "problem_duration": self.problem_timings.get(f"{template}|{problem}", 0),
        }

    def generate_console_table(self) -> None:
        """Generate formatted console table."""
        if not self.results_data:
            return

        print("\n" + "=" * 120)
        print("DETAILED RESULTS TABLE")
        print("=" * 120)

        # Header
        print(f"{'Template':<35} {'Problem':<12} {'All_Correct':<12} {'At_Least_One':<12} {'Test_Time':<12} {'Branch_Time':<12}")
        print("-" * 120)

        # Data rows
        for result in self.results_data:
            template_short = result['template'][:32] + "..." if len(result['template']) > 32 else result['template']
            test_time = self.format_duration(result['individual_duration'])
            branch_time = self.format_duration(result['problem_duration'])

            print(f"{template_short:<35} {result['problem']:<12} {result['all_correct_rate']:<12.2f} "
                  f"{result['at_least_one_correct_rate']:<12.2f} {test_time:<12} {branch_time:<12}")

        print("-" * 120)

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = output_dir / f"batch_results_{timestamp}.csv"

        df.to_csv(csv_file, index=False)
        return str(csv_file)

    def generate_html_output(self, output_dir: Path) -> str:
        """Generate HTML output file."""
        if not self.results_data:
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    def run_batch_experiments(self, args: argparse.Namespace) -> None:
        """Main execution method with comprehensive tracking."""
        # Record global start time
        self.global_start_time = time.time()

        print("=== ARC-AGI Batch Experiment Runner ===")
        self.log_timestamp("EXPERIMENT STARTED")

        # Resolve selections
        self.log_timestamp("Resolving template and problem selections...")
        templates_to_use = self.resolve_templates(args.templates)
        problems_to_use = self.resolve_problems(args.problems)

        print(f"Method Module: {args.method}")
        print(f"Model: {args.model}")
        print(f"Iterations per test: {args.iterations}")
        print(f"Templates to test ({len(templates_to_use)}):")
        for i, template in enumerate(templates_to_use):
            print(f"  {i}: {template}")

        print(f"Problems to test ({len(problems_to_use)}):")
        for i, problem in enumerate(problems_to_use):
            print(f"  {i}: {problem}")

        # Calculate total combinations
        total_combinations = len(templates_to_use) * len(problems_to_use)
        print(f"Total test combinations: {total_combinations}")
        self.log_timestamp(f"Configuration complete. Starting {total_combinations} test combinations.")

        # Create output directory
        output_dir = Path(args.output_dir)
        if not args.dry_run:
            output_dir.mkdir(exist_ok=True)

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
            self.log_timestamp(f"Starting template branch: {template} ({len(problems_to_use)} problems)")

            # Problem-level timing loop
            for problem in problems_to_use:
                problem_start_time = time.time()
                current_test += 1

                print(f"\n[{current_test}/{total_combinations}] Testing: {template} with {problem}")
                self.log_timestamp(f"Branch: {template} → {problem} (Test {current_test}/{total_combinations})")

                try:
                    llm_responses, rr_trains = self.run_single_experiment(
                        template, problem, method_module, args
                    )

                    # Collect LLM responses
                    self.all_llm_responses.extend(llm_responses)

                    # Analyze results
                    if not args.dry_run:
                        result_analysis = self.analyze_results(template, problem, rr_trains, args.iterations)
                        self.results_data.append(result_analysis)

                    print("  ✓ Success")

                except Exception as e:
                    print(f"  ✗ Failed: {str(e)}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()

                # Problem-level timing summary
                problem_end_time = time.time()
                problem_duration = problem_end_time - problem_start_time
                self.problem_timings[f"{template}|{problem}"] = problem_duration

                formatted_duration = self.format_duration(problem_duration)
                self.log_timestamp(f"Problem completed: {problem} in {formatted_duration}")

            # Template-level timing summary
            template_end_time = time.time()
            template_duration = template_end_time - template_start_time
            self.template_timings[template] = template_duration

            formatted_duration = self.format_duration(template_duration)
            self.log_timestamp(f"Template branch completed: {template} in {formatted_duration} ({len(problems_to_use)} problems)")

        # Record global end time
        self.global_end_time = time.time()
        total_duration = self.global_end_time - self.global_start_time

        self.log_timestamp("All tests completed. Generating results...")

        # Generate outputs
        if not args.dry_run and self.results_data:
            self.generate_console_table()

            csv_file = self.generate_csv_output(output_dir)
            html_file = self.generate_html_output(output_dir)

            print(f"\nResults saved to:")
            if csv_file:
                print(f"  CSV: {csv_file}")
            if html_file:
                print(f"  HTML: {html_file}")

        # Final summary
        print(f"\n=== Batch Experiment Summary ===")
        print(f"Start Time: {datetime.fromtimestamp(self.global_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {datetime.fromtimestamp(self.global_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {self.format_duration(total_duration)}")
        print(f"Total tests: {total_combinations}")

        if self.results_data:
            successful_tests = len([r for r in self.results_data if r['all_correct_rate'] > 0])
            print(f"Tests with some success: {successful_tests}")
            print(f"Success rate: {successful_tests / len(self.results_data):.1%}")
            print(f"Average time per test: {self.format_duration(total_duration / total_combinations)}")

            # Template timing breakdown
            print(f"\n=== Template Timing Breakdown ===")
            for template in templates_to_use:
                template_time = self.template_timings.get(template, 0)
                avg_per_problem = template_time / len(problems_to_use)
                print(f"  {template}: {self.format_duration(template_time)} "
                      f"(avg: {self.format_duration(avg_per_problem)} per problem)")

        # LLM usage statistics
        if self.all_llm_responses:
            provider_counts = Counter([response.provider for response in self.all_llm_responses])
            print(f"\nProvider counts: {dict(provider_counts)}")

            token_usages = [response.usage.total_tokens for response in self.all_llm_responses]
            print(f"Max token usage: {max(token_usages)}")
            print(f"Median token usage: {sorted(token_usages)[len(token_usages)//2]}")

        self.log_timestamp("EXPERIMENT COMPLETED")


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