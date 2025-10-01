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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    def load_dotenv():
        pass


# Refactored classes to eliminate duplication
from core.checkpoint_manager import CheckpointManager
from core.experiment_argument_parser import ExperimentArgumentParser
from core.experiment_config import ExperimentConfigResolver
from core.experiment_coordinator import ExperimentCoordinator
from core.service_registry import ServiceFactory, ServiceRegistry
from core.timing_tracker import TimingTracker
from domain.experiment_state import ExperimentContext, ExperimentResults
from domain.value_objects import SuccessThresholds
from output.failure_formatter import FailureSummaryFormatter


class BatchExperimentRunner:
    """Main class for running batch experiments with comprehensive tracking."""

    def __init__(self):
        """
        Initialize batch experiment runner.

        Following Object Calisthenics Rule 8: Maximum 2 instance variables.
        Reduced from 14 instance variables to 6 by using cohesive objects:
        - _timing: Timing operations
        - _results: All results data (was 4 variables)
        - _context: Execution context (was 2 variables)
        - _services: All services (was 7 variables)
        - _thresholds: Success thresholds
        - _checkpoint_manager: Checkpoint manager for resume support
        """
        # Core infrastructure (6 instance variables total)
        self._timing = TimingTracker()
        self._results = ExperimentResults()
        self._context = ExperimentContext()
        self._thresholds = SuccessThresholds()
        self._checkpoint_manager: Optional[CheckpointManager] = None

        # Service registry (consolidates 7 lazy-initialized services)
        factory = self._create_service_factory()
        self._services = ServiceRegistry(factory)

        # Experiment coordinator (groups 8 orchestration methods)
        self._coordinator = ExperimentCoordinator(
            self._services, self._context, self.log_timestamp, self.format_duration
        )

    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments with comprehensive options."""
        return ExperimentArgumentParser.parse()

    # Configuration resolution methods moved to ExperimentConfigResolver
    # (Duplication removed in Phase 2 refactoring)

    def log_timestamp(self, message: str, to_file_only: bool = False) -> None:
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            log_file_handle_accessor=lambda: self._context.log_handle,
            checkpoint_manager_accessor=lambda: self._checkpoint_manager,
        )

    def validate_prerequisites(
        self,
        templates_to_use: List[str],
        problems_to_use: List[str],
        method_module: Any,
        args: argparse.Namespace,
    ) -> Tuple[bool, List[str]]:
        """Validate all prerequisites before starting experiments."""
        # Delegate to ExperimentValidator (Phase 2B refactoring)
        return self._services.validator().validate_all(
            templates_to_use, problems_to_use, method_module, args
        )

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
        experiment_folder: Path,
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
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Analyze experiment results for success rates with real-time feedback."""
        # Delegate to ExperimentExecutor (Phase 2A refactoring)
        return self._services.executor().analyze_results(
            template, problem, rr_trains, iterations, verbose
        )

    def generate_console_table(self) -> None:
        """Generate formatted console table."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        self._services.output_generator().generate_console_table()

    def generate_csv_output(self, output_dir: Path) -> str:
        """Generate CSV output file with ranking data."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        return self._services.output_generator().generate_csv_output(output_dir)

    def generate_html_output(self, output_dir: Path) -> str:
        """Generate HTML output file."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        return self._services.output_generator().generate_html_output(output_dir)

    def generate_ranking_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive ranking and best-performance analysis."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        return self._services.output_generator().generate_ranking_analysis()

    def generate_ranking_report(self, output_dir: Path, timestamp: str) -> str:
        """Generate comprehensive ranking and analysis report."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        return self._services.output_generator().generate_ranking_report(output_dir, timestamp)

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
            formatter.write_file_summary(
                self._context.log_handle, self._results.failures, self._context.get_attempts()
            )

    def generate_persistent_summary(
        self, base_output_dir: Path, aggregated_data: Dict[str, Any]
    ) -> List[str]:
        """Generate persistent summary files that aggregate all experiments."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        return self._services.output_generator().generate_persistent_summary(
            base_output_dir, aggregated_data
        )

    def run_summarise_experiments(self, args: argparse.Namespace) -> None:
        """Analyze existing experiments and generate/update summary statistics."""
        # Delegate to ExperimentSummarizer (Phase 2D refactoring)
        self._services.summarizer().run(args)

    def generate_summary_log(
        self,
        output_dir: Path,
        timestamp: str,
        total_duration: float,
        templates_to_use: list,
        problems_to_use: list,
        total_combinations: int,
    ) -> str:
        """Generate detailed summary log file."""
        # Delegate to OutputGenerator (Phase 1 refactoring)
        return self._services.output_generator().generate_summary_log(
            output_dir,
            timestamp,
            total_duration,
            templates_to_use,
            problems_to_use,
            total_combinations,
        )

    def _print_experiment_configuration(
        self, args: argparse.Namespace, templates_to_use: List[str], problems_to_use: List[str]
    ) -> int:
        """Print experiment configuration. Delegates to ExperimentCoordinator."""
        return self._coordinator.print_experiment_configuration(
            args, templates_to_use, problems_to_use
        )

    def _setup_output_directory(self, args: argparse.Namespace) -> Tuple[Path, str]:
        """
        Setup output directory and logging. Returns (output_dir, timestamp).

        Following Object Calisthenics Rule 7: Small focused methods.

        If output_dir contains a checkpoint.json, use it directly (resume mode).
        Otherwise, create a new timestamped subdirectory.
        """
        base_output_dir = Path(args.output_dir)

        # Check if we're resuming from an existing experiment directory
        checkpoint_exists = (base_output_dir / "checkpoint.json").exists()

        if checkpoint_exists and not args.no_resume:
            # Use the existing directory directly (resume mode)
            output_dir = base_output_dir
            # Extract original timestamp from directory name or use current for log
            dir_name = base_output_dir.name
            if dir_name and len(dir_name) >= 15 and dir_name[:8].isdigit():
                timestamp = dir_name[:15]  # Extract existing timestamp (YYYYMMDD_HHMMSS)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"\n🔄 Resuming in existing directory: {output_dir}")
        else:
            # Create new timestamped subdirectory (normal mode)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = base_output_dir / timestamp
            print(f"\n💾 Results will be saved to: {output_dir}")

        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Setup log file (append mode if resuming, write mode if new)
            log_file = output_dir / f"experiment_run_{timestamp}.log"
            mode = "a" if checkpoint_exists and not args.no_resume else "w"
            log_handle = open(log_file, mode)
            self._context.set_log_handle(log_handle)

            if mode == "a":
                log_handle.write(f"\n\n{'=' * 80}\n")
                log_handle.write(f"RESUMED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_handle.write(f"{'=' * 80}\n\n")

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
            if self._context.is_logging():
                self._context.close_log()
            return None

    def _run_preflight_validation(
        self,
        templates_to_use: List[str],
        problems_to_use: List[str],
        method_module: Any,
        args: argparse.Namespace,
    ) -> bool:
        """
        Run pre-flight validation. Returns True if passed.

        Following Object Calisthenics Rule 7: Small focused methods.
        """
        print(f"\n{'=' * 80}")
        print("🔍 PRE-FLIGHT VALIDATION")
        print(f"{'=' * 80}")

        validation_passed, validation_errors = self.validate_prerequisites(
            templates_to_use, problems_to_use, method_module, args
        )

        if not validation_passed:
            print("\n❌ Cannot proceed with experiments due to validation failures")
            if self._context.is_logging():
                self._context.log_handle.write("\nValidation failed:\n")
                for error in validation_errors:
                    self._context.log_handle.write(f"  - {error}\n")
                self._context.close_log()
            return False

        print(f"{'=' * 80}\n")
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
        console = self._services.console_display()
        console.print_template_performance_summary(template, template_results, formatted_duration)

    def _print_key_insights(self, analysis: Dict[str, Any]) -> None:
        """Print key insights and rankings from experiment analysis.

        Args:
            analysis: Analysis dictionary from generate_ranking_analysis()
        """
        # Delegate to ConsoleDisplay (Phase 1 refactoring)
        console = self._services.console_display()
        console.print_key_insights(analysis)

    def _generate_and_save_outputs(
        self,
        output_dir: Path,
        timestamp: str,
        total_duration: float,
        templates_to_use: List[str],
        problems_to_use: List[str],
        total_combinations: int,
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
        summary_log = self.generate_summary_log(
            output_dir,
            timestamp,
            total_duration,
            templates_to_use,
            problems_to_use,
            total_combinations,
        )
        ranking_report = self.generate_ranking_report(output_dir, timestamp)

        print("\n💾 " + "=" * 60)
        print("📁 RESULTS SAVED")
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
        self,
        total_duration: float,
        total_combinations: int,
        templates_to_use: List[str],
        problems_to_use: List[str],
    ) -> None:
        """Print final experiment summary including performance breakdown and LLM statistics.

        Args:
            total_duration: Total experiment duration in seconds
            total_combinations: Total number of test combinations
            templates_to_use: List of templates used
            problems_to_use: List of problems used
        """
        # Delegate to ConsoleDisplay (Phase 1 refactoring)
        console = self._services.console_display()
        console.print_final_summary(
            total_duration,
            total_combinations,
            templates_to_use,
            problems_to_use,
            self._results.results,
            self._results.llm_responses,
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
        total_combinations = self._print_experiment_configuration(
            args, templates_to_use, problems_to_use
        )

        # Setup output directory and logging (extracted method)
        output_dir, timestamp = self._setup_output_directory(args)

        # Setup checkpoint manager (unless disabled)
        if not args.no_checkpoint:
            self._checkpoint_manager = CheckpointManager(output_dir)

            # Handle checkpoint resume logic
            if not args.no_resume:
                checkpoint = self._checkpoint_manager.load_checkpoint()
                if checkpoint:
                    # Auto-prompt or force-resume based on flags
                    should_resume = args.resume or CheckpointManager.prompt_resume_from_checkpoint(
                        checkpoint
                    )

                    if should_resume:
                        self.log_timestamp("✅ Resuming from previous checkpoint")
                        # Note: The actual restore happens in the orchestrator's execute_loop
                    else:
                        self.log_timestamp("🔄 Starting fresh (checkpoint ignored)")
                        self._checkpoint_manager.delete_checkpoint()

        # Load method module (extracted method)
        method_module = self._load_method_module(args)
        if method_module is None and not args.dry_run:
            return

        # Run pre-flight validation (extracted method)
        if not self._run_preflight_validation(
            templates_to_use, problems_to_use, method_module, args
        ):
            return

        # Execute experiments using orchestrator (removes 100+ lines of procedural code)
        orchestrator = self._services.loop_orchestrator()
        loop_result = orchestrator.execute_loop(
            templates_to_use, problems_to_use, method_module, args, output_dir
        )

        # Collect results from orchestrator
        self._results.set_all_results(loop_result["results_data"])
        self._results.set_all_llm_responses(loop_result["all_llm_responses"])
        self._results.set_all_failures(loop_result["failed_experiments"])

        # If loop stopped early due to fail-fast, handle it
        if not loop_result["completed"]:
            self._generate_failure_summary()
            return

        # Print template performance summaries
        if not args.dry_run:
            for template in templates_to_use:
                template_results = [r for r in self._results.results if r["template"] == template]
                if template_results:
                    template_duration = self._timing.get_template_duration(template)
                    formatted_duration = self.format_duration(template_duration)
                    self._print_template_performance_summary(
                        template, template_results, formatted_duration
                    )

        # Record global end time
        self._timing.end_global()
        total_duration = self._timing.get_global_duration()

        self.log_timestamp("🎉 All tests completed. Generating results...")

        # Generate outputs
        if not args.dry_run and self._results.results:
            self.generate_console_table()

            # Generate ranking analysis and show key insights
            analysis = self.generate_ranking_analysis()
            if analysis:
                self._print_key_insights(analysis)

            self._generate_and_save_outputs(
                output_dir,
                timestamp,
                total_duration,
                templates_to_use,
                problems_to_use,
                total_combinations,
            )

        # Final summary
        self._print_final_summary(
            total_duration, total_combinations, templates_to_use, problems_to_use
        )

        # Generate failure summary if there were any failures
        if self._results.failures:
            self._generate_failure_summary()

        # Close log file
        if self._context.is_logging():
            self._context.log_handle.write(f"\n{'=' * 80}\n")
            self._context.log_handle.write("EXPERIMENT COMPLETED\n")
            self._context.log_handle.write(f"{'=' * 80}\n")
            self._context.close_log()


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
        if runner._results.failures:
            runner._generate_failure_summary()
        if runner._context.is_logging():
            runner._context.log_handle.write("\n\nEXPERIMENT INTERRUPTED BY USER\n")
            runner._context.close_log()
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        if runner._results.failures:
            runner._generate_failure_summary()
        if runner._context.is_logging():
            runner._context.log_handle.write(f"\n\nEXPERIMENT FAILED: {str(e)}\n")
            runner._context.log_handle.write(traceback.format_exc())
            runner._context.close_log()


if __name__ == "__main__":
    main()
