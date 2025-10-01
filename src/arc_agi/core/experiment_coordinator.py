"""
Experiment coordination module.

Handles the orchestration of experiment setup, execution coordination,
and output generation. Groups related orchestration methods that were
previously scattered across BatchExperimentRunner.

Following Object Calisthenics: Small focused class with clear responsibility.
"""

import argparse
import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.service_registry import ServiceRegistry
from domain.experiment_state import ExperimentContext


class ExperimentCoordinator:
    """
    Coordinates experiment setup, validation, and output generation.

    Groups together methods that orchestrate different phases of experiment
    execution. Reduces complexity in BatchExperimentRunner by handling all
    coordination tasks in one cohesive object.

    Responsibilities:
    - Print experiment configuration
    - Setup output directories and logging
    - Load and validate method modules
    - Run pre-flight validation
    - Generate and save all outputs
    - Print performance summaries

    Follows Single Responsibility Principle: Only handles coordination tasks.
    """

    def __init__(
        self,
        services: ServiceRegistry,
        context: ExperimentContext,
        log_callback: Callable[[str], None],
        format_duration_callback: Callable[[float], str],
    ):
        """Initialize coordinator with required services.

        Args:
            services: ServiceRegistry for accessing services
            context: ExperimentContext for logging and state
            log_callback: Callback for logging with timestamps
            format_duration_callback: Callback to format durations
        """
        self.services = services
        self.context = context
        self.log_timestamp = log_callback
        self.format_duration = format_duration_callback

    def print_experiment_configuration(
        self,
        args: argparse.Namespace,
        templates_to_use: List[str],
        problems_to_use: List[str],
    ) -> int:
        """
        Print experiment configuration and return total combinations.

        Args:
            args: Command line arguments
            templates_to_use: List of templates to test
            problems_to_use: List of problems to test

        Returns:
            Total number of test combinations
        """
        total_combinations = (
            self.services.console_display().print_experiment_configuration(
                args, templates_to_use, problems_to_use
            )
        )
        self.log_timestamp(
            f"✅ Configuration complete. Starting {total_combinations} test combinations."
        )
        return total_combinations

    def setup_output_directory(self, args: argparse.Namespace) -> Tuple[Path, str]:
        """
        Setup output directory and logging.

        Args:
            args: Command line arguments with output_dir and dry_run flags

        Returns:
            Tuple of (output_dir, timestamp)
        """
        base_output_dir = Path(args.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output_dir / timestamp

        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n💾 Results will be saved to: {output_dir}")

            # Setup log file
            log_file = output_dir / f"experiment_run_{timestamp}.log"
            self.context.open_log_file(log_file)
            self.log_timestamp(f"Logging to: {log_file}")
            print(f"📝 Detailed log: {log_file.name}")

        return output_dir, timestamp

    def load_method_module(self, args: argparse.Namespace) -> Optional[Any]:
        """
        Load and validate method module.

        Args:
            args: Command line arguments with method name

        Returns:
            Loaded method module or None if error
        """
        if args.dry_run:
            return None

        try:
            method_module = importlib.import_module(args.method)
            print(f"Using module: {method_module.__file__}")
            print(
                f"Using entry point: {method_module.run_experiment_for_iterations.__name__}"
            )
            return method_module
        except ImportError as e:
            print(f"Error importing method module '{args.method}': {e}")
            self.context.close_log_file()
            return None

    def run_preflight_validation(
        self,
        templates_to_use: List[str],
        problems_to_use: List[str],
        method_module: Any,
        args: argparse.Namespace,
        validate_callback: Callable,
    ) -> bool:
        """
        Run pre-flight validation checks.

        Args:
            templates_to_use: List of templates to validate
            problems_to_use: List of problems to validate
            method_module: Method module to validate
            args: Command line arguments
            validate_callback: Callback to perform actual validation

        Returns:
            True if validation passed, False otherwise
        """
        print(f"\n{'=' * 80}")
        print("🔍 PRE-FLIGHT VALIDATION")
        print(f"{'=' * 80}")

        validation_passed, validation_errors = validate_callback(
            templates_to_use, problems_to_use, method_module, args
        )

        if not validation_passed:
            print("\n❌ Cannot proceed with experiments due to validation failures")
            if self.context.is_logging():
                self.context.write_to_log("\nValidation failed:")
                for error in validation_errors:
                    self.context.write_to_log(f"  - {error}")
                self.context.close_log_file()
            return False

        print(f"{'=' * 80}\n")
        return True

    def generate_and_save_outputs(
        self,
        output_dir: Path,
        timestamp: str,
        total_duration: float,
        templates_to_use: List[str],
        problems_to_use: List[str],
        total_combinations: int,
        generate_csv_callback: Callable,
        generate_html_callback: Callable,
        generate_summary_log_callback: Callable,
        generate_ranking_report_callback: Callable,
    ) -> None:
        """
        Generate and save all output files.

        Args:
            output_dir: Output directory path
            timestamp: Timestamp string for filenames
            total_duration: Total experiment duration
            templates_to_use: List of templates used
            problems_to_use: List of problems used
            total_combinations: Total number of test combinations
            generate_csv_callback: Callback to generate CSV
            generate_html_callback: Callback to generate HTML
            generate_summary_log_callback: Callback to generate summary log
            generate_ranking_report_callback: Callback to generate ranking report
        """
        csv_file = generate_csv_callback(output_dir)
        html_file = generate_html_callback(output_dir)
        summary_log = generate_summary_log_callback(
            output_dir,
            timestamp,
            total_duration,
            templates_to_use,
            problems_to_use,
            total_combinations,
        )
        ranking_report = generate_ranking_report_callback(output_dir, timestamp)

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

    def print_template_performance_summary(
        self,
        template: str,
        template_results: List[Dict[str, Any]],
        formatted_duration: str,
    ) -> None:
        """
        Print performance summary for a completed template.

        Args:
            template: Template name
            template_results: List of result dictionaries for this template
            formatted_duration: Formatted duration string
        """
        self.services.console_display().print_template_performance_summary(
            template, template_results, formatted_duration
        )

    def print_key_insights(self, analysis: Dict[str, Any]) -> None:
        """
        Print key insights and rankings from experiment analysis.

        Args:
            analysis: Analysis dictionary from generate_ranking_analysis()
        """
        self.services.console_display().print_key_insights(analysis)

    def print_final_summary(
        self,
        total_duration: float,
        total_combinations: int,
        templates_to_use: List[str],
        problems_to_use: List[str],
        results_data: List[Dict[str, Any]],
        llm_responses: List[Any],
    ) -> None:
        """
        Print final experiment summary.

        Args:
            total_duration: Total experiment duration in seconds
            total_combinations: Total number of test combinations
            templates_to_use: List of templates used
            problems_to_use: List of problems used
            results_data: All experiment results
            llm_responses: All LLM responses
        """
        self.services.console_display().print_final_summary(
            total_duration,
            total_combinations,
            templates_to_use,
            problems_to_use,
            results_data,
            llm_responses,
        )
        self.log_timestamp("🎉 EXPERIMENT COMPLETED! 🎉")
