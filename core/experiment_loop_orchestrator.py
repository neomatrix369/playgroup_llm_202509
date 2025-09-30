"""
Experiment loop orchestration module.

Extracts the procedural experiment loop from run_all_problems.py into a
cohesive, object-oriented design. Removes primitive obsession by using
value objects and proper encapsulation.

Following SOLID principles:
- Single Responsibility: Only handles the experiment loop execution
- Dependency Inversion: Depends on abstractions (callbacks)
- Open/Closed: Easy to extend loop behavior without modification

Includes checkpoint/resume support for interrupted experiments.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple, Set
import argparse
import traceback

from domain.value_objects import SuccessThresholds
from core.timing_tracker import TimingTracker
from core.checkpoint_manager import CheckpointManager, ExperimentCheckpoint


class ProgressTracker:
    """
    Encapsulates progress tracking state.
    
    Removes primitive obsession: replaces naked int counters with
    a proper object that knows how to format and display progress.
    """
    
    def __init__(self, total_combinations: int):
        self.current_test: int = 0
        self.total_combinations: int = total_combinations
    
    def increment(self) -> int:
        """Increment and return current test number."""
        self.current_test += 1
        return self.current_test
    
    def format_progress(self) -> str:
        """Format progress as '01/20' style string."""
        return f"{self.current_test:02d}/{self.total_combinations:02d}"
    
    def is_complete(self) -> bool:
        """Check if all tests are complete."""
        return self.current_test >= self.total_combinations


class ExperimentLoopOrchestrator:
    """
    Orchestrates the main experiment execution loop.
    
    Converts the procedural loop from run_all_problems.py into a
    cohesive object that encapsulates:
    - Loop execution logic
    - Progress tracking
    - Error handling
    - Result collection
    - Performance feedback
    
    This removes the 200+ line procedural method and replaces it
    with a well-structured, testable object.
    """
    
    def __init__(
        self,
        timing_tracker: TimingTracker,
        log_callback: Callable[[str], None],
        format_duration_callback: Callable[[float], str],
        run_experiment_callback: Callable,
        analyze_results_callback: Callable,
        checkpoint_manager: Optional[CheckpointManager] = None,
        success_thresholds: SuccessThresholds = SuccessThresholds()
    ):
        """Initialize the experiment loop orchestrator.
        
        Args:
            timing_tracker: TimingTracker instance for timing operations
            log_callback: Function to log timestamped messages
            format_duration_callback: Function to format duration in human-readable format
            run_experiment_callback: Callback to run a single experiment
            analyze_results_callback: Callback to analyze experiment results
            checkpoint_manager: Optional checkpoint manager for resume support
            success_thresholds: Thresholds for success evaluation (removes magic numbers)
        """
        self.timing = timing_tracker
        self.log_timestamp = log_callback
        self.format_duration = format_duration_callback
        self.run_experiment = run_experiment_callback
        self.analyze_results = analyze_results_callback
        self.checkpoint_manager = checkpoint_manager
        self.thresholds = success_thresholds
        
        # State to be collected during loop
        self.results_data: List[Dict[str, Any]] = []
        self.all_llm_responses: List[Any] = []
        self.failed_experiments: List[Dict[str, Any]] = []
        self.completed_experiments: Set[Tuple[str, str]] = set()
        
        # Timing for checkpoint
        self.start_time: float = time.time()
        self.current_templates: List[str] = []
        self.current_problems: List[str] = []
        self.current_args: Optional[argparse.Namespace] = None
    
    def execute_loop(
        self,
        templates: List[str],
        problems: List[str],
        method_module: Any,
        args: argparse.Namespace,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Execute the complete experiment loop.
        
        Args:
            templates: List of template names to iterate over
            problems: List of problem IDs to iterate over
            method_module: Method module for running experiments
            args: Command line arguments
            output_dir: Output directory for results
            
        Returns:
            Dictionary with collected results:
                - results_data: List of result dictionaries
                - all_llm_responses: List of LLM responses
                - failed_experiments: List of failed experiments
                - completed: Boolean indicating successful completion
        """
        # Store for checkpoint creation
        self.current_templates = templates
        self.current_problems = problems
        self.current_args = args
        self.start_time = time.time()
        
        total_combinations = len(templates) * len(problems)
        progress = ProgressTracker(total_combinations)
        
        # Load checkpoint if resuming
        if self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                self._restore_from_checkpoint(checkpoint, progress)
        
        # Template-level loop
        for template in templates:
            if not self._execute_template_branch(
                template, problems, method_module, args, output_dir, progress
            ):
                # Failed with fail-fast
                self._save_checkpoint_if_enabled(output_dir)
                return self._build_result_dict(completed=False)
        
        # Clean up checkpoint on successful completion
        if self.checkpoint_manager:
            self.checkpoint_manager.archive_checkpoint()
        
        return self._build_result_dict(completed=True)
    
    def _execute_template_branch(
        self,
        template: str,
        problems: List[str],
        method_module: Any,
        args: argparse.Namespace,
        output_dir: Path,
        progress: ProgressTracker
    ) -> bool:
        """Execute all problems for a single template.
        
        Args:
            template: Template name
            problems: List of problem IDs
            method_module: Method module for running experiments
            args: Command line arguments
            output_dir: Output directory
            progress: Progress tracker instance
            
        Returns:
            True if completed successfully, False if failed with fail-fast
        """
        template_start_time = time.time()
        
        self._print_template_header(template, len(problems))
        
        # Problem-level loop
        for problem in problems:
            if not self._execute_single_test(
                template, problem, method_module, args, output_dir, progress
            ):
                # Failed with fail-fast
                return False
        
        self._finalize_template_branch(template, template_start_time, problems)
        return True
    
    def _execute_single_test(
        self,
        template: str,
        problem: str,
        method_module: Any,
        args: argparse.Namespace,
        output_dir: Path,
        progress: ProgressTracker
    ) -> bool:
        """Execute a single test (template + problem combination).
        
        Args:
            template: Template name
            problem: Problem ID
            method_module: Method module for running experiments
            args: Command line arguments
            output_dir: Output directory
            progress: Progress tracker
            
        Returns:
            True if completed or should continue, False if should stop (fail-fast)
        """
        # Skip if already completed (from checkpoint resume)
        if (template, problem) in self.completed_experiments:
            current_test = progress.increment()
            print(f"  ⏭️  Skipping (already completed from checkpoint)")
            return True
        
        problem_start_time = time.time()
        current_test = progress.increment()
        
        self._print_test_header(template, problem, progress)
        
        try:
            # Run the experiment
            llm_responses, rr_trains = self.run_experiment(
                template, problem, method_module, args, output_dir
            )
            
            # Collect responses
            self.all_llm_responses.extend(llm_responses)
            
            # Analyze and collect results
            if not args.dry_run:
                result_analysis = self.analyze_results(
                    template, problem, rr_trains, args.iterations, args.verbose
                )
                self.results_data.append(result_analysis)
                
                # Print success feedback
                self._print_result_feedback(result_analysis, args.dry_run)
            else:
                print("  ✓ Test Completed (dry run)")
            
            # Mark as completed and save checkpoint
            self.completed_experiments.add((template, problem))
            self._save_checkpoint_if_enabled(output_dir)
        
        except Exception as e:
            if not self._handle_experiment_failure(
                e, template, problem, current_test, progress.total_combinations, args
            ):
                return False  # Stop execution (fail-fast)
        
        # Finalize problem timing
        self._finalize_problem_timing(template, problem, problem_start_time)
        return True
    
    def _print_template_header(self, template: str, problem_count: int) -> None:
        """Print template branch header."""
        print(f"\n🔄 " + "=" * 70)
        print(f"📝 TEMPLATE BRANCH: {template}")
        print("=" * 75)
        self.log_timestamp(
            f"🌿 Starting template branch: {template} ({problem_count} problems)"
        )
    
    def _print_test_header(
        self, template: str, problem: str, progress: ProgressTracker
    ) -> None:
        """Print individual test header."""
        print(f"\n🎯 TEST [{progress.format_progress()}] " + "━" * 50)
        print(f"    📝 Template: {template}")
        print(f"    🎲 Problem:  {problem}")
        self.log_timestamp(
            f"🔍 Branch: {template} → {problem} "
            f"(Test {progress.current_test}/{progress.total_combinations})"
        )
    
    def _print_result_feedback(
        self, result_analysis: Dict[str, Any], is_dry_run: bool
    ) -> None:
        """Print feedback based on result quality.
        
        Removes primitive obsession: Uses SuccessThresholds instead of magic numbers.
        """
        if is_dry_run:
            print("  ✓ Test Completed (dry run)")
            return
        
        all_correct = result_analysis['all_correct_rate']
        partial = result_analysis['at_least_one_correct_rate']
        
        if all_correct >= self.thresholds.EXCELLENT:
            print("  🎯 Excellent Results!")
        elif all_correct >= self.thresholds.ACCEPTABLE:
            print("  ✅ Good Results!")
        elif partial >= self.thresholds.ACCEPTABLE:
            print("  ⚠️ Partial Success")
        else:
            print("  ❌ Poor Results")
    
    def _handle_experiment_failure(
        self,
        exception: Exception,
        template: str,
        problem: str,
        current_test: int,
        total_tests: int,
        args: argparse.Namespace
    ) -> bool:
        """Handle experiment failure.
        
        Args:
            exception: The exception that occurred
            template: Template name
            problem: Problem ID
            current_test: Current test number
            total_tests: Total number of tests
            args: Command line arguments
            
        Returns:
            True if should continue, False if should stop
        """
        print(f"\n  ❌ EXPERIMENT FAILED: {str(exception)}")
        print(f"     Template: {template}")
        print(f"     Problem: {problem}")
        
        # Record failure
        self.failed_experiments.append({
            'template': template,
            'problem': problem,
            'error': str(exception),
            'test_number': current_test
        })
        
        # Show traceback in verbose mode
        if args.verbose:
            traceback.print_exc()
        
        # Check fail-fast flag
        if args.fail_fast:
            print(f"\n❌ Stopping execution due to --fail-fast flag")
            print(f"   Failed on test {current_test}/{total_tests}")
            print(f"   Template: {template}, Problem: {problem}")
            return False  # Stop execution
        
        return True  # Continue execution
    
    def _finalize_problem_timing(
        self, template: str, problem: str, start_time: float
    ) -> None:
        """Finalize timing for a completed problem."""
        end_time = time.time()
        duration = end_time - start_time
        self.timing.record_problem_duration(template, problem, duration)
        
        formatted = self.format_duration(duration)
        self.log_timestamp(f"✅ Problem completed: {problem} in {formatted}")
    
    def _finalize_template_branch(
        self, template: str, start_time: float, problems: List[str]
    ) -> None:
        """Finalize timing and summary for a completed template branch."""
        end_time = time.time()
        duration = end_time - start_time
        self.timing.record_template_duration(template, duration)
        
        formatted = self.format_duration(duration)
        self.log_timestamp(
            f"🏁 Template branch completed: {template} in {formatted} "
            f"({len(problems)} problems)"
        )
    
    def _build_result_dict(self, completed: bool) -> Dict[str, Any]:
        """Build result dictionary with collected data."""
        return {
            'results_data': self.results_data,
            'all_llm_responses': self.all_llm_responses,
            'failed_experiments': self.failed_experiments,
            'completed': completed
        }
    
    def _save_checkpoint_if_enabled(self, output_dir: Path) -> None:
        """Save checkpoint if checkpoint manager is configured."""
        if not self.checkpoint_manager or not self.current_args:
            return
        
        # Convert args to dict (excluding non-serializable objects)
        args_dict = {
            'iterations': self.current_args.iterations,
            'model': self.current_args.model,
            'template': self.current_args.template if hasattr(self.current_args, 'template') else None,
            'problem': self.current_args.problem if hasattr(self.current_args, 'problem') else None,
            'dry_run': self.current_args.dry_run,
            'verbose': self.current_args.verbose,
            'fail_fast': self.current_args.fail_fast
        }
        
        checkpoint = ExperimentCheckpoint(
            output_dir=str(output_dir),
            templates=self.current_templates,
            problems=self.current_problems,
            completed_experiments=list(self.completed_experiments),
            results_data=self.results_data,
            failed_experiments=self.failed_experiments,
            start_time=self.start_time,
            checkpoint_time=time.time(),
            total_combinations=len(self.current_templates) * len(self.current_problems),
            args_dict=args_dict
        )
        
        self.checkpoint_manager.save_checkpoint(checkpoint)
    
    def _restore_from_checkpoint(self, checkpoint: ExperimentCheckpoint, progress: ProgressTracker) -> None:
        """Restore state from checkpoint."""
        self.log_timestamp("🔄 Resuming from checkpoint...")
        
        # Restore completed experiments
        self.completed_experiments = set(checkpoint.completed_experiments)
        
        # Restore results
        self.results_data = checkpoint.results_data
        self.failed_experiments = checkpoint.failed_experiments
        
        # Update progress tracker
        progress.current_test = len(checkpoint.completed_experiments)
        
        # Restore timing
        self.start_time = checkpoint.start_time
        
        # Log resume info
        next_exp = checkpoint.get_next_experiment()
        if next_exp:
            template, problem, test_num = next_exp
            self.log_timestamp(
                f"📍 Resuming from test {test_num}/{checkpoint.total_combinations}: {template} + {problem}"
            )
        
        self.log_timestamp(f"✅ Restored {len(checkpoint.completed_experiments)} completed experiments")
