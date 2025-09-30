"""
Experiment execution module for running individual experiments.

Handles:
- Experiment setup (logger, database)
- Single experiment execution with timing and error handling
- Results analysis and success rate calculation

Following SOLID principles:
- Single Responsibility: Only handles experiment execution
- Dependency Inversion: Depends on abstractions (callbacks, timing tracker)

Extracted from BatchExperimentRunner to reduce complexity and improve testability.
"""

import time
import traceback
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Callable, Optional, TextIO

import utils
from core.timing_tracker import TimingTracker
from output.iteration_formatter import IterationStatusFormatter


class ExperimentExecutor:
    """Executes individual experiments with timing and error handling.
    
    Centralizes execution logic that was scattered in BatchExperimentRunner,
    reducing the main class by ~150 lines.
    """
    
    def __init__(
        self, 
        timing_tracker: TimingTracker,
        log_callback: Callable[[str], None],
        format_duration_callback: Callable[[float], str],
        log_file_handle: Optional[TextIO] = None
    ):
        """Initialize experiment executor.
        
        Args:
            timing_tracker: TimingTracker instance for recording durations
            log_callback: Callback function for logging with timestamps
            format_duration_callback: Callback function for formatting durations
            log_file_handle: Optional file handle for detailed logging
        """
        self.timing = timing_tracker
        self.log = log_callback
        self.format_duration = format_duration_callback
        self.log_file_handle = log_file_handle
        
        # Execution tracking
        self.failed_experiments: List[Dict[str, Any]] = []
        self.total_experiments_attempted: int = 0
    
    def setup_experiment(self, experiment_folder: Path) -> Tuple[Any, str]:
        """Setup experiment folder and database without re-parsing arguments.
        
        Args:
            experiment_folder: Path to experiment output folder
            
        Returns:
            Tuple of (logger, db_filename)
        """
        logger = utils.setup_logging(experiment_folder)
        logger.info("Started experiment")
        db_filename = utils.make_db(experiment_folder)
        logger.info(f"Database created at: {db_filename}")
        return logger, db_filename
    
    def run_experiment(
        self,
        template: str,
        problem: str,
        method_module: Any,
        args: argparse.Namespace,
        experiment_folder: Path
    ) -> Tuple[List[Any], List[Any]]:
        """Run a single experiment with full timing and error handling.
        
        Args:
            template: Template name to use
            problem: Problem ID to solve
            method_module: Method module with run_experiment_for_iterations function
            args: Command line arguments
            experiment_folder: Path to experiment output folder
            
        Returns:
            Tuple of (llm_responses, rr_trains) from the experiment
            
        Raises:
            Exception: Re-raises any exception after logging it
        """
        self.log(f"Starting individual test: {template} + {problem}")
        self.total_experiments_attempted += 1
        
        if args.dry_run:
            print(f"[DRY-RUN] Would run {template} with {problem}")
            return [], []
        
        individual_start_time = time.time()
        
        try:
            # Setup for this specific experiment
            logger, db_filename = self.setup_experiment(experiment_folder)
            
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
            self.timing.record_individual_duration(template, problem, individual_duration)
            
            formatted_duration = self.format_duration(individual_duration)
            self.log(f"Individual test completed successfully in {formatted_duration}")
            
            return llm_responses, rr_trains
        
        except Exception as e:
            individual_end_time = time.time()
            individual_duration = individual_end_time - individual_start_time
            
            # Store timing even for failures
            self.timing.record_individual_duration(template, problem, individual_duration)
            
            # Record the failure
            self.failed_experiments.append({
                'template': template,
                'problem': problem,
                'error': str(e),
                'duration': individual_duration,
                'traceback': traceback.format_exc() if args.verbose else None
            })
            
            formatted_duration = self.format_duration(individual_duration)
            self.log(f"Individual test failed after {formatted_duration}: {str(e)}")
            
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
        """Analyze experiment results for success rates with real-time feedback.
        
        Args:
            template: Template name used
            problem: Problem ID tested
            rr_trains: List of (run_result, execution_outcomes) tuples from iterations
            iterations: Number of iterations run
            verbose: Whether to show verbose output
            
        Returns:
            Dictionary with analysis results including success rates and timing
        """
        all_correct = 0
        at_least_one_correct = 0
        
        print(f"    📊 Analyzing {iterations} iteration(s) for {problem}:")
        
        formatter = IterationStatusFormatter()
        
        for i, rr_train in enumerate(rr_trains, 1):
            ran_all_train_problems_correctly = rr_train[0].transform_ran_and_matched_for_all_inputs
            ran_at_least_one_train_problem_correctly = rr_train[0].transform_ran_and_matched_at_least_once
            
            # Get detailed sub-problem results
            execution_outcomes = rr_train[1] if len(rr_train) > 1 else []
            
            # Determine status and display
            status_icon, status_text = formatter.format_status(
                ran_all_train_problems_correctly, ran_at_least_one_train_problem_correctly
            )
            formatter.print_iteration_status(i, status_icon, status_text, execution_outcomes, verbose)
            
            # Count successes
            if ran_all_train_problems_correctly:
                all_correct += 1
            if ran_at_least_one_train_problem_correctly:
                at_least_one_correct += 1
        
        # Summary for this problem
        all_correct_rate = all_correct / iterations
        at_least_one_rate = at_least_one_correct / iterations
        
        summary_icon, summary_color = formatter.format_summary_status(all_correct_rate, at_least_one_rate)
        formatter.print_problem_summary(
            problem, all_correct, at_least_one_correct, iterations,
            all_correct_rate, at_least_one_rate, summary_icon, summary_color
        )
        
        return {
            "template": template,
            "problem": problem,
            "total_runs": iterations,
            "all_correct": all_correct,
            "at_least_one_correct": at_least_one_correct,
            "all_correct_rate": all_correct_rate,
            "at_least_one_correct_rate": at_least_one_rate,
            "individual_duration": self.timing.get_individual_duration(template, problem),
            "problem_duration": self.timing.get_problem_duration(template, problem),
        }
