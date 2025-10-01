"""Formatter for iteration status display during experiment execution."""

from typing import Any, List, Tuple


class IterationStatusFormatter:
    """Formats and displays iteration status with visual indicators.

    Separates presentation logic from analysis logic following Single Responsibility Principle.
    """

    @staticmethod
    def format_status(
        ran_all_correctly: bool, ran_at_least_one_correctly: bool
    ) -> Tuple[str, str]:
        """Determine status icon and text based on iteration results.

        Args:
            ran_all_correctly: Whether all train problems succeeded
            ran_at_least_one_correctly: Whether at least one train problem succeeded

        Returns:
            Tuple of (status_icon, status_text)
        """
        if ran_all_correctly:
            return "✅", "ALL_CORRECT"
        elif ran_at_least_one_correctly:
            return "⚠️", "PARTIAL"
        else:
            return "❌", "FAILED"

    @staticmethod
    def print_iteration_status(
        iteration_num: int,
        status_icon: str,
        status_text: str,
        execution_outcomes: List[Any] = None,
        verbose: bool = False,
    ) -> None:
        """Print status for a single iteration.

        Args:
            iteration_num: Current iteration number (1-indexed)
            status_icon: Visual icon for status (✅, ⚠️, ❌)
            status_text: Text description of status
            execution_outcomes: Optional list of execution outcome objects
            verbose: Whether to show detailed sub-problem information
        """
        print(f"      Iteration {iteration_num}: {status_icon} {status_text}")

        if execution_outcomes and verbose:
            # Show detailed sub-problem results
            for j, outcome in enumerate(execution_outcomes, 1):
                sub_icon = "✅" if outcome.was_correct else "❌"
                sub_status = "PASS" if outcome.was_correct else "FAIL"
                print(f"        Sub-problem {j}: {sub_icon} {sub_status}")
        elif execution_outcomes:
            # Show summary without verbose details
            correct_count = sum(
                1 for outcome in execution_outcomes if outcome.was_correct
            )
            total_count = len(execution_outcomes)
            print(f"        Sub-problems: {correct_count}/{total_count} correct")

    @staticmethod
    def format_summary_status(
        all_correct_rate: float, at_least_one_rate: float
    ) -> Tuple[str, str]:
        """Determine summary status icon and color based on rates.

        Args:
            all_correct_rate: Rate of all-correct iterations (0.0-1.0)
            at_least_one_rate: Rate of at-least-one-correct iterations (0.0-1.0)

        Returns:
            Tuple of (summary_icon, summary_color)
        """
        if all_correct_rate >= 0.8:
            return "🎯", "EXCELLENT"
        elif all_correct_rate >= 0.5:
            return "✅", "GOOD"
        elif at_least_one_rate >= 0.5:
            return "⚠️", "PARTIAL"
        else:
            return "❌", "POOR"

    @staticmethod
    def print_problem_summary(
        problem: str,
        all_correct: int,
        at_least_one_correct: int,
        iterations: int,
        all_correct_rate: float,
        at_least_one_rate: float,
        summary_icon: str,
        summary_color: str,
    ) -> None:
        """Print summary statistics for a problem.

        Args:
            problem: Problem identifier
            all_correct: Number of all-correct iterations
            at_least_one_correct: Number of partial success iterations
            iterations: Total number of iterations
            all_correct_rate: Rate of all-correct iterations
            at_least_one_rate: Rate of partial success iterations
            summary_icon: Visual icon for summary (🎯, ✅, ⚠️, ❌)
            summary_color: Color/quality descriptor (EXCELLENT, GOOD, PARTIAL, POOR)
        """
        print(
            f"    {summary_icon} Problem Results: {all_correct}/{iterations} all correct ({all_correct_rate:.1%}), "
            f"{at_least_one_correct}/{iterations} partial ({at_least_one_rate:.1%}) - {summary_color}"
        )
