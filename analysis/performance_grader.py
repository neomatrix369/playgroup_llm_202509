"""
Performance grading logic.

Single Responsibility: Grade experiment performance based on success rates.
Eliminates duplication from 3 sites in run_all_problems.py:
- Lines 466-475 (generate_console_table)
- Lines 517-527 (get_grade function)
- Lines 1273-1274 (generate_summary_log)
"""

from domain.value_objects import SuccessThresholds


class PerformanceGrader:
    """
    Single responsibility: Grade experiment performance.

    Follows SOLID Single Responsibility Principle.
    Follows Object Calisthenics Rule 8: Max 2 instance variables.
    """

    def __init__(self, thresholds: SuccessThresholds = None):
        """
        Initialize grader with configurable thresholds.

        Args:
            thresholds: Custom thresholds, defaults to SuccessThresholds()
        """
        self._thresholds = thresholds or SuccessThresholds()

    def grade(self, success_rate: float) -> str:
        """
        Calculate letter grade from success rate.

        Single method implementing what was duplicated 3+ times.

        Args:
            success_rate: Float between 0.0 and 1.0

        Returns:
            Single letter grade: A, B, C, D, or F

        Examples:
            >>> grader = PerformanceGrader()
            >>> grader.grade(0.85)
            'A'
            >>> grader.grade(0.65)
            'B'
            >>> grader.grade(0.25)
            'F'
        """
        if success_rate >= self._thresholds.EXCELLENT:
            return "A"
        if success_rate >= self._thresholds.GOOD:
            return "B"
        if success_rate >= self._thresholds.ACCEPTABLE:
            return "C"
        if success_rate >= self._thresholds.PARTIAL:
            return "D"
        return "F"

    def grade_with_icon(self, success_rate: float) -> str:
        """
        Calculate grade with emoji icon.

        Used in console output and reports for visual feedback.

        Args:
            success_rate: Float between 0.0 and 1.0

        Returns:
            Emoji + grade (e.g., "🎯 A", "❌ F")

        Examples:
            >>> grader = PerformanceGrader()
            >>> grader.grade_with_icon(0.85)
            '🎯 A'
            >>> grader.grade_with_icon(0.25)
            '❌ F'
        """
        grade = self.grade(success_rate)
        icons = {
            "A": "🎯",
            "B": "✅",
            "C": "⚠️",
            "D": "🔶",
            "F": "❌"
        }
        return f"{icons[grade]} {grade}"

    def success_icon(self, success_rate: float, partial_rate: float = None) -> str:
        """
        Get emoji icon based on success rates.

        Replaces inline conditional logic scattered throughout the codebase.

        Args:
            success_rate: All-correct success rate (0.0-1.0)
            partial_rate: Optional partial success rate (0.0-1.0)

        Returns:
            Emoji representing success level

        Examples:
            >>> grader = PerformanceGrader()
            >>> grader.success_icon(0.85)
            '🎯'
            >>> grader.success_icon(0.4, 0.6)
            '⚠️'
        """
        if success_rate >= self._thresholds.EXCELLENT:
            return "🎯"
        if success_rate >= self._thresholds.GOOD:
            return "✅"
        if partial_rate and partial_rate >= self._thresholds.ACCEPTABLE:
            return "⚠️"
        if success_rate >= self._thresholds.ACCEPTABLE:
            return "⚠️"
        return "❌"

    def is_excellent(self, success_rate: float) -> bool:
        """Check if rate qualifies as excellent performance."""
        return success_rate >= self._thresholds.EXCELLENT

    def is_good(self, success_rate: float) -> bool:
        """Check if rate qualifies as good performance."""
        return success_rate >= self._thresholds.GOOD

    def is_acceptable(self, success_rate: float) -> bool:
        """Check if rate qualifies as acceptable performance."""
        return success_rate >= self._thresholds.ACCEPTABLE