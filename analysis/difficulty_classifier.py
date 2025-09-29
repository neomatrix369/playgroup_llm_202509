"""
Problem difficulty classification logic.

Single Responsibility: Classify problem difficulty based on success rates.
Eliminates duplication from 2 sites in run_all_problems.py:
- Lines 700-707 (generate_ranking_analysis)
- Lines 1118-1120 (generate_persistent_summary)
"""

from domain.value_objects import DifficultyThresholds, DifficultyLevel


class DifficultyClassifier:
    """
    Single responsibility: Classify problem difficulty.

    Follows SOLID Single Responsibility Principle.
    Follows Object Calisthenics Rule 8: Max 2 instance variables.
    """

    def __init__(self, thresholds: DifficultyThresholds = None):
        """
        Initialize classifier with configurable thresholds.

        Args:
            thresholds: Custom thresholds, defaults to DifficultyThresholds()
        """
        self._thresholds = thresholds or DifficultyThresholds()

    def classify(self, avg_all_correct: float, avg_partial: float) -> str:
        """
        Classify problem difficulty based on average success rates.

        Single method implementing what was duplicated 2 times.

        Args:
            avg_all_correct: Average all-correct success rate (0.0-1.0)
            avg_partial: Average at-least-one-correct rate (0.0-1.0)

        Returns:
            Difficulty level: "EASY", "MEDIUM", "HARD", or "VERY_HARD"

        Logic:
            - EASY: 70%+ all-correct average
            - MEDIUM: 40-69% all-correct average
            - HARD: <40% all-correct but ≥50% partial success
            - VERY_HARD: <40% all-correct and <50% partial

        Examples:
            >>> classifier = DifficultyClassifier()
            >>> classifier.classify(0.75, 0.85)
            'EASY'
            >>> classifier.classify(0.55, 0.70)
            'MEDIUM'
            >>> classifier.classify(0.35, 0.60)
            'HARD'
            >>> classifier.classify(0.15, 0.25)
            'VERY_HARD'
        """
        if avg_all_correct >= self._thresholds.EASY:
            return DifficultyLevel.EASY

        if avg_all_correct >= self._thresholds.MEDIUM:
            return DifficultyLevel.MEDIUM

        if avg_partial >= self._thresholds.MEDIUM:
            return DifficultyLevel.HARD

        return DifficultyLevel.VERY_HARD

    def classify_simple(self, avg_success_rate: float) -> str:
        """
        Classify difficulty based on single success rate metric.

        Simplified version used in some reports.

        Args:
            avg_success_rate: Average success rate (0.0-1.0)

        Returns:
            Difficulty level: "EASY", "MEDIUM", "HARD", or "VERY_HARD"

        Examples:
            >>> classifier = DifficultyClassifier()
            >>> classifier.classify_simple(0.75)
            'EASY'
            >>> classifier.classify_simple(0.15)
            'VERY_HARD'
        """
        if avg_success_rate >= self._thresholds.EASY:
            return DifficultyLevel.EASY

        if avg_success_rate >= self._thresholds.MEDIUM:
            return DifficultyLevel.MEDIUM

        if avg_success_rate >= self._thresholds.HARD:
            return DifficultyLevel.HARD

        return DifficultyLevel.VERY_HARD

    def difficulty_icon(self, difficulty: str) -> str:
        """
        Get emoji icon for difficulty level.

        Args:
            difficulty: One of "EASY", "MEDIUM", "HARD", "VERY_HARD"

        Returns:
            Emoji representing difficulty level

        Examples:
            >>> classifier = DifficultyClassifier()
            >>> classifier.difficulty_icon("EASY")
            '🟢'
            >>> classifier.difficulty_icon("VERY_HARD")
            '🔴'
        """
        icons = {
            DifficultyLevel.EASY: "🟢",
            DifficultyLevel.MEDIUM: "🟡",
            DifficultyLevel.HARD: "🟠",
            DifficultyLevel.VERY_HARD: "🔴"
        }
        return icons.get(difficulty, "⚪")

    def is_easy(self, avg_all_correct: float, avg_partial: float) -> bool:
        """Check if problem qualifies as easy."""
        return self.classify(avg_all_correct, avg_partial) == DifficultyLevel.EASY

    def is_hard_or_harder(self, avg_all_correct: float, avg_partial: float) -> bool:
        """Check if problem qualifies as hard or very hard."""
        difficulty = self.classify(avg_all_correct, avg_partial)
        return difficulty in (DifficultyLevel.HARD, DifficultyLevel.VERY_HARD)