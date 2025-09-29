"""
Domain value objects for experiment management.

Following Object Calisthenics Rule 3: Wrap all primitives and strings.
These immutable value objects replace magic numbers and strings throughout the codebase.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SuccessThresholds:
    """
    Immutable thresholds for performance grading.

    Replaces hardcoded magic numbers: 0.8, 0.6, 0.5, 0.4, 0.3
    Used in: grade calculation, performance analysis, report generation
    """
    EXCELLENT: float = 0.8  # 80%+ all correct
    GOOD: float = 0.6       # 60-79% all correct
    ACCEPTABLE: float = 0.5 # 50-59% all correct
    PARTIAL: float = 0.4    # 40-49% all correct
    POOR: float = 0.3       # 30-39% all correct


@dataclass(frozen=True)
class DifficultyThresholds:
    """
    Immutable thresholds for problem difficulty classification.

    Replaces hardcoded magic numbers: 0.7, 0.4, 0.2
    Used in: problem analysis, ranking reports, difficulty categorization
    """
    EASY: float = 0.7    # 70%+ average success rate
    MEDIUM: float = 0.4  # 40-69% average success rate
    HARD: float = 0.2    # 20-39% average success rate or partial success


@dataclass(frozen=True)
class DifficultyLevel:
    """
    Immutable difficulty level labels.

    Replaces hardcoded strings: "EASY", "MEDIUM", "HARD", "VERY_HARD"
    """
    EASY: str = "EASY"
    MEDIUM: str = "MEDIUM"
    HARD: str = "HARD"
    VERY_HARD: str = "VERY_HARD"


@dataclass
class ExperimentResult:
    """
    First-class experiment result with behavior (Tell, Don't Ask principle).

    Replaces: Raw dictionaries with 'template', 'problem', 'all_correct_rate', etc.
    Benefits:
    - Type safety
    - Self-documenting
    - Encapsulates behavior (grade calculation, success checks)
    - Follows Object Calisthenics Rule 9: Objects expose behavior, not data
    """
    template: str
    problem: str
    all_correct_rate: float
    at_least_one_correct_rate: float
    duration: float
    total_runs: int = 1
    all_correct: Optional[int] = None
    at_least_one_correct: Optional[int] = None

    def grade(self, thresholds: SuccessThresholds = SuccessThresholds()) -> str:
        """
        Tell, Don't Ask: Result knows its own grade.

        Returns: Single letter grade A-F based on all_correct_rate
        """
        if self.all_correct_rate >= thresholds.EXCELLENT:
            return "A"
        if self.all_correct_rate >= thresholds.GOOD:
            return "B"
        if self.all_correct_rate >= thresholds.ACCEPTABLE:
            return "C"
        if self.all_correct_rate >= thresholds.PARTIAL:
            return "D"
        return "F"

    def grade_with_icon(self, thresholds: SuccessThresholds = SuccessThresholds()) -> str:
        """
        Tell, Don't Ask: Result knows its visual representation.

        Returns: Emoji + grade (e.g., "🎯 A")
        """
        grade = self.grade(thresholds)
        icons = {
            "A": "🎯",
            "B": "✅",
            "C": "⚠️",
            "D": "🔶",
            "F": "❌"
        }
        return f"{icons[grade]} {grade}"

    def is_excellent(self, thresholds: SuccessThresholds = SuccessThresholds()) -> bool:
        """Tell, Don't Ask: Result knows if it's excellent."""
        return self.all_correct_rate >= thresholds.EXCELLENT

    def is_good(self, thresholds: SuccessThresholds = SuccessThresholds()) -> bool:
        """Tell, Don't Ask: Result knows if it's good."""
        return self.all_correct_rate >= thresholds.GOOD

    def is_successful(self) -> bool:
        """Tell, Don't Ask: Result knows if it has any success."""
        return self.all_correct_rate > 0

    def has_partial_success(self, thresholds: SuccessThresholds = SuccessThresholds()) -> bool:
        """Tell, Don't Ask: Result knows if it has partial success."""
        return (self.all_correct_rate < thresholds.ACCEPTABLE and
                self.at_least_one_correct_rate >= thresholds.ACCEPTABLE)

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility with existing code."""
        return {
            'template': self.template,
            'problem': self.problem,
            'all_correct_rate': self.all_correct_rate,
            'at_least_one_correct_rate': self.at_least_one_correct_rate,
            'individual_duration': self.duration,
            'total_runs': self.total_runs,
            'all_correct': self.all_correct or int(self.all_correct_rate * self.total_runs),
            'at_least_one_correct': self.at_least_one_correct or int(self.at_least_one_correct_rate * self.total_runs),
        }