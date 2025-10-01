"""
Domain statistics models for experiment analysis.

Following Object Calisthenics Rule 3: Wrap primitives and create first-class collections.
These models encapsulate statistics computation behavior (Tell, Don't Ask principle).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class TemplateStats:
    """
    First-class statistics for a template across multiple experiments.

    Encapsulates template performance metrics with behavior.
    Replaces: Raw dictionaries with 'results', 'total_duration', 'problem_count'
    """

    template_name: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    total_duration: float = 0.0
    problem_count: int = 0
    success_rates: List[float] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)
    total_runs: int = 0
    experiments: int = 0

    def add_result(self, result: Dict[str, Any]) -> None:
        """Tell, Don't Ask: Stats object knows how to add a result."""
        self.results.append(result)
        self.total_duration += result.get("individual_duration", 0)
        self.problem_count += 1
        self.success_rates.append(result.get("all_correct_rate", 0))
        self.durations.append(result.get("individual_duration", 0))
        self.total_runs += 1
        self.experiments += 1

    def average_all_correct_rate(self) -> float:
        """Calculate average all-correct success rate."""
        if not self.results:
            return 0.0
        return sum(r["all_correct_rate"] for r in self.results) / len(self.results)

    def average_partial_rate(self) -> float:
        """Calculate average at-least-one-correct rate."""
        if not self.results:
            return 0.0
        return sum(r["at_least_one_correct_rate"] for r in self.results) / len(self.results)

    def average_duration(self) -> float:
        """Calculate average duration per problem."""
        if self.problem_count == 0:
            return 0.0
        return self.total_duration / self.problem_count

    def excellent_count(self, threshold: float = 0.8) -> int:
        """Count problems with excellent performance (≥80%)."""
        return len([r for r in self.results if r["all_correct_rate"] >= threshold])

    def good_count(self, min_threshold: float = 0.5, max_threshold: float = 0.8) -> int:
        """Count problems with good performance (50-79%)."""
        return len(
            [r for r in self.results if min_threshold <= r["all_correct_rate"] < max_threshold]
        )

    def weighted_score(self, all_correct_weight: float = 0.8, partial_weight: float = 0.2) -> float:
        """Calculate weighted performance score."""
        return (
            self.average_all_correct_rate() * all_correct_weight
            + self.average_partial_rate() * partial_weight
        )

    def max_success_rate(self) -> float:
        """Get maximum success rate achieved."""
        if not self.success_rates:
            return 0.0
        return max(self.success_rates)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "template": self.template_name,
            "avg_all_correct_rate": self.average_all_correct_rate(),
            "avg_partial_rate": self.average_partial_rate(),
            "excellent_problems": self.excellent_count(),
            "good_problems": self.good_count(),
            "total_problems": len(self.results),
            "avg_duration": self.average_duration(),
            "score": self.weighted_score(),
            "avg_success_rate": self.average_all_correct_rate(),
            "max_success_rate": self.max_success_rate(),
            "total_runs": self.total_runs,
            "experiments": self.experiments,
            "success_rates": self.success_rates,  # Include for consistency calculations
        }


@dataclass
class ProblemStats:
    """
    First-class statistics for a problem across multiple templates.

    Encapsulates problem difficulty metrics with behavior.
    """

    problem_name: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    success_rates: List[float] = field(default_factory=list)
    templates_used: Set[str] = field(default_factory=set)
    total_runs: int = 0
    experiments: int = 0

    def add_result(self, result: Dict[str, Any]) -> None:
        """Tell, Don't Ask: Stats object knows how to add a result."""
        self.results.append(result)
        self.success_rates.append(result.get("all_correct_rate", 0))
        self.templates_used.add(result["template"])
        self.total_runs += 1
        self.experiments += 1

    def average_all_correct_rate(self) -> float:
        """Calculate average all-correct success rate."""
        if not self.results:
            return 0.0
        return sum(r["all_correct_rate"] for r in self.results) / len(self.results)

    def average_partial_rate(self) -> float:
        """Calculate average at-least-one-correct rate."""
        if not self.results:
            return 0.0
        return sum(r["at_least_one_correct_rate"] for r in self.results) / len(self.results)

    def best_template_result(self) -> Optional[Dict[str, Any]]:
        """Find the best performing template for this problem."""
        if not self.results:
            return None
        return max(self.results, key=lambda x: x["all_correct_rate"])

    def max_success_rate(self) -> float:
        """Get maximum success rate achieved."""
        if not self.success_rates:
            return 0.0
        return max(self.success_rates)

    def to_dict(self, difficulty: str = None) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        best = self.best_template_result()
        return {
            "problem": self.problem_name,
            "difficulty": difficulty,
            "avg_all_correct_rate": self.average_all_correct_rate(),
            "avg_partial_rate": self.average_partial_rate(),
            "best_template": best["template"] if best else None,
            "best_score": best["all_correct_rate"] if best else 0.0,
            "template_count": len(self.results),
            "avg_success_rate": self.average_all_correct_rate(),
            "max_success_rate": self.max_success_rate(),
            "templates_used": list(self.templates_used),
            "total_runs": self.total_runs,
            "experiments": self.experiments,
        }


@dataclass
class ExperimentStats:
    """
    First-class statistics for an experiment combination (template+problem).

    Tracks performance across multiple runs of the same combination.
    """

    experiment_key: str  # Format: "template|problem"
    runs: int = 0
    success_rates: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)

    def add_result(self, result: Dict[str, Any]) -> None:
        """Tell, Don't Ask: Stats object knows how to add a result."""
        self.runs += 1
        self.success_rates.append(result.get("all_correct_rate", 0))
        self.timestamps.append(result.get("experiment_timestamp", ""))

    def average_success_rate(self) -> float:
        """Calculate average success rate across runs."""
        if not self.success_rates:
            return 0.0
        return sum(self.success_rates) / len(self.success_rates)

    def max_success_rate(self) -> float:
        """Get maximum success rate achieved."""
        if not self.success_rates:
            return 0.0
        return max(self.success_rates)

    def latest_timestamp(self) -> str:
        """Get the timestamp of the most recent run."""
        if not self.timestamps:
            return ""
        return max(self.timestamps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "runs": self.runs,
            "success_rates": self.success_rates,
            "avg_success_rate": self.average_success_rate(),
            "max_success_rate": self.max_success_rate(),
            "latest_run": self.latest_timestamp(),
            "timestamps": self.timestamps,
        }
