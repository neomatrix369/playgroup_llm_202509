"""
Timing tracker for experiment execution.

Single Responsibility: Track all timing data for batch experiments.
Follows Object Calisthenics Rule 8: Focused state management with minimal instance variables.

Eliminates 4 timing-related instance variables from BatchExperimentRunner:
- timing_data
- template_timings
- individual_timings
- problem_timings
- global_start_time
- global_end_time
"""

import time
from datetime import datetime
from typing import Dict


class TimingTracker:
    """
    Single responsibility: Track timing for experiments at multiple levels.

    Tracks timing at three levels:
    1. Global: Overall batch experiment execution
    2. Template: Per-template execution across all problems
    3. Individual: Per template-problem combination
    4. Problem: Per problem across different runs

    Follows Object Calisthenics Rule 8: Max 2 instance variables (has 2: _global, _details)
    """

    def __init__(self):
        # Global timing (start/end timestamps)
        self._global = {"start": 0.0, "end": 0.0}

        # Detailed timings (all other timing data)
        self._details = {
            "template": {},  # template_name -> total_duration
            "individual": {},  # "template|problem" -> duration
            "problem": {},  # "template|problem" -> duration
        }

    def start_global(self) -> None:
        """Start the global timer for the entire batch run."""
        self._global["start"] = time.time()

    def end_global(self) -> None:
        """End the global timer for the entire batch run."""
        self._global["end"] = time.time()

    def get_global_duration(self) -> float:
        """
        Get total duration of the batch run in seconds.

        Returns:
            Duration in seconds, or 0.0 if not started/ended
        """
        if self._global["start"] == 0 or self._global["end"] == 0:
            return 0.0
        return self._global["end"] - self._global["start"]

    def get_global_start_time(self) -> float:
        """Get the global start timestamp."""
        return self._global["start"]

    def get_global_end_time(self) -> float:
        """Get the global end timestamp."""
        return self._global["end"]

    def format_global_start(self) -> str:
        """Format the global start time as human-readable string."""
        if self._global["start"] == 0:
            return "Not started"
        return datetime.fromtimestamp(self._global["start"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def format_global_end(self) -> str:
        """Format the global end time as human-readable string."""
        if self._global["end"] == 0:
            return "Not ended"
        return datetime.fromtimestamp(self._global["end"]).strftime("%Y-%m-%d %H:%M:%S")

    def record_individual_duration(
        self, template: str, problem: str, duration: float
    ) -> None:
        """
        Record duration for a single template-problem combination.

        Args:
            template: Template name
            problem: Problem identifier
            duration: Duration in seconds
        """
        key = f"{template}|{problem}"
        self._details["individual"][key] = duration

    def get_individual_duration(self, template: str, problem: str) -> float:
        """
        Get duration for a specific template-problem combination.

        Args:
            template: Template name
            problem: Problem identifier

        Returns:
            Duration in seconds, or 0.0 if not recorded
        """
        key = f"{template}|{problem}"
        return self._details["individual"].get(key, 0.0)

    def record_problem_duration(
        self, template: str, problem: str, duration: float
    ) -> None:
        """
        Record problem-level duration (can differ from individual due to retries/multiple runs).

        Args:
            template: Template name
            problem: Problem identifier
            duration: Duration in seconds
        """
        key = f"{template}|{problem}"
        self._details["problem"][key] = duration

    def get_problem_duration(self, template: str, problem: str) -> float:
        """
        Get problem-level duration.

        Args:
            template: Template name
            problem: Problem identifier

        Returns:
            Duration in seconds, or 0.0 if not recorded
        """
        key = f"{template}|{problem}"
        return self._details["problem"].get(key, 0.0)

    def record_template_duration(self, template: str, duration: float) -> None:
        """
        Record total duration for a template across all problems.

        Args:
            template: Template name
            duration: Total duration in seconds
        """
        self._details["template"][template] = duration

    def get_template_duration(self, template: str) -> float:
        """
        Get total duration for a template.

        Args:
            template: Template name

        Returns:
            Duration in seconds, or 0.0 if not recorded
        """
        return self._details["template"].get(template, 0.0)

    def get_all_template_durations(self) -> Dict[str, float]:
        """Get all template durations as a dictionary."""
        return self._details["template"].copy()

    def format_duration(self, seconds: float) -> str:
        """
        Format duration in seconds to human-readable string.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string like "1h 23m 45s" or "45.2s"
        """
        if seconds < 60:
            return f"{seconds:.1f}s"

        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60

        if minutes < 60:
            return f"{minutes}m {remaining_seconds:.1f}s"

        hours = minutes // 60
        remaining_minutes = minutes % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.0f}s"

    def get_summary(self) -> Dict[str, any]:
        """
        Get a summary of all timing data.

        Returns:
            Dictionary with timing summary including:
            - global_duration: Total batch duration
            - global_start: Start timestamp
            - global_end: End timestamp
            - template_count: Number of templates tracked
            - individual_count: Number of individual experiments tracked
        """
        return {
            "global_duration": self.get_global_duration(),
            "global_start": self._global["start"],
            "global_end": self._global["end"],
            "global_start_formatted": self.format_global_start(),
            "global_end_formatted": self.format_global_end(),
            "template_count": len(self._details["template"]),
            "individual_count": len(self._details["individual"]),
            "problem_count": len(self._details["problem"]),
        }
