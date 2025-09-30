"""Formatter for failure summary output during experiment execution."""

from typing import List, Dict, Any, TextIO


class FailureSummaryFormatter:
    """Formats and displays failure summaries for console and file output.

    Eliminates duplication between console and file logging by centralizing
    formatting logic following the DRY principle.
    """

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string (e.g., "1h 23m 45s")
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"

    @classmethod
    def format_summary_stats(
        cls, total_attempted: int, failed_count: int
    ) -> List[str]:
        """Format summary statistics lines.

        Args:
            total_attempted: Total number of experiments attempted
            failed_count: Number of failed experiments

        Returns:
            List of formatted statistics lines
        """
        success_rate = (total_attempted - failed_count) / total_attempted if total_attempted > 0 else 0

        return [
            f"Total experiments attempted: {total_attempted}",
            f"Failed experiments: {failed_count}",
            f"Success rate: {success_rate:.1%}"
        ]

    @classmethod
    def format_failure_detail(cls, index: int, failure: Dict[str, Any]) -> List[str]:
        """Format details for a single failure.

        Args:
            index: Failure number (1-indexed)
            failure: Failure dictionary with template, problem, error, duration

        Returns:
            List of formatted detail lines
        """
        return [
            f"Failure #{index}:",
            f"  Template: {failure['template']}",
            f"  Problem: {failure['problem']}",
            f"  Error: {failure['error']}",
            f"  Duration: {cls.format_duration(failure['duration'])}"
        ]

    @classmethod
    def print_console_summary(
        cls, failed_experiments: List[Dict[str, Any]], total_attempted: int
    ) -> None:
        """Print failure summary to console.

        Args:
            failed_experiments: List of failure dictionaries
            total_attempted: Total number of experiments attempted
        """
        if not failed_experiments:
            return

        print(f"\n{'='*80}")
        print("⚠️  FAILURE SUMMARY ⚠️")
        print(f"{'='*80}")

        # Print statistics
        for line in cls.format_summary_stats(total_attempted, len(failed_experiments)):
            print(line)

        print(f"\n📋 Failed Experiment Details:")
        print("─" * 70)

        # Print failure details
        for i, failure in enumerate(failed_experiments, 1):
            print(f"\n❌ {cls.format_failure_detail(i, failure)[0]}")
            for line in cls.format_failure_detail(i, failure)[1:]:
                print(f"   {line}")

        print(f"\n{'='*80}")

    @classmethod
    def write_file_summary(
        cls,
        file_handle: TextIO,
        failed_experiments: List[Dict[str, Any]],
        total_attempted: int
    ) -> None:
        """Write failure summary to file.

        Args:
            file_handle: Open file handle for writing
            failed_experiments: List of failure dictionaries
            total_attempted: Total number of experiments attempted
        """
        if not failed_experiments:
            return

        file_handle.write(f"\n{'='*80}\n")
        file_handle.write("FAILURE SUMMARY\n")
        file_handle.write(f"{'='*80}\n")

        # Write statistics
        for line in cls.format_summary_stats(total_attempted, len(failed_experiments)):
            file_handle.write(f"{line}\n")
        file_handle.write("\n")

        # Write failure details
        for i, failure in enumerate(failed_experiments, 1):
            file_handle.write(f"\n")
            for line in cls.format_failure_detail(i, failure):
                file_handle.write(f"{line}\n")

            # Include traceback if available
            if failure.get('traceback'):
                file_handle.write(f"\nTraceback:\n{failure['traceback']}\n")

        file_handle.write(f"\n{'='*80}\n\n")
        file_handle.flush()