"""
Experiment state management module.

Consolidates experiment execution state into cohesive objects,
following Object Calisthenics Rule 8: Maximum 2 instance variables per class.

This eliminates the code smell of having 10+ instance variables in
BatchExperimentRunner by grouping related state.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path


class ExperimentResults:
    """
    Encapsulates all experiment results data.
    
    Removes data clump code smell: Instead of having results_data,
    all_llm_responses, failed_experiments as separate instance variables,
    they are grouped into a cohesive object that knows how to manage them.
    
    Follows Object Calisthenics Rule 8: Only 3 instance variables.
    """
    
    def __init__(self):
        self._results: List[Dict[str, Any]] = []
        self._llm_responses: List[Any] = []
        self._failures: List[Dict[str, Any]] = []
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a result to the collection."""
        self._results.append(result)
    
    def add_llm_responses(self, responses: List[Any]) -> None:
        """Add LLM responses to the collection."""
        self._llm_responses.extend(responses)
    
    def record_failure(self, failure: Dict[str, Any]) -> None:
        """Record a failed experiment."""
        self._failures.append(failure)
    
    def set_all_results(self, results: List[Dict[str, Any]]) -> None:
        """Set all results (for orchestrator integration)."""
        self._results = results
    
    def set_all_llm_responses(self, responses: List[Any]) -> None:
        """Set all LLM responses (for orchestrator integration)."""
        self._llm_responses = responses
    
    def set_all_failures(self, failures: List[Dict[str, Any]]) -> None:
        """Set all failures (for orchestrator integration)."""
        self._failures = failures
    
    @property
    def results(self) -> List[Dict[str, Any]]:
        """Get all results."""
        return self._results
    
    @property
    def llm_responses(self) -> List[Any]:
        """Get all LLM responses."""
        return self._llm_responses
    
    @property
    def failures(self) -> List[Dict[str, Any]]:
        """Get all failures."""
        return self._failures
    
    def has_results(self) -> bool:
        """Check if any results exist."""
        return len(self._results) > 0
    
    def has_failures(self) -> bool:
        """Check if any failures exist."""
        return len(self._failures) > 0
    
    def failure_count(self) -> int:
        """Get number of failures."""
        return len(self._failures)
    
    def result_count(self) -> int:
        """Get number of results."""
        return len(self._results)
    
    def clear(self) -> None:
        """Clear all results (for testing)."""
        self._results.clear()
        self._llm_responses.clear()
        self._failures.clear()


class ExperimentContext:
    """
    Encapsulates experiment execution context.
    
    Groups related state that was previously scattered across instance variables.
    Follows Tell, Don't Ask principle.
    
    Follows Object Calisthenics Rule 8: Only 2 instance variables.
    """
    
    def __init__(self, log_file_path: Optional[Path] = None):
        self._log_file_handle: Optional[Any] = None
        self._total_experiments_attempted: int = 0
        
        if log_file_path:
            self.open_log_file(log_file_path)
    
    def open_log_file(self, path: Path) -> None:
        """Open log file for writing."""
        if self._log_file_handle:
            self._log_file_handle.close()
        self._log_file_handle = open(path, 'w')

    def set_log_handle(self, handle: Any) -> None:
        """Set an already-opened log file handle (for resume mode)."""
        if self._log_file_handle and self._log_file_handle != handle:
            self._log_file_handle.close()
        self._log_file_handle = handle

    def write_to_log(self, message: str) -> None:
        """Write message to log file."""
        if self._log_file_handle:
            self._log_file_handle.write(message + "\n")
            self._log_file_handle.flush()
    
    def close_log_file(self) -> None:
        """Close log file."""
        if self._log_file_handle:
            self._log_file_handle.close()
            self._log_file_handle = None

    def close_log(self) -> None:
        """Alias for close_log_file() for backward compatibility."""
        self.close_log_file()

    def increment_attempts(self) -> int:
        """Increment and return experiments attempted."""
        self._total_experiments_attempted += 1
        return self._total_experiments_attempted
    
    def get_attempts(self) -> int:
        """Get total experiments attempted."""
        return self._total_experiments_attempted
    
    def set_attempts(self, count: int) -> None:
        """Set total experiments attempted (for sync with executor)."""
        self._total_experiments_attempted = count
    
    @property
    def log_handle(self):
        """Get log file handle (for backward compatibility)."""
        return self._log_file_handle
    
    def is_logging(self) -> bool:
        """Check if logging to file."""
        return self._log_file_handle is not None
