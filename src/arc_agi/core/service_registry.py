"""
Service registry module for managing experiment services.

Consolidates all lazy-initialized services into a single registry,
following Object Calisthenics Rule 8: Maximum 2 instance variables.

This eliminates the code smell of having 7+ service references as
separate instance variables.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

# Avoid circular imports by using TYPE_CHECKING
if TYPE_CHECKING:
    from output.output_generator import OutputGenerator
    from output.console_display import ConsoleDisplay
    from core.experiment_executor import ExperimentExecutor
    from core.experiment_validator import ExperimentValidator
    from analysis.experiment_aggregator import ExperimentAggregator
    from analysis.experiment_summarizer import ExperimentSummarizer
    from core.experiment_loop_orchestrator import ExperimentLoopOrchestrator

from core.timing_tracker import TimingTracker
from domain.value_objects import SuccessThresholds


class ServiceRegistry:
    """
    Service locator pattern for experiment services.

    Consolidates 7+ service references into a single registry object.
    Manages lazy initialization and provides clean access to services.

    Benefits:
    - Reduces instance variables from 7+ to 1
    - Centralizes service creation logic
    - Makes dependencies explicit
    - Simplifies testing (can inject mock registry)

    Follows Object Calisthenics Rule 8: Only 2 instance variables (services + factory)
    """

    def __init__(self, service_factory: "ServiceFactory"):
        """Initialize with service factory.

        Args:
            service_factory: Factory for creating services
        """
        self._services: Dict[str, Any] = {}
        self._factory = service_factory

    def output_generator(self) -> "OutputGenerator":
        """Get or create OutputGenerator."""
        if "output_generator" not in self._services:
            self._services["output_generator"] = self._factory.create_output_generator()
        return self._services["output_generator"]

    def console_display(self) -> "ConsoleDisplay":
        """Get or create ConsoleDisplay."""
        if "console_display" not in self._services:
            self._services["console_display"] = self._factory.create_console_display()
        return self._services["console_display"]

    def executor(self) -> "ExperimentExecutor":
        """Get or create ExperimentExecutor."""
        if "executor" not in self._services:
            self._services["executor"] = self._factory.create_executor()
        return self._services["executor"]

    def validator(self) -> "ExperimentValidator":
        """Get or create ExperimentValidator."""
        if "validator" not in self._services:
            self._services["validator"] = self._factory.create_validator()
        return self._services["validator"]

    def aggregator(self) -> "ExperimentAggregator":
        """Get or create ExperimentAggregator."""
        if "aggregator" not in self._services:
            self._services["aggregator"] = self._factory.create_aggregator()
        return self._services["aggregator"]

    def summarizer(self) -> "ExperimentSummarizer":
        """Get or create ExperimentSummarizer."""
        if "summarizer" not in self._services:
            self._services["summarizer"] = self._factory.create_summarizer(
                self.aggregator()
            )
        return self._services["summarizer"]

    def loop_orchestrator(self) -> "ExperimentLoopOrchestrator":
        """Get or create ExperimentLoopOrchestrator."""
        if "loop_orchestrator" not in self._services:
            # Ensure executor is created first
            self.executor()
            self._services["loop_orchestrator"] = (
                self._factory.create_loop_orchestrator()
            )
        return self._services["loop_orchestrator"]

    def clear(self) -> None:
        """Clear all cached services (for testing)."""
        self._services.clear()


class ServiceFactory:
    """
    Factory for creating experiment services.

    Encapsulates service creation logic, making it easy to inject
    dependencies and configure services.

    Follows Object Calisthenics: Minimal instance variables, focused responsibility.
    """

    def __init__(
        self,
        timing_tracker: TimingTracker,
        results_accessor: Callable[[], Any],
        log_callback: Callable[[str, bool], None],
        format_duration_callback: Callable[[float], str],
        run_experiment_callback: Callable,
        analyze_results_callback: Callable,
        generate_ranking_analysis_callback: Callable,
        generate_persistent_summary_callback: Callable,
        thresholds: SuccessThresholds,
        log_file_handle_accessor: Callable[[], Any],
        checkpoint_manager_accessor: Callable[[], Optional[Any]] = lambda: None,
    ):
        """Initialize factory with required dependencies."""
        self.timing = timing_tracker
        self.results_accessor = results_accessor
        self.log_callback = log_callback
        self.format_duration = format_duration_callback
        self.run_experiment = run_experiment_callback
        self.analyze_results = analyze_results_callback
        self.generate_ranking_analysis = generate_ranking_analysis_callback
        self.generate_persistent_summary = generate_persistent_summary_callback
        self.thresholds = thresholds
        self.log_file_handle_accessor = log_file_handle_accessor
        self.checkpoint_manager_accessor = checkpoint_manager_accessor

    def create_output_generator(self) -> "OutputGenerator":
        """Create OutputGenerator instance."""
        from output.output_generator import OutputGenerator

        return OutputGenerator(self.results_accessor(), self.timing)

    def create_console_display(self) -> "ConsoleDisplay":
        """Create ConsoleDisplay instance."""
        from output.console_display import ConsoleDisplay

        return ConsoleDisplay(self.timing)

    def create_executor(self) -> "ExperimentExecutor":
        """Create ExperimentExecutor instance."""
        from core.experiment_executor import ExperimentExecutor

        return ExperimentExecutor(
            timing_tracker=self.timing,
            log_callback=self.log_callback,
            format_duration_callback=self.format_duration,
            log_file_handle=self.log_file_handle_accessor(),
        )

    def create_validator(self) -> "ExperimentValidator":
        """Create ExperimentValidator instance."""
        from core.experiment_validator import ExperimentValidator

        return ExperimentValidator(log_callback=self.log_callback)

    def create_aggregator(self) -> "ExperimentAggregator":
        """Create ExperimentAggregator instance."""
        from analysis.experiment_aggregator import ExperimentAggregator

        return ExperimentAggregator(
            ranking_analysis_callback=self.generate_ranking_analysis
        )

    def create_summarizer(self, aggregator) -> "ExperimentSummarizer":
        """Create ExperimentSummarizer instance."""
        from analysis.experiment_summarizer import ExperimentSummarizer

        return ExperimentSummarizer(
            aggregator=aggregator,
            output_generator_callback=self.generate_persistent_summary,
            log_callback=self.log_callback,
        )

    def create_loop_orchestrator(self) -> "ExperimentLoopOrchestrator":
        """Create ExperimentLoopOrchestrator instance."""
        from core.experiment_loop_orchestrator import ExperimentLoopOrchestrator

        return ExperimentLoopOrchestrator(
            timing_tracker=self.timing,
            log_callback=self.log_callback,
            format_duration_callback=self.format_duration,
            run_experiment_callback=self.run_experiment,
            analyze_results_callback=self.analyze_results,
            checkpoint_manager=self.checkpoint_manager_accessor(),
            success_thresholds=self.thresholds,
        )
