"""
Experiment aggregation module for discovering and combining past experiments.

Handles:
- Discovery of existing experiment result directories
- Loading experiment data from CSV files
- Aggregation of data across multiple experiment runs
- Generation of cross-experiment statistics

Following SOLID principles:
- Single Responsibility: Only handles experiment discovery and aggregation
- Dependency Inversion: Depends on abstractions (output generator)

Extracted from BatchExperimentRunner to reduce complexity and improve testability.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List

try:
    import pandas as pd
except ImportError:
    pd = None

from arc_agi.analysis.statistics_aggregator import (
    ExperimentStatisticsAggregator,
    ProblemStatisticsAggregator,
    TemplateStatisticsAggregator,
)


class ExperimentAggregator:
    """Discovers and aggregates data from past experiment runs.

    Centralizes experiment discovery and aggregation logic that was in
    BatchExperimentRunner, enabling independent testing and reuse.
    """

    def __init__(self, ranking_analysis_callback: Callable[[List[Dict]], Dict] = None):
        """Initialize experiment aggregator.

        Args:
            ranking_analysis_callback: Optional callback to generate ranking analysis
                                      from results data
        """
        self.generate_ranking_analysis = ranking_analysis_callback

    def discover_experiments(self, base_output_dir: Path) -> List[Dict[str, Any]]:
        """Discover existing experiment result directories and load their data.

        Args:
            base_output_dir: Base directory containing experiment subdirectories

        Returns:
            List of experiment metadata dictionaries, sorted by date (recent first)
        """
        if not base_output_dir.exists():
            print(f"⚠️  Output directory {base_output_dir} does not exist")
            return []

        experiment_dirs = []
        for item in base_output_dir.iterdir():
            if item.is_dir() and item.name.replace("_", "").replace("-", "").isdigit():
                # Look for CSV files in the directory
                csv_files = list(item.glob("batch_results_*.csv"))
                if csv_files:
                    experiment_dirs.append(
                        {
                            "directory": item,
                            "timestamp": item.name,
                            "csv_file": csv_files[0],
                            "date": item.stat().st_mtime,
                        }
                    )

        # Sort by date (most recent first)
        experiment_dirs.sort(key=lambda x: x["date"], reverse=True)
        return experiment_dirs

    def load_data(self, csv_file: Path) -> List[Dict[str, Any]]:
        """Load experiment data from CSV file.

        Args:
            csv_file: Path to CSV file containing experiment results

        Returns:
            List of result dictionaries, or empty list if loading fails
        """
        if pd is None:
            print("⚠️  pandas not available, cannot load experiment data")
            return []

        try:
            df = pd.read_csv(csv_file)
            return df.to_dict("records")
        except Exception as e:
            print(f"⚠️  Error loading {csv_file}: {e}")
            return []

    def aggregate(self, all_experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data from multiple experiment runs.

        Args:
            all_experiments: List of experiment metadata dictionaries from discover_experiments()

        Returns:
            Dictionary containing:
            - analysis: Ranking analysis of aggregated results
            - template_stats: Statistics per template
            - problem_stats: Statistics per problem
            - experiment_stats: Overall experiment statistics
            - experiment_metadata: Metadata about each experiment run
            - total_results: Total number of individual results
            - total_experiments: Total number of experiment runs
        """
        if not all_experiments:
            return {}

        # Combine all experiment data
        all_results = []
        experiment_metadata = []

        for exp in all_experiments:
            data = self.load_data(exp["csv_file"])
            if data:
                # Add metadata to each result
                for result in data:
                    result["experiment_timestamp"] = exp["timestamp"]
                    result["experiment_date"] = exp["date"]
                all_results.extend(data)

                experiment_metadata.append(
                    {
                        "timestamp": exp["timestamp"],
                        "date": exp["date"],
                        "result_count": len(data),
                        "directory": str(exp["directory"]),
                    }
                )

        if not all_results:
            return {}

        # Generate analysis (if callback provided)
        analysis = {}
        if self.generate_ranking_analysis:
            analysis = self.generate_ranking_analysis(all_results)

        # Add aggregated statistics using existing aggregator classes
        template_aggregator = TemplateStatisticsAggregator(all_results)
        template_stats = template_aggregator.aggregate_to_dict()

        problem_aggregator = ProblemStatisticsAggregator(all_results)
        problem_stats = problem_aggregator.aggregate_to_dict()

        experiment_aggregator = ExperimentStatisticsAggregator(all_results)
        experiment_stats = experiment_aggregator.aggregate_to_dict()

        return {
            "analysis": analysis,
            "template_stats": template_stats,
            "problem_stats": problem_stats,
            "experiment_stats": experiment_stats,
            "experiment_metadata": experiment_metadata,
            "total_results": len(all_results),
            "total_experiments": len(all_experiments),
        }
