"""
Statistics aggregators for experiment analysis.

Following Object Calisthenics:
- Single Responsibility: Each aggregator handles one type of statistics
- Small classes: Each aggregator is focused and minimal
- Tell, Don't Ask: Stats objects know how to aggregate themselves

Eliminates 200+ lines of duplication across run_all_problems.py
"""

from typing import Any, Dict, List

from domain.stats_models import ExperimentStats, ProblemStats, TemplateStats

from analysis.difficulty_classifier import DifficultyClassifier


class TemplateStatisticsAggregator:
    """
    Single responsibility: Aggregate statistics by template.

    Replaces duplication in:
    - generate_ranking_analysis() lines 636-666
    - aggregate_experiment_data() lines 905-965
    """

    def __init__(self, results: List[Dict[str, Any]]):
        self._results = results

    def aggregate(self) -> Dict[str, TemplateStats]:
        """Aggregate results by template, returning stats objects."""
        stats_by_template = {}

        for result in self._results:
            template = result["template"]
            if template not in stats_by_template:
                stats_by_template[template] = TemplateStats(template_name=template)
            stats_by_template[template].add_result(result)

        return stats_by_template

    def aggregate_to_ranking(self) -> List[Dict[str, Any]]:
        """Aggregate and return sorted ranking list."""
        stats = self.aggregate()
        ranking = [s.to_dict() for s in stats.values()]
        ranking.sort(key=lambda x: x["score"], reverse=True)
        return ranking

    def aggregate_to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate and return dictionary format (for backward compatibility)."""
        stats = self.aggregate()
        return {name: s.to_dict() for name, s in stats.items()}


class ProblemStatisticsAggregator:
    """
    Single responsibility: Aggregate statistics by problem.

    Replaces duplication in:
    - generate_ranking_analysis() lines 671-697
    - aggregate_experiment_data() lines 905-965
    """

    def __init__(self, results: List[Dict[str, Any]]):
        self._results = results
        self._classifier = DifficultyClassifier()

    def aggregate(self) -> Dict[str, ProblemStats]:
        """Aggregate results by problem, returning stats objects."""
        stats_by_problem = {}

        for result in self._results:
            problem = result["problem"]
            if problem not in stats_by_problem:
                stats_by_problem[problem] = ProblemStats(problem_name=problem)
            stats_by_problem[problem].add_result(result)

        return stats_by_problem

    def aggregate_to_analysis(self) -> List[Dict[str, Any]]:
        """Aggregate and return sorted analysis list with difficulty classification."""
        stats = self.aggregate()
        analysis = []

        for problem_stats in stats.values():
            avg_all_correct = problem_stats.average_all_correct_rate()
            avg_partial = problem_stats.average_partial_rate()
            difficulty = self._classifier.classify(avg_all_correct, avg_partial)

            analysis.append(problem_stats.to_dict(difficulty=difficulty))

        analysis.sort(key=lambda x: x["avg_all_correct_rate"], reverse=True)
        return analysis

    def aggregate_to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate and return dictionary format (for backward compatibility)."""
        stats = self.aggregate()
        result = {}

        for name, problem_stats in stats.items():
            avg_all_correct = problem_stats.average_all_correct_rate()
            avg_partial = problem_stats.average_partial_rate()
            difficulty = self._classifier.classify(avg_all_correct, avg_partial)
            result[name] = problem_stats.to_dict(difficulty=difficulty)

        return result


class ExperimentStatisticsAggregator:
    """
    Single responsibility: Aggregate statistics by experiment combination.

    Replaces duplication in:
    - aggregate_experiment_data() lines 940-965
    """

    def __init__(self, results: List[Dict[str, Any]]):
        self._results = results

    def aggregate(self) -> Dict[str, ExperimentStats]:
        """Aggregate results by experiment (template|problem combination)."""
        stats_by_experiment = {}

        for result in self._results:
            template = result["template"]
            problem = result["problem"]
            experiment_key = f"{template}|{problem}"

            if experiment_key not in stats_by_experiment:
                stats_by_experiment[experiment_key] = ExperimentStats(
                    experiment_key=experiment_key
                )
            stats_by_experiment[experiment_key].add_result(result)

        return stats_by_experiment

    def aggregate_to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate and return dictionary format (for backward compatibility)."""
        stats = self.aggregate()
        return {key: s.to_dict() for key, s in stats.items()}


class BestTemplateRecommender:
    """
    Single responsibility: Generate best template recommendations per problem.

    Replaces code in generate_ranking_analysis() lines 701-719.
    """

    def __init__(self, results: List[Dict[str, Any]]):
        self._results = results

    def recommend_for_problem(self, problem: str) -> Dict[str, Any]:
        """Generate recommendation for a specific problem."""
        problem_results = [r for r in self._results if r["problem"] == problem]

        if not problem_results:
            return {}

        best_result = max(
            problem_results,
            key=lambda x: (x["all_correct_rate"], x["at_least_one_correct_rate"]),
        )

        good_alternatives = [
            r
            for r in problem_results
            if r["all_correct_rate"] >= 0.5 and r["template"] != best_result["template"]
        ]
        good_alternatives.sort(key=lambda x: x["all_correct_rate"], reverse=True)

        return {
            "best_template": best_result["template"],
            "best_score": best_result["all_correct_rate"],
            "alternatives": good_alternatives[:3],
        }

    def recommend_all(self) -> Dict[str, Dict[str, Any]]:
        """Generate recommendations for all problems."""
        problems = set(r["problem"] for r in self._results)
        return {problem: self.recommend_for_problem(problem) for problem in problems}
