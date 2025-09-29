#!/usr/bin/env python3
"""
Quick validation tests for statistics aggregator classes.

Run: python test_statistics_aggregator.py
"""

from domain.stats_models import TemplateStats, ProblemStats, ExperimentStats
from analysis.statistics_aggregator import (
    TemplateStatisticsAggregator,
    ProblemStatisticsAggregator,
    ExperimentStatisticsAggregator,
    BestTemplateRecommender
)


def test_template_stats_model():
    """Test TemplateStats behaves correctly."""
    print("Testing TemplateStats model...")

    stats = TemplateStats(template_name="test_template.j2")

    # Add results
    result1 = {
        'template': 'test_template.j2',
        'problem': 'problem1',
        'all_correct_rate': 0.8,
        'at_least_one_correct_rate': 0.9,
        'individual_duration': 100.0
    }
    result2 = {
        'template': 'test_template.j2',
        'problem': 'problem2',
        'all_correct_rate': 0.6,
        'at_least_one_correct_rate': 0.7,
        'individual_duration': 150.0
    }

    stats.add_result(result1)
    stats.add_result(result2)

    # Test calculations
    assert stats.average_all_correct_rate() == 0.7, "Expected avg 0.7"
    assert stats.average_partial_rate() == 0.8, "Expected avg 0.8"
    assert stats.average_duration() == 125.0, "Expected avg 125.0"
    assert stats.excellent_count() == 1, "Expected 1 excellent"
    assert stats.good_count() == 1, "Expected 1 good"
    assert stats.weighted_score() == 0.72, "Expected weighted score 0.72"

    # Test to_dict
    result_dict = stats.to_dict()
    assert result_dict['template'] == 'test_template.j2'
    assert result_dict['total_problems'] == 2

    print("  ✅ TemplateStats model tests passed")


def test_problem_stats_model():
    """Test ProblemStats behaves correctly."""
    print("Testing ProblemStats model...")

    stats = ProblemStats(problem_name="test_problem")

    # Add results
    result1 = {
        'template': 'template1.j2',
        'problem': 'test_problem',
        'all_correct_rate': 0.8,
        'at_least_one_correct_rate': 0.9
    }
    result2 = {
        'template': 'template2.j2',
        'problem': 'test_problem',
        'all_correct_rate': 0.6,
        'at_least_one_correct_rate': 0.7
    }

    stats.add_result(result1)
    stats.add_result(result2)

    # Test calculations
    assert stats.average_all_correct_rate() == 0.7
    assert stats.average_partial_rate() == 0.8
    assert len(stats.templates_used) == 2

    best = stats.best_template_result()
    assert best['template'] == 'template1.j2'
    assert best['all_correct_rate'] == 0.8

    print("  ✅ ProblemStats model tests passed")


def test_experiment_stats_model():
    """Test ExperimentStats behaves correctly."""
    print("Testing ExperimentStats model...")

    stats = ExperimentStats(experiment_key="template1.j2|problem1")

    # Add results
    result1 = {
        'all_correct_rate': 0.8,
        'experiment_timestamp': '2025-09-29_10:00:00'
    }
    result2 = {
        'all_correct_rate': 0.6,
        'experiment_timestamp': '2025-09-29_11:00:00'
    }

    stats.add_result(result1)
    stats.add_result(result2)

    # Test calculations
    assert stats.average_success_rate() == 0.7
    assert stats.max_success_rate() == 0.8
    assert stats.runs == 2
    assert stats.latest_timestamp() == '2025-09-29_11:00:00'

    print("  ✅ ExperimentStats model tests passed")


def test_template_statistics_aggregator():
    """Test TemplateStatisticsAggregator behaves correctly."""
    print("Testing TemplateStatisticsAggregator...")

    results = [
        {
            'template': 'template1.j2',
            'problem': 'problem1',
            'all_correct_rate': 0.8,
            'at_least_one_correct_rate': 0.9,
            'individual_duration': 100.0
        },
        {
            'template': 'template1.j2',
            'problem': 'problem2',
            'all_correct_rate': 0.6,
            'at_least_one_correct_rate': 0.7,
            'individual_duration': 150.0
        },
        {
            'template': 'template2.j2',
            'problem': 'problem1',
            'all_correct_rate': 0.9,
            'at_least_one_correct_rate': 0.95,
            'individual_duration': 120.0
        }
    ]

    aggregator = TemplateStatisticsAggregator(results)
    stats = aggregator.aggregate()

    # Check aggregation
    assert len(stats) == 2, "Expected 2 templates"
    assert 'template1.j2' in stats
    assert 'template2.j2' in stats

    template1_stats = stats['template1.j2']
    assert template1_stats.average_all_correct_rate() == 0.7
    assert template1_stats.problem_count == 2

    # Test ranking
    ranking = aggregator.aggregate_to_ranking()
    assert len(ranking) == 2
    assert ranking[0]['template'] == 'template2.j2', "template2 should rank first"
    assert ranking[0]['score'] > ranking[1]['score']

    print("  ✅ TemplateStatisticsAggregator tests passed")


def test_problem_statistics_aggregator():
    """Test ProblemStatisticsAggregator behaves correctly."""
    print("Testing ProblemStatisticsAggregator...")

    results = [
        {
            'template': 'template1.j2',
            'problem': 'problem1',
            'all_correct_rate': 0.8,
            'at_least_one_correct_rate': 0.9
        },
        {
            'template': 'template2.j2',
            'problem': 'problem1',
            'all_correct_rate': 0.6,
            'at_least_one_correct_rate': 0.7
        },
        {
            'template': 'template1.j2',
            'problem': 'problem2',
            'all_correct_rate': 0.3,
            'at_least_one_correct_rate': 0.4
        }
    ]

    aggregator = ProblemStatisticsAggregator(results)
    stats = aggregator.aggregate()

    # Check aggregation
    assert len(stats) == 2, "Expected 2 problems"
    assert 'problem1' in stats
    assert 'problem2' in stats

    problem1_stats = stats['problem1']
    assert problem1_stats.average_all_correct_rate() == 0.7
    assert len(problem1_stats.templates_used) == 2

    # Test analysis with difficulty
    analysis = aggregator.aggregate_to_analysis()
    assert len(analysis) == 2
    assert analysis[0]['problem'] == 'problem1', "problem1 should rank first"
    assert 'difficulty' in analysis[0]

    print("  ✅ ProblemStatisticsAggregator tests passed")


def test_experiment_statistics_aggregator():
    """Test ExperimentStatisticsAggregator behaves correctly."""
    print("Testing ExperimentStatisticsAggregator...")

    results = [
        {
            'template': 'template1.j2',
            'problem': 'problem1',
            'all_correct_rate': 0.8,
            'experiment_timestamp': '2025-09-29_10:00:00'
        },
        {
            'template': 'template1.j2',
            'problem': 'problem1',
            'all_correct_rate': 0.6,
            'experiment_timestamp': '2025-09-29_11:00:00'
        },
        {
            'template': 'template2.j2',
            'problem': 'problem1',
            'all_correct_rate': 0.9,
            'experiment_timestamp': '2025-09-29_12:00:00'
        }
    ]

    aggregator = ExperimentStatisticsAggregator(results)
    stats = aggregator.aggregate()

    # Check aggregation
    assert len(stats) == 2, "Expected 2 experiment combinations"
    assert 'template1.j2|problem1' in stats
    assert 'template2.j2|problem1' in stats

    exp1_stats = stats['template1.j2|problem1']
    assert exp1_stats.runs == 2
    assert exp1_stats.average_success_rate() == 0.7

    print("  ✅ ExperimentStatisticsAggregator tests passed")


def test_best_template_recommender():
    """Test BestTemplateRecommender behaves correctly."""
    print("Testing BestTemplateRecommender...")

    results = [
        {
            'template': 'template1.j2',
            'problem': 'problem1',
            'all_correct_rate': 0.8,
            'at_least_one_correct_rate': 0.9
        },
        {
            'template': 'template2.j2',
            'problem': 'problem1',
            'all_correct_rate': 0.6,
            'at_least_one_correct_rate': 0.7
        },
        {
            'template': 'template3.j2',
            'problem': 'problem1',
            'all_correct_rate': 0.55,
            'at_least_one_correct_rate': 0.65
        }
    ]

    recommender = BestTemplateRecommender(results)
    recommendation = recommender.recommend_for_problem('problem1')

    # Check recommendation
    assert recommendation['best_template'] == 'template1.j2'
    assert recommendation['best_score'] == 0.8
    assert len(recommendation['alternatives']) == 2, "Expected 2 alternatives"
    assert recommendation['alternatives'][0]['template'] == 'template2.j2'

    # Test recommend_all
    all_recommendations = recommender.recommend_all()
    assert 'problem1' in all_recommendations

    print("  ✅ BestTemplateRecommender tests passed")


def test_backward_compatibility():
    """Test that new classes produce same results as old code."""
    print("Testing backward compatibility...")

    # Sample data matching old code structure
    results = [
        {
            'template': 'template1.j2',
            'problem': 'problem1',
            'all_correct_rate': 0.8,
            'at_least_one_correct_rate': 0.9,
            'individual_duration': 100.0
        },
        {
            'template': 'template1.j2',
            'problem': 'problem2',
            'all_correct_rate': 0.6,
            'at_least_one_correct_rate': 0.7,
            'individual_duration': 150.0
        }
    ]

    # Test template aggregation produces same format
    aggregator = TemplateStatisticsAggregator(results)
    ranking = aggregator.aggregate_to_ranking()

    # Check all expected keys are present
    assert 'template' in ranking[0]
    assert 'avg_all_correct_rate' in ranking[0]
    assert 'avg_partial_rate' in ranking[0]
    assert 'excellent_problems' in ranking[0]
    assert 'good_problems' in ranking[0]
    assert 'total_problems' in ranking[0]
    assert 'avg_duration' in ranking[0]
    assert 'score' in ranking[0]

    # Verify calculations match old logic
    assert ranking[0]['avg_all_correct_rate'] == 0.7
    assert ranking[0]['avg_partial_rate'] == 0.8
    assert ranking[0]['avg_duration'] == 125.0
    assert ranking[0]['score'] == 0.72  # 0.7 * 0.8 + 0.8 * 0.2

    print("  ✅ Backward compatibility tests passed")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪  Running Statistics Aggregator Validation Tests")
    print("="*60 + "\n")

    test_template_stats_model()
    test_problem_stats_model()
    test_experiment_stats_model()
    test_template_statistics_aggregator()
    test_problem_statistics_aggregator()
    test_experiment_statistics_aggregator()
    test_best_template_recommender()
    test_backward_compatibility()

    print("\n" + "="*60)
    print("✅  All tests passed! Statistics aggregator classes are working correctly.")
    print("="*60 + "\n")
    print("📝 Next step: Update run_all_problems.py to use these classes")
    print("   This will eliminate 200+ lines of duplication!")
    print()


if __name__ == "__main__":
    main()