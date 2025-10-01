#!/usr/bin/env python3
"""
Quick validation tests for refactored classes.

Run: python test_refactoring.py
"""

from domain.value_objects import (
    DifficultyThresholds,
    ExperimentResult,
    SuccessThresholds,
)

from analysis.difficulty_classifier import DifficultyClassifier
from analysis.performance_grader import PerformanceGrader


def test_performance_grader():
    """Test PerformanceGrader behaves correctly."""
    print("Testing PerformanceGrader...")
    grader = PerformanceGrader()

    # Test grade calculation
    assert grader.grade(0.85) == "A", "Expected 'A' for 85%"
    assert grader.grade(0.65) == "B", "Expected 'B' for 65%"
    assert grader.grade(0.55) == "C", "Expected 'C' for 55%"
    assert grader.grade(0.45) == "D", "Expected 'D' for 45%"
    assert grader.grade(0.25) == "F", "Expected 'F' for 25%"

    # Test with icons
    assert grader.grade_with_icon(0.85) == "🎯 A"
    assert grader.grade_with_icon(0.65) == "✅ B"
    assert grader.grade_with_icon(0.25) == "❌ F"

    # Test boolean checks
    assert grader.is_excellent(0.85) is True
    assert grader.is_excellent(0.65) is False
    assert grader.is_good(0.65) is True
    assert grader.is_good(0.45) is False

    print("  ✅ PerformanceGrader tests passed")


def test_difficulty_classifier():
    """Test DifficultyClassifier behaves correctly."""
    print("Testing DifficultyClassifier...")
    classifier = DifficultyClassifier()

    # Test difficulty classification
    assert classifier.classify(0.75, 0.85) == "EASY"
    assert classifier.classify(0.55, 0.70) == "MEDIUM"
    assert classifier.classify(0.35, 0.60) == "HARD"
    assert classifier.classify(0.15, 0.25) == "VERY_HARD"

    # Test simple classification
    assert classifier.classify_simple(0.75) == "EASY"
    assert classifier.classify_simple(0.55) == "MEDIUM"
    assert classifier.classify_simple(0.15) == "VERY_HARD"

    # Test icons
    assert classifier.difficulty_icon("EASY") == "🟢"
    assert classifier.difficulty_icon("HARD") == "🟠"
    assert classifier.difficulty_icon("VERY_HARD") == "🔴"

    # Test boolean checks
    assert classifier.is_easy(0.75, 0.85) is True
    assert classifier.is_easy(0.35, 0.60) is False
    assert classifier.is_hard_or_harder(0.35, 0.60) is True
    assert classifier.is_hard_or_harder(0.75, 0.85) is False

    print("  ✅ DifficultyClassifier tests passed")


def test_experiment_result():
    """Test ExperimentResult behaves correctly."""
    print("Testing ExperimentResult...")

    result = ExperimentResult(
        template="test_template.j2",
        problem="test_problem",
        all_correct_rate=0.85,
        at_least_one_correct_rate=0.95,
        duration=120.5,
        total_runs=10,
    )

    # Test grade calculation
    assert result.grade() == "A"
    assert result.grade_with_icon() == "🎯 A"

    # Test boolean checks
    assert result.is_excellent() is True
    assert result.is_good() is True
    assert result.is_successful() is True
    assert result.has_partial_success() is False

    # Test to_dict conversion
    result_dict = result.to_dict()
    assert result_dict["template"] == "test_template.j2"
    assert result_dict["all_correct_rate"] == 0.85
    assert result_dict["individual_duration"] == 120.5

    print("  ✅ ExperimentResult tests passed")


def test_thresholds_immutability():
    """Test that threshold objects are immutable."""
    print("Testing threshold immutability...")

    thresholds = SuccessThresholds()
    try:
        thresholds.EXCELLENT = 0.9  # Should fail
        assert False, "Thresholds should be immutable"
    except AttributeError:
        pass  # Expected

    difficulty_thresholds = DifficultyThresholds()
    try:
        difficulty_thresholds.EASY = 0.8  # Should fail
        assert False, "Difficulty thresholds should be immutable"
    except AttributeError:
        pass  # Expected

    print("  ✅ Threshold immutability tests passed")


def test_backward_compatibility():
    """Test that new classes produce same results as old code."""
    print("Testing backward compatibility...")

    grader = PerformanceGrader()

    # Test cases from original code (lines 466-475)
    test_cases = [
        (0.85, "A"),  # >= 0.8
        (0.65, "B"),  # >= 0.6
        (0.55, "C"),  # >= 0.5
        (0.45, "D"),  # >= 0.4
        (0.25, "F"),  # < 0.4
    ]

    for rate, expected_grade in test_cases:
        actual_grade = grader.grade(rate)
        assert actual_grade == expected_grade, (
            f"Grade mismatch for {rate}: expected {expected_grade}, got {actual_grade}"
        )

    print("  ✅ Backward compatibility tests passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪  Running Refactoring Validation Tests")
    print("=" * 60 + "\n")

    test_performance_grader()
    test_difficulty_classifier()
    test_experiment_result()
    test_thresholds_immutability()
    test_backward_compatibility()

    print("\n" + "=" * 60)
    print("✅  All tests passed! Refactored classes are working correctly.")
    print("=" * 60 + "\n")
    print("📝 Next step: Update run_all_problems.py to use these classes")
    print("   (Will NOT impact your currently running experiment)")
    print()


if __name__ == "__main__":
    main()
