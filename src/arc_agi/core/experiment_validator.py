"""
Experiment validation module for pre-flight checks.

Handles:
- Template file validation
- Problem data validation
- Method module validation
- Configuration validation

Following SOLID principles:
- Single Responsibility: Only handles validation
- Open/Closed: Extensible for additional validation rules

Extracted from BatchExperimentRunner to reduce complexity and improve testability.
"""

import argparse
from typing import Any, Callable, List, Tuple

from arc_agi import utils
from arc_agi.core.experiment_config import ExperimentConfigResolver


class ExperimentValidator:
    """Validates experiment prerequisites and configuration.

    Centralizes validation logic that was in BatchExperimentRunner,
    reducing the main class and enabling independent testing of validation rules.
    """

    def __init__(self, log_callback: Callable[[str], None] = None):
        """Initialize experiment validator.

        Args:
            log_callback: Optional callback function for logging with timestamps
        """
        self.log = log_callback if log_callback else lambda msg: print(msg)

    def validate_all(
        self,
        templates_to_use: List[str],
        problems_to_use: List[str],
        method_module: Any,
        args: argparse.Namespace = None,
    ) -> Tuple[bool, List[str]]:
        """Run all validation checks before starting experiments.

        Args:
            templates_to_use: List of template names to validate
            problems_to_use: List of problem IDs to validate
            method_module: Method module to validate
            args: Optional command line arguments (for future use)

        Returns:
            Tuple of (validation_passed: bool, errors: List[str])
        """
        errors = []

        self.log("🔍 Running pre-flight validation...")

        # 1. Validate templates exist
        template_errors = self.validate_templates(templates_to_use)
        errors.extend(template_errors)

        # 2. Validate at least one problem can be loaded
        problem_errors = self.validate_problems(problems_to_use)
        errors.extend(problem_errors)

        # 3. Validate method module has required function
        module_errors = self.validate_method_module(method_module)
        errors.extend(module_errors)

        # Note: Database and API validation will happen during first experiment setup
        # We skip them here to avoid argument parsing conflicts with do_first_setup()
        print("  ✓ Database and API validation will occur during experiment setup")

        if errors:
            print("\n❌ PRE-FLIGHT VALIDATION FAILED:")
            for error in errors:
                print(f"   ✗ {error}")
            return False, errors
        else:
            print("\n✅ All pre-flight checks passed!")
            return True, []

    def validate_templates(self, templates_to_use: List[str]) -> List[str]:
        """Validate that template files exist.

        Args:
            templates_to_use: List of template names to validate

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        print("  ✓ Checking templates...")
        available_templates = ExperimentConfigResolver.discover_templates()

        for template in templates_to_use:
            if template not in available_templates:
                errors.append(f"Template not found: {template}")
            else:
                print(f"    ✓ {template}")

        return errors

    def validate_problems(self, problems_to_use: List[str]) -> List[str]:
        """Validate that problems can be loaded.

        Args:
            problems_to_use: List of problem IDs to validate

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        print("  ✓ Checking problem data access...")

        if problems_to_use:
            try:
                # Test loading the first problem as a sanity check
                test_problem = utils.get_examples(problems_to_use[0])
                if not test_problem:
                    errors.append(f"Problem {problems_to_use[0]} loaded but returned empty data")
                else:
                    print(f"    ✓ Successfully loaded problem: {problems_to_use[0]}")
                    print(f"      - Train examples: {len(test_problem.get('train', []))}")
                    print(f"      - Test examples: {len(test_problem.get('test', []))}")
            except Exception as e:
                errors.append(f"Failed to load problem {problems_to_use[0]}: {str(e)}")

        return errors

    def validate_method_module(self, method_module: Any) -> List[str]:
        """Validate that method module has required functions.

        Args:
            method_module: Method module to validate (can be None in dry-run mode)

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        print("  ✓ Checking method module...")

        # Skip validation if method_module is None (e.g., in dry-run mode)
        if method_module is None:
            print("    ⚠ Method module validation skipped (dry-run mode)")
            return errors

        if not hasattr(method_module, "run_experiment_for_iterations"):
            errors.append("Method module missing 'run_experiment_for_iterations' function")
        else:
            print(f"    ✓ Method module loaded: {method_module.__file__}")

        return errors

    def validate_configuration(self, config: dict) -> List[str]:
        """Validate experiment configuration dictionary.

        This is a placeholder for future configuration validation.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        # Future: Add validation for:
        # - Model availability
        # - API keys
        # - Output directory permissions
        # - Resource limits
        # etc.

        return errors
