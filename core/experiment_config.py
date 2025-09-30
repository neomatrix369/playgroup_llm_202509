"""
Experiment configuration discovery and resolution.

Following Object Calisthenics and SOLID principles:
- Single Responsibility: Handles template/problem discovery and selection only
- Small class: Focused methods for specific tasks
- DRY: Eliminates need for repeated discovery logic

Extracts 77 lines from BatchExperimentRunner for better separation of concerns.
"""

from pathlib import Path
from typing import List, Optional


class ExperimentConfigResolver:
    """
    Discovers and resolves experiment configuration (templates and problems).

    Separates configuration logic from experiment execution logic.
    """

    @staticmethod
    def discover_templates() -> List[str]:
        """Discover available J2 template files."""
        prompts_dir = Path("prompts")
        if not prompts_dir.exists():
            raise FileNotFoundError("prompts directory not found")

        templates = []
        for j2_file in prompts_dir.glob("*.j2"):
            # Skip templates with 'spelke' in name as per original script
            if 'spelke' not in j2_file.name.lower():
                templates.append(j2_file.name)

        # Sort to ensure consistent ordering
        return sorted(templates)

    @staticmethod
    def get_default_problems() -> List[str]:
        """Get default problem IDs from run_all_problems.py."""
        return [
            "0d3d703e",  # fixed colour mapping, 3x3 grid min
            "08ed6ac7",  # coloured in order of height, 9x9 grid min
            "178fcbfb",  # dots form coloured lines, 9x9 grid min
            "9565186b",  # most frequent colour wins, 3x3 grid min
            "0a938d79",  # dots form repeated coloured lines, 9x22 grid min
        ]

    @classmethod
    def resolve_templates(cls, selection: Optional[str]) -> List[str]:
        """Resolve template selection from string or indices."""
        available_templates = cls.discover_templates()

        if not selection:
            return available_templates

        selected = []
        for item in selection.split(","):
            item = item.strip()
            if item.isdigit():
                # Index selection
                idx = int(item)
                if 0 <= idx < len(available_templates):
                    selected.append(available_templates[idx])
                else:
                    print(f"Warning: Template index {idx} out of range")
            else:
                # Name selection
                if item in available_templates:
                    selected.append(item)
                else:
                    print(f"Warning: Template '{item}' not found")

        return selected

    @classmethod
    def resolve_problems(cls, selection: Optional[str]) -> List[str]:
        """Resolve problem selection from string or indices."""
        default_problems = cls.get_default_problems()

        if not selection:
            return default_problems

        selected = []
        for item in selection.split(","):
            item = item.strip()
            if item.isdigit():
                # Check if it's an index or a problem ID
                if len(item) <= 2:  # Assume index if short number
                    idx = int(item)
                    if 0 <= idx < len(default_problems):
                        selected.append(default_problems[idx])
                    else:
                        print(f"Warning: Problem index {idx} out of range")
                else:
                    # Treat as problem ID
                    selected.append(item)
            else:
                # Direct problem ID
                selected.append(item)

        return selected