import os

from jinja2 import Environment, FileSystemLoader

from arc_agi import representations, utils


def test_baseline_template_rendering():
    # Get the examples for problem 956
    patterns = utils.get_examples("9565186b")
    problems = patterns["train"]

    # Set up Jinja2 environment - templates are in project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    template_dir = os.path.join(project_root, "prompts")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("baseline_justjson.j2")

    # Render the template
    rendered = template.render(
        patterns_input_output=problems, make_grid_plain=representations.make_grid_plain
    )

    # Basic assertions about the rendered content
    assert "You are a clever problem solving machine" in rendered
    assert "<EXPLANATION>" in rendered
    assert "def transform(initial):" in rendered
    assert "import numpy as np" in rendered

    # Check that each example is included
    for problem in problems:
        assert str(problem["input"]) in rendered
        assert str(problem["output"]) in rendered

    print("Template rendered successfully with all required components")
