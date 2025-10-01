# Development Guide

Setup instructions and development workflows for contributing to the project.

## Development Environment Setup

### Prerequisites

- Python 3.12+
- Git
- Virtual environment tool (venv or conda)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/ianozsvald/playgroup_llm_202509.git
cd playgroup_llm_202509

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project as editable package
pip install -e .

# Verify installation
pytest
python src/arc_agi/prompt.py --help
```

### With Conda (Ian's Setup)

```bash
# Create base Python 3.12 environment
conda activate basepython312

# Navigate to project
cd /home/ian/workspace/personal/playgroup/playgroup_llm_202509

# Create venv from conda base
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Pre-commit Hooks

### Installation

```bash
# Install pre-commit tool
pip install pre-commit

# Install git hook scripts
pre-commit install
pre-commit install --hook-type pre-commit
```

### What's Included

The pre-commit hooks automatically run:
- **isort**: Sorts imports alphabetically and by type
- **ruff**: Fast Python linter
- **ruff-format**: Code formatter
- **pytest**: Runs test suite

### Configuration

Configured in `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/PyCQA/isort
    rev: 6.0.1
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.13.0
    hooks:
      - id: ruff
        args: ["--fix"]
      - id: ruff-format

  - repo: local
    hooks:
      - id: pytest
        name: Run pytest
        entry: pytest
        language: system
        pass_filenames: false
```

### Skipping Hooks (when needed)

```bash
# Skip all hooks for a commit
git commit --no-verify -m "message"

# Skip specific hook
SKIP=pytest git commit -m "message"
```

## Tool Configuration

All development tools are configured in `pyproject.toml`:

### isort Configuration

```toml
[tool.isort]
profile = "black"
line_length = 100
known_first_party = ["arc_agi"]
src_paths = ["src", "tests"]
```

### ruff Configuration

```toml
[tool.ruff]
line-length = 100
target-version = "py312"
```

### pytest Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v"
```

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_retry_logic.py

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run specific test
pytest tests/test_run_code.py::test_execute_transform

# With coverage
pytest --cov=src/arc_agi --cov-report=term-missing
```

### Coverage Reports

```bash
# Generate HTML coverage report
python -m pytest --cov=. --cov-report=html

# View in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Test Structure

```
tests/
├── test_prompt.py                 # Template rendering
├── test_refactoring.py            # Refactored code quality
├── test_representations.py        # Grid representations
├── test_retry_logic.py            # Automatic retry logic
├── test_run_code.py               # Code execution
├── test_statistics_aggregator.py  # Statistical analysis
└── test_utils.py                  # Utility functions
```

## Development Workflows

### File Monitoring (Auto-test on change)

Install fswatch for automatic test runs:

```bash
# Install fswatch
# macOS:
brew install fswatch
# Linux:
sudo apt install fswatch

# Watch for changes and run pytest
fswatch -r -x src/ tests/ | while read file event; do
    clear
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Change detected in $file ($event)"
    pytest
    sleep 1
done
```

### Simple File Change Monitor

```bash
# Simpler output - just track changes
fswatch --event Created --event Updated --event Removed -x ./*.py | \
while read file event; do
    echo "$(date '+%Y-%m-%d %H:%M:%S')  $file  $event"
done
```

### Watch Specific Directories

```bash
# Watch src/ and tests/ directories
fswatch -r -x src/ tests/ | while read file event; do
    clear
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Change in $file"
    pytest -x  # Stop on first failure
    sleep 1
done
```

## Code Style Guidelines

### Import Organization

Imports are automatically organized by isort:

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
from jinja2 import Environment

# Local
from arc_agi import utils
from arc_agi.methods import method1_text_prompt
```

### Formatting

- Line length: 100 characters
- Use double quotes for strings
- Follow PEP 8 conventions
- ruff-format handles most formatting automatically

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Modules**: `snake_case.py`

## Adding New Features

### Step-by-Step Process

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests first** (TDD approach)
   ```bash
   # Create test file
   touch tests/test_your_feature.py
   # Write failing tests
   # Then implement feature
   ```

3. **Implement feature**
   - Follow existing code structure
   - Add docstrings
   - Keep functions small and focused

4. **Run tests**
   ```bash
   pytest tests/test_your_feature.py -v
   ```

5. **Update documentation**
   - Update relevant docs in `docs/`
   - Add usage examples
   - Update CLI_REFERENCE.md if adding commands

6. **Commit changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   # Pre-commit hooks will run automatically
   ```

7. **Update IMPROVEMENTS.md**
   - Add entry with description
   - List benefits and usage

## Debugging

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Interactive Debugging

```python
# Add breakpoint in code
breakpoint()

# Or use pdb
import pdb; pdb.set_trace()
```

### Experiment Debugging

```bash
# Dry run to see what would execute
python scripts/run_all_problems.py --dry-run --verbose

# Run single iteration with verbose output
python src/arc_agi/methods/method1_text_prompt.py -p 0d3d703e -i 1 -v

# Check experiment logs
tail -f experiments/exp_*/experiment.log
```

## Common Development Tasks

### Adding a New Template

1. Create template in `prompts/`
2. Test rendering:
   ```bash
   python src/arc_agi/prompt.py -p 0d3d703e -t your_template.j2
   ```
3. Test with method:
   ```bash
   python src/arc_agi/methods/method1_text_prompt.py -p 0d3d703e -t your_template.j2 -i 3
   ```

### Adding a New Method

1. Create module in `src/arc_agi/methods/`
2. Follow method1/method2 structure
3. Add argument parser
4. Integrate with batch runner
5. Add tests
6. Document in CLI_REFERENCE.md

### Modifying Core Functionality

1. Check for existing tests
2. Add/update tests for changes
3. Run full test suite: `pytest`
4. Check for regressions
5. Update documentation

## Environment Variables

### Required

- `OPENROUTER_API_KEY` - OpenRouter API key for LLM access

### Optional

- `ARC_DATA_PATH` - Path to ARC dataset (default: `./arc_data/arc-prize-2025`)
- `EXPERIMENT_ROOT` - Root for experiment outputs (default: `experiments/`)

### Configuration File

Create `.env` file in project root:

```bash
# .env file
OPENROUTER_API_KEY=your_key_here
# ARC_DATA_PATH=/custom/path/to/arc/data
# EXPERIMENT_ROOT=/custom/experiments/path
```

## Performance Profiling

### Profile Execution Time

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Memory Profiling

```bash
pip install memory-profiler

# Add decorator to function
@profile
def my_function():
    pass

# Run with profiler
python -m memory_profiler script.py
```

## Continuous Integration

### GitHub Actions (if configured)

Typical CI pipeline:
1. Install dependencies
2. Run linters (ruff, isort)
3. Run tests with coverage
4. Build documentation
5. Create release artifacts

### Local CI Simulation

```bash
# Run everything that CI would run
pip install -r requirements.txt
pip install -e .
isort --check src/ tests/
ruff check src/ tests/
pytest --cov=src/arc_agi
```

## Troubleshooting

### Import Errors After Reorganization

```bash
# Reinstall package in editable mode
pip uninstall arc-agi
pip install -e .
```

### Pre-commit Hooks Failing

```bash
# Update hooks
pre-commit autoupdate

# Clear cache
pre-commit clean

# Reinstall
pre-commit uninstall
pre-commit install
```

### Tests Failing

```bash
# Check Python version
python --version  # Should be 3.12+

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Clear pytest cache
rm -rf .pytest_cache
pytest --cache-clear
```

## Best Practices

1. **Write tests first**: TDD approach catches issues early
2. **Keep commits atomic**: One logical change per commit
3. **Use descriptive commit messages**: Explain why, not just what
4. **Run tests locally**: Before pushing, ensure tests pass
5. **Update documentation**: Keep docs in sync with code changes
6. **Use type hints**: Add type annotations where helpful
7. **Profile before optimizing**: Measure, don't guess
8. **Ask for review**: Get feedback on significant changes

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [ruff Documentation](https://docs.astral.sh/ruff/)
- [pre-commit Documentation](https://pre-commit.com/)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)

## Getting Help

- **Project Issues**: GitHub Issues
- **Playgroup Slack**: https://the-playgroup.slack.com/
- **Documentation**: See [docs/](../docs/) directory
