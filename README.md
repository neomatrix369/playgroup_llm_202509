# playgroup_llm_202509

LLM-powered solver for ARC-AGI problems using prompt engineering and reflexion methods.

Created for playgroup attendees on 2025-09 in London.

**License:** MIT
**Links:** [Playgroup Slack](https://the-playgroup.slack.com/) | [GitHub](https://github.com/ianozsvald/playgroup_llm_202509)

---

## Quick Start

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Verify installation
pytest
python src/arc_agi/prompt.py --help
```

## Project Structure

```
playgroup_llm_202509/
├── src/arc_agi/              # Main package
│   ├── methods/              # Experiment methods (method1, method2, variants)
│   ├── core/                 # Orchestration and coordination
│   ├── domain/               # Domain models
│   ├── analysis/             # Result analysis
│   ├── output/               # Output formatting
│   └── *.py                  # Core modules (prompt, run_code, utils, etc.)
├── scripts/                  # Executable scripts
│   ├── run_all_problems.py   # Batch experiment runner
│   ├── analysis.py           # Statistical analysis
│   └── create_retroactive_checkpoint.py
├── tests/                    # Test suite (50 tests)
├── prompts/                  # Jinja2 templates
├── docs/                     # Documentation (see below)
└── pyproject.toml            # Package configuration
```

## Basic Usage

### Generate a Prompt
```bash
python src/arc_agi/prompt.py -p 0d3d703e -t baseline_justjson.j2
```

### Run Code Solution
```bash
python src/arc_agi/run_code.py -p 0d3d703e -c example_solutions/ex_soln_0d3d703e.py
```

### Run Single-Pass LLM Method
```bash
python src/arc_agi/methods/method1_text_prompt.py -p 0d3d703e -i 5
```

### Run Batch Experiments
```bash
python scripts/run_all_problems.py -t "baseline_justjson_enhanced.j2" -p "0d3d703e,9565186b" -i 10
```

## Documentation

Comprehensive documentation is organized by topic:

📖 **[Walkthrough](docs/WALKTHROUGH.md)** - Step-by-step tutorial for playgroup attendees
🔧 **[CLI Reference](docs/CLI_REFERENCE.md)** - Complete command-line tools documentation
🚀 **[Batch Runner Guide](docs/BATCH_RUNNER.md)** - Detailed guide to batch experiment system
📝 **[Recent Improvements](docs/IMPROVEMENTS.md)** - Changelog and feature history
⚙️ **[Development Guide](docs/DEVELOPMENT.md)** - Setup, workflows, and contributing

## Key Features

- **Multiple Methods**: Single-pass prompts (method1) and multi-iteration reflexion (method2)
- **Batch Testing**: Run systematic experiments across templates and problems
- **Automatic Retry**: Intelligent retry logic for structural code errors
- **Checkpoint System**: Resume interrupted long-running experiments
- **Statistical Analysis**: Fisher exact test for comparing experiments
- **Rich Output**: CSV, HTML reports, detailed logs, SQLite databases

## Quick Examples

<details>
<summary><b>Run 5 iterations on easy problem</b></summary>

```bash
python src/arc_agi/methods/method1_text_prompt.py -p 0d3d703e -i 5
# View results:
tail -f experiments/exp_*/experiment.log
```
</details>

<details>
<summary><b>Compare two templates statistically</b></summary>

```bash
# Run template A
python src/arc_agi/methods/method1_text_prompt.py -p 9565186b -t baseline_justjson_enhanced.j2 -i 10

# Run template B
python src/arc_agi/methods/method1_text_prompt.py -p 9565186b -t reflexion_enhanced.j2 -i 10

# Compare (assume 3/10 vs 7/10 success)
python scripts/analysis.py 3 10 7 10
```
</details>

<details>
<summary><b>Run comprehensive batch test</b></summary>

```bash
# Preview what would run
python scripts/run_all_problems.py --dry-run --verbose

# Run all templates on all problems
python scripts/run_all_problems.py -i 10 -v

# View results in HTML report
open batch_results/latest/batch_results_*.html
```
</details>

## ARC-AGI Problems

Example problems used in the project:

| Problem | Description | Difficulty |
|---------|-------------|------------|
| [0d3d703e](https://arcprize.org/play?task=0d3d703e) | Fixed colour mapping | Medium |
| [08ed6ac7](https://arcprize.org/play?task=08ed6ac7) | Coloured by height | Hard |
| [9565186b](https://arcprize.org/play?task=9565186b) | Most frequent wins | Medium |
| [178fcbfb](https://arcprize.org/play?task=178fcbfb) | Dots to lines | Very Hard |
| [0a938d79](https://arcprize.org/play?task=0a938d79) | Repeated lines | Very Hard |

Visit [arcprize.org](https://arcprize.org/) for more problems and information.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/arc_agi --cov-report=html
open htmlcov/index.html
```

**50 tests** covering core functionality, retry logic, representations, and more.

## Exploring Further

Now compare this to the EXPT (experiments) results - better prompt engineering gets us further. **How much further could we go?**

### Try Method2 Reflexion on Harder Problems

```bash
# Run reflexion method - watch explanations grow and code evolve
python src/arc_agi/methods/method2_reflexion.py -p 9565186b -t reflexion_wquotedgridcsv_excel.j2 -i 20

# Open logs and follow the growing set of explanations
tail -f experiments/exp_*/experiment.log | grep -A 5 "EXPLANATION"
```

See [Walkthrough Part 4](docs/WALKTHROUGH.md#part-4-multi-iteration-reflexion-method-method2) for detailed explanation.

### Experiment with Prompt Hinting

```bash
# Copy a good baseline template
cp prompts/baseline_wquotedgridcsv_excel.j2 prompts/my_enhanced_template.j2

# Edit to add hints, structure, or different instructions
# Test your enhanced prompt
python src/arc_agi/methods/method1_text_prompt.py -p 9565186b -t my_enhanced_template.j2 -i 3
```

See [Walkthrough Part 6](docs/WALKTHROUGH.md#part-6-prompt-engineering) for prompt engineering guide.

### Discussion Questions

**Representation Optimization:**
- Is the baseline representation suboptimal? How could it be improved?
- Look in `src/arc_agi/representations.py` - what could we add?
- Should we include grid size? Would scipy's connected components help or mislead?

**Prompt Architecture:**
- Is a 1-pass prompt optimal? Should it be split into discrete chunks?
- Break into: analysis phase → pattern identification → code generation?

**Methodology:**
- Is reflexion better than fresh attempts? When does context become noise?
- How many iterations needed for statistical significance?

See [Discussion Questions](docs/WALKTHROUGH.md#discussion-questions) for complete list.

## Development

See [Development Guide](docs/DEVELOPMENT.md) for:
- Development environment setup
- Pre-commit hooks configuration
- Testing workflows
- Code style guidelines
- Contributing instructions

## Recent Updates

**October 2025:**
- ✅ Reorganized to Python package structure (`pip install -e .`)
- ✅ Split documentation into focused guides
- ✅ All 50 tests passing after reorganization

**September 2025:**
- ✅ Automatic code regeneration with retry logic
- ✅ Checkpoint and resume system for batch experiments
- ✅ Object-oriented refactoring (64.7% code reduction)
- ✅ Comprehensive CLI documentation

See [IMPROVEMENTS.md](docs/IMPROVEMENTS.md) for complete changelog.

## API Keys

Set your OpenRouter API key:

```bash
# In .env file
OPENROUTER_API_KEY=your_key_here

# Or export in shell
export OPENROUTER_API_KEY=your_key_here
```

## Getting Help

- 📚 **Documentation**: See [docs/](docs/) directory
- 💬 **Questions**: [Playgroup Slack](https://the-playgroup.slack.com/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/ianozsvald/playgroup_llm_202509/issues)
- 📖 **Tutorial**: Start with [Walkthrough](docs/WALKTHROUGH.md)

## License

MIT License - see repository for details.

---

**Next Steps:**
1. 📖 Read the [Walkthrough](docs/WALKTHROUGH.md) for step-by-step tutorial
2. 🔧 Check [CLI Reference](docs/CLI_REFERENCE.md) for all command options
3. 🚀 Explore [Batch Runner Guide](docs/BATCH_RUNNER.md) for systematic testing
4. ⚙️ See [Development Guide](docs/DEVELOPMENT.md) to contribute

---

## Developer Quick Reference

Quick commands for development workflow. See [Development Guide](docs/DEVELOPMENT.md) for comprehensive setup.

### Environment Setup (Ian's Workflow)

```bash
# Using conda base environment
conda activate basepython312
cd /home/ian/workspace/personal/playgroup/playgroup_llm_202509
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Pre-commit Hooks

```bash
# Install and configure pre-commit
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-commit

# Pre-commit runs: isort, ruff, ruff-format, pytest
```

### File Monitoring and Auto-Testing

```bash
# Install fswatch (cross-platform file monitoring)
# macOS: brew install fswatch
# Linux: sudo apt install fswatch

# Watch for file changes (simple output)
fswatch --event Created --event Updated --event Removed -x ./*.py | \
  while read file event; do
    echo "$(date '+%Y-%m-%d %H:%M:%S')  $file  $event"
  done

# Auto-run pytest on any file change
fswatch -r -x *.py | while read file event; do
    clear
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Change detected in $file ($event)"
    pytest
    sleep 1
done
```

**See [Development Guide](docs/DEVELOPMENT.md) for:**
- Complete environment setup instructions
- Testing workflows and coverage reports
- Code style guidelines
- Contributing process
