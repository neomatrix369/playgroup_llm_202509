# playgroup_llm_202509

Created for playgroup attendees on 2025-09 in London.

License - MIT.

* https://the-playgroup.slack.com/
* https://github.com/ianozsvald/playgroup_llm_202509

# Setup

```bash
# I'm assuming you have a Python 3.12 environment setup already
# you'll see further below in 'Ian's notes' I use conda to make a plain 3.12 env, then I make this venv (on my linux machine)
python -m venv .venv
. .venv/bin/activate # activate local env
pip install -r requirements.txt

# Install the project as an editable package (required after reorganization)
pip install -e .

# Now you should be able to run pytest and the main scripts
pytest
python src/arc_agi/prompt.py --help # check it runs
python src/arc_agi/run_code.py --help # check it runs
```

## Project Structure

The project follows Python best practices with a src/ layout:

```
playgroup_llm_202509/
├── src/arc_agi/              # Main package (installed via pip install -e .)
│   ├── __init__.py
│   ├── methods/              # Experiment methods
│   │   ├── method1_text_prompt.py    # Single-pass LLM prompt method
│   │   ├── method2_reflexion.py      # Multi-iteration reflexion method
│   │   └── variants/                 # Method variants
│   ├── analysis/             # Result analysis and statistics
│   ├── core/                 # Core orchestration and coordination
│   ├── domain/               # Domain models and value objects
│   ├── output/               # Output formatting and display
│   ├── db.py                 # SQLite database operations
│   ├── litellm_helper.py     # LLM API integration
│   ├── prompt.py             # Prompt generation and rendering
│   ├── representations.py    # Grid representation formats
│   ├── run_code.py           # Code execution and validation
│   └── utils.py              # Common utilities
├── scripts/                  # Executable scripts
│   ├── run_all_problems.py   # Batch experiment runner
│   ├── analysis.py           # Analysis utilities
│   └── create_retroactive_checkpoint.py
├── tests/                    # All test files
│   ├── test_run_code.py
│   ├── test_retry_logic.py
│   ├── test_utils.py
│   └── ...
├── prompts/                  # Jinja2 prompt templates
├── arc_data/                 # ARC-AGI dataset
├── pyproject.toml            # Package configuration and tool settings
└── requirements.txt          # Python dependencies
```

**Key Points:**
- The project is installed as a package via `pip install -e .`
- Import using `from arc_agi import module` or `from arc_agi.methods import method1_text_prompt`
- Scripts are in `scripts/`, modules in `src/arc_agi/`, tests in `tests/`
- Tool configurations (isort, ruff, pytest) are in `pyproject.toml`

# Walkthrough (during playgroup)

```bash
# Visit https://arcprize.org/play?task=0d3d703e
python src/arc_agi/prompt.py -t baseline_justjson.j2 -p 0d3d703e # render this problem as a prompt
python src/arc_agi/prompt.py -t baseline_wquotedgridcsv_excel.j2 -p 0d3d703e # render with different template
```

```bash
python src/arc_agi/run_code.py --help
python src/arc_agi/run_code.py -p 0d3d703e -c example_solutions/ex_soln_0d3d703e.py # run good solution
python src/arc_agi/run_code.py -p 0d3d703e -c example_solutions/ex_soln_08ed6ac7.py # run wrong solution on different problem
```

The `run_code.execute_transform` module builds a `utils.RunResult` result, this tracks if and how many of the example `initial` problems were transformed correctly to the desired `final` states. It also generates an `ExecutionOutcome` object which tracks how each initial grid is transformed, by the code.

## The problems we'll look at

* https://arcprize.org/play?task=0d3d703e  # fixed colour mapping, 3x3 grid min
* https://arcprize.org/play?task=08ed6ac7  # coloured in order of height, 9x9 grid min
* https://arcprize.org/play?task=9565186b  # most frequent colour wins, 3x3 grid min
* https://arcprize.org/play?task=178fcbfb  # dots form coloured lines, 9x9 grid min
* https://arcprize.org/play?task=0a938d79  # dots form repeated coloured lines, 9x22 grid min
* https://arcprize.org/play?task=1a07d186  # dots attach to same coloured line, 14x15 grid min (bonus - hard!)

### run method1 with the default prompt on an easy problem for 5 iterations

```bash
# run the basic method with the default prompt for 5 iterations
python src/arc_agi/methods/method1_text_prompt.py --help # see the arg description
python src/arc_agi/methods/method1_text_prompt.py -p 0d3d703e -i 5  # maybe 2-3 minutes and 20% correctness?

# This is equivalent to the fully formed version which selects the prompt and model to run:
# python src/arc_agi/methods/method1_text_prompt.py -p 0d3d703e -t baseline_justjson.j2 -m openrouter/deepseek/deepseek-chat-v3-0324 -i 5

# You can examine the experiment.log logfile (path detailed at the top of stdout)
# You can also open the SQLite database (path shown in stdout):
sqlite3 experiments/exp_TIMESTAMP/experiments.db
sqlite3> .schema
sqlite3> select code from experiments where all_train_transformed_correctly=true;
sqlite3> select final_explanation from experiments where all_train_transformed_correctly=false;
# Note: if it got an explanation right, but wrote bad code (e.g. with a SyntaxError), then it won't transform correctly

# In method1's run_experiment function we receive an object after trying a proposed solution
# execute_transform() returns a tuple of (RunResult, ExecutionOutcomes, exception_message)
# - RunResult: code_ran_on_all_inputs, transform_ran_and_matched_for_all_inputs, etc.
# - ExecutionOutcomes: list of ExecutionOutcome objects, one per train example
# - exception_message: error details if execution failed, None otherwise
#
# Automatic Code Regeneration with Retry Logic:
# The should_request_regeneration() function analyzes exceptions to categorize errors:
# - STRUCTURAL errors (missing transform, wrong signature, data instead of code): AUTO-RETRY
# - LOGIC errors (AssertionError, IndexError, ZeroDivisionError): NO RETRY (needs debugging)
# - SYNTAX errors: AUTO-RETRY if severe (code too short), otherwise debugging preferred
#
# When structural errors occur, the system automatically retries up to 3 times:
# 1. Attempt 1: Initial LLM call returns bad code → structural error detected
# 2. Attempt 2: Retry same prompt → system logs "Retrying (attempt 2/3)..."
# 3. Attempt 3: Final retry → if still fails, logs "Failed after 3 attempts, giving up"
# 4. System continues to next iteration (does not block entire experiment)
#
# This prevents wasted iterations on structural failures like LLM returning data instead of code.
```

## Batch Experiment Runner

The `scripts/run_all_problems.py` script provides a comprehensive batch experiment runner for systematic ARC-AGI testing across multiple templates and problems.

### Features

- **Multi-template and multi-problem testing**: Run experiments across all combinations of templates and problems
- **Flexible selection**: Choose specific templates and problems by name or index
- **Comprehensive timing tracking**: Global, template-level, and individual test timing
- **Real-time progress reporting**: Live feedback with timestamps and success indicators
- **Multi-format output**: Console tables, CSV files, HTML reports, and detailed logs
- **Robust error handling**: Graceful fallback when parallel execution fails
- **Success rate analysis**: Detailed performance breakdown with visual indicators

### Command-Line Options

```bash
# Core options
-m METHOD, --method METHOD          # Method module to use (default: method1_text_prompt)
--model MODEL                       # Model name (default: openrouter/deepseek/deepseek-chat-v3-0324)
-i ITERATIONS, --iterations N       # Number of iterations per test (default: 1)
-t TEMPLATES, --templates LIST      # Comma-separated template names or indices
-p PROBLEMS, --problems LIST        # Comma-separated problem IDs or indices
-o OUTPUT_DIR, --output-dir DIR     # Output directory (default: batch_results)

# Execution control
--dry-run                           # Show what would be run without executing
-v, --verbose                       # Verbose output
--fail-fast                         # Stop on first error (default: continue)

# Checkpoint and resume
--no-checkpoint                     # Disable automatic checkpoint saving
--resume                            # Force resume from last checkpoint
--no-resume                         # Start fresh, ignore existing checkpoints

# Analysis
--summarise-experiments             # Analyze existing results and generate summary statistics
```

### Usage Examples

```bash
# Run all templates and problems (comprehensive test)
python scripts/run_all_problems.py

# Run specific templates and problems with multiple iterations
python scripts/run_all_problems.py -t "baseline_justjson_enhanced.j2,reflexion_enhanced.j2" -p "0d3d703e,08ed6ac7" -i 5

# Test with different method module and model
python scripts/run_all_problems.py --method method2_reflexion --model openrouter/deepseek/deepseek-chat-v3-0324

# Dry run to preview what would be executed
python scripts/run_all_problems.py --dry-run --verbose

# Quick test of a single template-problem combination
python scripts/run_all_problems.py -t "baseline_justjson_enhanced.j2" -p "0d3d703e" -i 1

# Resume interrupted experiment
python scripts/run_all_problems.py --resume

# Analyze existing experiments and generate summary statistics
python scripts/run_all_problems.py --summarise-experiments --verbose

# Fail fast mode - stop on first error
python scripts/run_all_problems.py --fail-fast -i 10
```

### Checkpoint and Resume

When running long experiments, checkpoints are automatically saved every N experiments. If interrupted:

```bash
# Resume from last checkpoint (will prompt automatically)
python scripts/run_all_problems.py

# Force resume without prompting
python scripts/run_all_problems.py --resume

# Create checkpoint for interrupted experiment (retroactive)
python scripts/create_retroactive_checkpoint.py batch_results/20250929_194041
```

### Output Structure

Results are saved in timestamped directories under `batch_results/` containing:
- **CSV file**: Structured data for analysis (`batch_results_TIMESTAMP.csv`)
- **HTML report**: Visual dashboard with color-coded results (`batch_results_TIMESTAMP.html`)
- **Summary log**: Detailed execution log with timing breakdown (`batch_summary_TIMESTAMP.log`)
- **Checkpoint file**: Resume point for interrupted experiments (`checkpoint.json`)

### Performance Indicators

- 🎯 **Excellent (≥80%)**: Grade A performance
- ✅ **Good (50-79%)**: Grade B performance
- ⚠️ **Partial**: Some success but inconsistent
- ❌ **Poor (<50%)**: Grade F, needs improvement

## Other Scripts and Tools

### prompt.py - Prompt Generation and Rendering

Generate prompts for ARC-AGI problems using Jinja2 templates.

**Options:**
```bash
-p, --problem_name    # Problem ID (default: 9565186b)
-t, --template_name   # Template in ./prompts/ (default: baseline_justjson.j2)
```

**Examples:**
```bash
# Generate prompt for specific problem and template
python src/arc_agi/prompt.py -p 0d3d703e -t baseline_justjson.j2

# Use different template
python src/arc_agi/prompt.py -p 08ed6ac7 -t baseline_wquotedgridcsv_excel.j2
```

### run_code.py - Code Execution and Validation

Execute Python code solutions on ARC-AGI problems and validate results.

**Options:**
```bash
-p, --problem_name    # Problem ID (default: 9565186b)
-c, --code_filename   # Path to Python file with transform function
```

**Examples:**
```bash
# Run solution on specific problem
python src/arc_agi/run_code.py -p 0d3d703e -c example_solutions/ex_soln_0d3d703e.py

# Validate code against different problem
python src/arc_agi/run_code.py -p 08ed6ac7 -c /tmp/my_solution.py
```

### method1_text_prompt.py - Single-Pass LLM Method

Single-pass LLM prompt method for solving ARC-AGI problems.

**Options:**
```bash
-p, --problem_name    # Problem ID (default: 9565186b)
-t, --template_name   # Template in ./prompts/ (default: baseline_justjson.j2)
-i, --iterations      # Number of iterations to run (default: 1)
-m, --model_name      # OpenRouter model name (default: openrouter/deepseek/deepseek-chat-v3-0324)
```

**Examples:**
```bash
# Run 5 iterations on specific problem
python src/arc_agi/methods/method1_text_prompt.py -p 0d3d703e -i 5

# Use different template and model
python src/arc_agi/methods/method1_text_prompt.py -p 9565186b -t reflexion_enhanced.j2 -m openrouter/deepseek/deepseek-chat-v3-0324 -i 10
```

### method2_reflexion.py - Multi-Iteration Reflexion Method

Multi-iteration reflexion method with self-correction and explanation accumulation.

**Options:**
```bash
-p, --problem_name    # Problem ID (default: 9565186b)
-t, --template_name   # Template in ./prompts/ (default: baseline_justjson.j2)
-i, --iterations      # Number of iterations to run (default: 1)
-m, --model_name      # OpenRouter model name (default: openrouter/deepseek/deepseek-chat-v3-0324)
```

**Examples:**
```bash
# Run reflexion method with 20 iterations
python src/arc_agi/methods/method2_reflexion.py -p 9565186b -t reflexion_wquotedgridcsv_excel.j2 -i 20

# Test on harder problem
python src/arc_agi/methods/method2_reflexion.py -p 178fcbfb -i 15
```

### analysis.py - Statistical Analysis

Perform Fisher exact test to compare two experiments statistically.

**Usage:**
```bash
python scripts/analysis.py <successes1> <total1> <successes2> <total2>
```

**Examples:**
```bash
# Compare two experiments: 32/50 vs 43/50
python scripts/analysis.py 32 50 43 50

# Compare different sized experiments: 65/300 vs 77/300
python scripts/analysis.py 65 300 77 300
```

**Output:**
- Fisher exact test statistic (odds ratio)
- P-value (statistical significance)
- Interpretation (significant if p < 0.05)

### create_retroactive_checkpoint.py - Checkpoint Recovery

Create checkpoint files for interrupted experiments to enable resumption.

**Usage:**
```bash
python scripts/create_retroactive_checkpoint.py <experiment_directory>
```

**Examples:**
```bash
# Create checkpoint from interrupted experiment
python scripts/create_retroactive_checkpoint.py batch_results/20250929_194041

# Then resume the experiment
python scripts/run_all_problems.py --resume
```

**What it does:**
1. Parses experiment logs for timing and configuration
2. Extracts results from SQLite database
3. Creates valid `checkpoint.json` file
4. Generates resume instructions

### Recent Improvements

#### Project Reorganization (2025-10)
- **Python Package Structure**: Reorganized project following Python best practices
  - Created `src/arc_agi/` package structure with proper `__init__.py` files
  - Moved all modules to `src/arc_agi/` (methods, core, domain, analysis, output)
  - Moved executable scripts to `scripts/` directory
  - Moved all tests to `tests/` directory
  - Added `pyproject.toml` for package configuration (PEP 517/518 compliant)
  - Configured isort with `known_first_party = ["arc_agi"]` to preserve package imports
  - Configured ruff and pytest in `pyproject.toml`
  - Installed as editable package: `pip install -e .`
  - Updated all imports to use `from arc_agi import module` pattern
  - Fixed template paths in tests to use project root
  - **All 50 tests passing** after reorganization

#### Code Quality and Architecture (2025-09)
- **Object Calisthenics Refactoring**: Extracted 16+ classes following SOLID principles for better maintainability
  - ExperimentArgumentParser, ExperimentExecutor, ExperimentAggregator, ExperimentSummarizer
  - ExperimentCoordinator, TimingTracker, ExperimentResults, ExperimentContext
  - 64.7% code reduction in main orchestrator (1,647 → 582 lines)
- **Automatic Code Regeneration**: Intelligent LLM output validation with automatic retry (up to 3 attempts)
  - Detects when LLM returns data instead of code
  - Categorizes errors: STRUCTURAL (auto-retry), LOGIC (debug), SYNTAX (context-dependent)
  - Retries same prompt automatically for structural failures
  - Prevents wasted experiment iterations on malformed code
  - Sanitizes Unicode artifacts (arrows →, smart quotes "", etc.)
- **Type Safety**: Enforces np.ndarray return type from transform functions (rejects lists/primitives)

#### Checkpoint and Resume System
- **Interrupted Experiment Recovery**: Resume long-running batch experiments from last checkpoint
  - Automatic checkpoint creation every N experiments
  - Retroactive checkpoint creation for interrupted runs
  - Detailed progress display showing completed vs pending work
  - Prevents duplicate execution and maintains result integrity

#### Robust Error Handling
- **Fixed file path handling**: Resolves "'str' object has no attribute 'name'" errors
- **Improved execution resilience**: Fallback to serial execution when parallel processing fails
- **Multi-strategy code extraction**: 3-tier fallback for extracting code from LLM responses
- **Better dependency management**: Added missing scikit-image dependency
- **Enhanced logging**: Proper logger initialization across modules

#### Testing
- **Core functionality tests**: `test_run_code.py` - Code execution, validation, sanitization
- **Retry logic tests**: `test_retry_logic.py` - Automatic regeneration on structural errors
- **Run tests**: `pytest` or `pytest test_retry_logic.py -v` for specific test files
- **Coverage report**: `python -m pytest --cov=. --cov-report=html` (view with `open htmlcov/index.html`) 

Now compare this to the EXPT (experiments) results - Ian on screen - better prompt sort of gets us further.

How much further could we go?

### run method2 on a harder problem, observe the logs

```bash
python src/arc_agi/methods/method2_reflexion.py -t reflexion_wquotedgridcsv_excel.j2 -p 9565186b -i 20
# Now open the logs and follow them - watch the growing set of (5) explanations and more-complex code solutions
# Is this a good direction?
```

### hinting - use method1 again on a copy of a prompt

```bash
# In prompts/ copy e.g. baseline_wquotedgridcsv_excel.j2 (a good one)
# We'll discuss what we could add...
python src/arc_agi/methods/method1_text_prompt.py -p 9565186b -t your_hinted_prompt.j2 -i 3
```

### thoughts

* is the baseline representation suboptimal? how could it be improved? look in `representations.py` and extend?
* is a 1-pass prompt a good idea? should it be split into discrete chunks?
* in `representations.py` we could add grid size, should we?
* I've never tried scipy's connected components, would that help? might it mislead?


### code dev notes

`python -m pytest --cov=. --cov-report=html` will run an HTML coverage report, view with `open htmlcov/index.html` in a browser.

`pytest` will run all your tests. If you setup `pre-commit` then any commits will kick off `isort`, `ruff` and `pytest`.

# Setup notes


# Ian's stuff below here

## Setup notes by Ian for Ian

```
conda activate basepython312
cd /home/ian/workspace/personal/playgroup/playgroup_llm_202509
python -m venv .venv
. .venv/bin/activate # activate local env
pip install -r requirements.txt

# chatgpt recommendation for folder monitoring, cross platform
sudo apt install fswatch
```

### pre-commit

```
# pre-commit, on .pre-commit-config.yaml
# note we don't need ruff & isort in the main requirements.txt file
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-commit
```

### track folder file changes

```
# simpler output
fswatch --event Created --event Updated --event Removed -x ./*.py | while read file event; do     echo "$(date '+%Y-%m-%d %H:%M:%S')  $file  $event"; done
# works, very verbose, multiple reports per single change
fswatch -x ./*.py | while read file event; do     echo "$(date '+%Y-%m-%d %H:%M:%S')  $file  $event"; done
```

### run pytest every time a file changes

```
#fswatch -r -x tests/ src/ | while read file event; do
# run tests after a file change
fswatch -r -x *.py | while read file event; do
    clear
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Change detected in $file ($event)"
    pytest
    sleep 1
done
```
