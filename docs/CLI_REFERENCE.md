# Command-Line Interface Reference

Complete reference for all command-line scripts and tools in the project.

## Table of Contents

- [prompt.py - Prompt Generation](#promptpy---prompt-generation-and-rendering)
- [run_code.py - Code Execution](#run_codepy---code-execution-and-validation)
- [method1_text_prompt.py - Single-Pass Method](#method1_text_promptpy---single-pass-llm-method)
- [method2_reflexion.py - Reflexion Method](#method2_reflexionpy---multi-iteration-reflexion-method)
- [analysis.py - Statistical Analysis](#analysispy---statistical-analysis)
- [create_retroactive_checkpoint.py - Checkpoint Recovery](#create_retroactive_checkpointpy---checkpoint-recovery)

---

## prompt.py - Prompt Generation and Rendering

Generate prompts for ARC-AGI problems using Jinja2 templates.

**Location:** `src/arc_agi/prompt.py`

### Options

```bash
-p, --problem_name    # Problem ID (default: 9565186b)
-t, --template_name   # Template in ./prompts/ (default: baseline_justjson.j2)
```

### Examples

```bash
# Generate prompt for specific problem and template
python src/arc_agi/prompt.py -p 0d3d703e -t baseline_justjson.j2

# Use different template
python src/arc_agi/prompt.py -p 08ed6ac7 -t baseline_wquotedgridcsv_excel.j2

# Preview reflexion template
python src/arc_agi/prompt.py -p 9565186b -t reflexion_enhanced.j2
```

---

## run_code.py - Code Execution and Validation

Execute Python code solutions on ARC-AGI problems and validate results.

**Location:** `src/arc_agi/run_code.py`

### Options

```bash
-p, --problem_name    # Problem ID (default: 9565186b)
-c, --code_filename   # Path to Python file with transform function
```

### Examples

```bash
# Run solution on specific problem
python src/arc_agi/run_code.py -p 0d3d703e -c example_solutions/ex_soln_0d3d703e.py

# Validate code against different problem
python src/arc_agi/run_code.py -p 08ed6ac7 -c /tmp/my_solution.py

# Test your own solution
python src/arc_agi/run_code.py -p 9565186b -c my_solutions/attempt1.py
```

### Output

The script returns a `RunResult` tuple containing:
- `code_did_execute`: Whether the code ran without syntax errors
- `code_ran_on_all_inputs`: Code executed on all training examples
- `transform_ran_and_matched_for_all_inputs`: All outputs matched expected
- `transform_ran_and_matched_at_least_once`: At least one output matched
- `transform_ran_and_matched_score`: Number of correct transformations

---

## method1_text_prompt.py - Single-Pass LLM Method

Single-pass LLM prompt method for solving ARC-AGI problems.

**Location:** `src/arc_agi/methods/method1_text_prompt.py`

### Options

```bash
-p, --problem_name    # Problem ID (default: 9565186b)
-t, --template_name   # Template in ./prompts/ (default: baseline_justjson.j2)
-i, --iterations      # Number of iterations to run (default: 1)
-m, --model_name      # OpenRouter model name (default: openrouter/deepseek/deepseek-chat-v3-0324)
```

### Examples

```bash
# Run 5 iterations on specific problem
python src/arc_agi/methods/method1_text_prompt.py -p 0d3d703e -i 5

# Use different template and model
python src/arc_agi/methods/method1_text_prompt.py -p 9565186b -t reflexion_enhanced.j2 -m openrouter/deepseek/deepseek-chat-v3-0324 -i 10

# Quick test with 1 iteration
python src/arc_agi/methods/method1_text_prompt.py -p 0d3d703e -i 1

# Test different template variations
python src/arc_agi/methods/method1_text_prompt.py -p 08ed6ac7 -t baseline_wquotedgridcsv_excel_enhanced.j2 -i 3
```

### Output Files

Creates an experiment folder with:
- `experiment.log` - Detailed execution log
- `experiments.db` - SQLite database with all attempts

**View results:**
```bash
# Check the log
tail -f experiments/exp_TIMESTAMP/experiment.log

# Query the database
sqlite3 experiments/exp_TIMESTAMP/experiments.db
sqlite> SELECT code FROM experiments WHERE all_train_transformed_correctly=true;
sqlite> SELECT final_explanation FROM experiments WHERE all_train_transformed_correctly=false;
```

### Automatic Retry Logic

The method automatically retries (up to 3 times) when encountering **structural errors**:
- LLM returns data instead of code
- Missing `transform` function
- Wrong function signature
- Syntax errors (if code is too short/malformed)

**Logic errors** (AssertionError, IndexError, etc.) do not trigger retries as they need debugging, not regeneration.

---

## method2_reflexion.py - Multi-Iteration Reflexion Method

Multi-iteration reflexion method with self-correction and explanation accumulation.

**Location:** `src/arc_agi/methods/method2_reflexion.py`

### Options

```bash
-p, --problem_name    # Problem ID (default: 9565186b)
-t, --template_name   # Template in ./prompts/ (default: baseline_justjson.j2)
-i, --iterations      # Number of iterations to run (default: 1)
-m, --model_name      # OpenRouter model name (default: openrouter/deepseek/deepseek-chat-v3-0324)
```

### Examples

```bash
# Run reflexion method with 20 iterations
python src/arc_agi/methods/method2_reflexion.py -p 9565186b -t reflexion_wquotedgridcsv_excel.j2 -i 20

# Test on harder problem
python src/arc_agi/methods/method2_reflexion.py -p 178fcbfb -i 15

# Use reflexion-specific template
python src/arc_agi/methods/method2_reflexion.py -p 0a938d79 -t reflexion_enhanced.j2 -i 25
```

### How Reflexion Works

1. **First iteration**: LLM generates initial explanation and code
2. **Subsequent iterations**: LLM sees:
   - Previous explanations (up to 5 most recent)
   - Previous code attempts
   - Error messages and execution outcomes
3. **Self-correction**: LLM refines understanding and generates improved code
4. **Accumulation**: Growing set of explanations helps build better mental model

### Output

Watch the logs to see the reflexion process:
```bash
tail -f experiments/exp_TIMESTAMP/experiment.log | grep -A 5 "EXPLANATION"
```

You'll see the growing set of explanations and increasingly complex code solutions.

---

## analysis.py - Statistical Analysis

Perform Fisher exact test to compare two experiments statistically.

**Location:** `scripts/analysis.py`

### Usage

```bash
python scripts/analysis.py <successes1> <total1> <successes2> <total2>
```

**Arguments:**
- `successes1` - Number of successes in experiment 1
- `total1` - Total number of trials in experiment 1
- `successes2` - Number of successes in experiment 2
- `total2` - Total number of trials in experiment 2

### Examples

```bash
# Compare two experiments: 32/50 vs 43/50
python scripts/analysis.py 32 50 43 50

# Compare different sized experiments: 65/300 vs 77/300
python scripts/analysis.py 65 300 77 300

# Test significance of improvement: 18/100 vs 25/100
python scripts/analysis.py 18 100 25 100
```

### Output

The script provides:
- **Fisher exact test statistic** (odds ratio)
- **P-value** (statistical significance)
- **Interpretation**: "Significant difference (p < 0.05)" or "No significant difference"

**Example output:**
```
Fisher exact test result: SignificanceResult(statistic=2.1875, pvalue=0.0368)
Odds ratio: 2.1875
P-value: 0.036834
Significant difference (p < 0.05)
```

### When to Use

- Compare two different prompts/templates on same problem set
- Evaluate if a method improvement is statistically significant
- Determine if differences in success rates are real or due to chance
- A/B testing of different LLM models

---

## create_retroactive_checkpoint.py - Checkpoint Recovery

Create checkpoint files for interrupted experiments to enable resumption.

**Location:** `scripts/create_retroactive_checkpoint.py`

### Usage

```bash
python scripts/create_retroactive_checkpoint.py <experiment_directory>
```

### Examples

```bash
# Create checkpoint from interrupted experiment
python scripts/create_retroactive_checkpoint.py batch_results/20250929_194041

# Then resume the experiment
python scripts/run_all_problems.py --resume

# Check checkpoint was created
ls batch_results/20250929_194041/checkpoint.json
```

### What It Does

1. **Parses experiment logs** for timing and configuration
2. **Extracts results** from SQLite database
3. **Creates valid `checkpoint.json`** file
4. **Generates resume instructions** to continue where you left off

### When to Use

- Experiment was interrupted (ctrl-C, system crash, timeout)
- Checkpoint system wasn't enabled (older experiments)
- Want to continue an experiment that partially completed
- Need to recover progress from interrupted batch run

### Output

Creates `checkpoint.json` in the experiment directory with:
- Completed template-problem combinations
- Timing information
- Configuration (templates, problems, iterations)
- Resume point for continuing the experiment

**After creating checkpoint:**
```bash
# Resume automatically
python scripts/run_all_problems.py --resume

# Or the script will auto-detect and prompt you
python scripts/run_all_problems.py
# Output: "Found checkpoint. Resume from last position? (y/n)"
```

---

## Tips and Best Practices

### General Tips

1. **Start with dry-run**: Use `--dry-run` on batch runner to preview execution
2. **Use verbose mode**: Add `-v` for detailed output when debugging
3. **Check logs**: Always check experiment.log for detailed execution traces
4. **Database queries**: Use SQLite to analyze results across iterations
5. **Template testing**: Test prompts with `prompt.py` before running experiments

### Performance Optimization

1. **Batch operations**: Use `run_all_problems.py` instead of manual loops
2. **Checkpoints**: Enable checkpoints for long-running experiments
3. **Iterations**: Start with fewer iterations (5-10) to test, then scale up
4. **Fail-fast**: Use `--fail-fast` when debugging to stop on first error

### Statistical Rigor

1. **Sample size**: Run enough iterations for statistical significance (30+ recommended)
2. **Compare fairly**: Use Fisher exact test to compare experiments
3. **Control variables**: Only change one variable at a time when comparing
4. **Document**: Keep notes on what you're testing and why

### Debugging

1. **Check syntax first**: Use `python -m py_compile script.py` to check syntax
2. **Test incrementally**: Run 1-2 iterations first before scaling
3. **Read error messages**: The retry logic distinguishes structural vs logic errors
4. **Use verbose mode**: See detailed execution flow with `-v`
