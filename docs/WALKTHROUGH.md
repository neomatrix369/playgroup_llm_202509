# Playgroup Walkthrough

Step-by-step walkthrough for playgroup attendees learning ARC-AGI with LLMs.

## Prerequisites

1. **Environment setup**: Python 3.12, virtual environment activated
2. **Package installed**: `pip install -e .` completed
3. **API keys configured**: OpenRouter API key in environment or `.env` file

## The Problems We'll Look At

Visit these ARC-AGI problems to understand what we're solving:

| Problem | Description | Difficulty | Grid Size |
|---------|-------------|------------|-----------|
| [0d3d703e](https://arcprize.org/play?task=0d3d703e) | Fixed colour mapping | Medium | 3x3 min |
| [08ed6ac7](https://arcprize.org/play?task=08ed6ac7) | Coloured in order of height | Hard | 9x9 min |
| [9565186b](https://arcprize.org/play?task=9565186b) | Most frequent colour wins | Medium | 3x3 min |
| [178fcbfb](https://arcprize.org/play?task=178fcbfb) | Dots form coloured lines | Very Hard | 9x9 min |
| [0a938d79](https://arcprize.org/play?task=0a938d79) | Dots form repeated lines | Very Hard | 9x22 min |
| [1a07d186](https://arcprize.org/play?task=1a07d186) | Dots attach to coloured line | Extreme | 14x15 min |

## Part 1: Understanding Prompts

### Step 1: Generate a Prompt

First, let's see what prompts look like for different template styles:

```bash
# Visit the problem first to understand it
# https://arcprize.org/play?task=0d3d703e

# Generate prompt with JSON representation
python src/arc_agi/prompt.py -t baseline_justjson.j2 -p 0d3d703e

# Generate prompt with CSV grid representation
python src/arc_agi/prompt.py -t baseline_wquotedgridcsv_excel.j2 -p 0d3d703e
```

**Observe:**
- How are grids represented differently?
- Which format is easier to understand?
- Does one provide more structural hints?

### Step 2: Compare Representations

Try generating prompts for the same problem with different templates:

```bash
# Plain grid representation
python src/arc_agi/prompt.py -t baseline_wplaingrid_enhanced.j2 -p 0d3d703e

# Enhanced JSON with reflexion
python src/arc_agi/prompt.py -t reflexion_enhanced.j2 -p 0d3d703e
```

**Discussion points:**
- What information is included/excluded in each template?
- How much do we hint vs. let the LLM discover?
- Trade-offs between specificity and generality?

## Part 2: Running Code Solutions

### Step 3: Test Known Solutions

Let's test pre-written solutions to understand the execution framework:

```bash
# Run a good solution on the right problem
python src/arc_agi/run_code.py -p 0d3d703e -c example_solutions/ex_soln_0d3d703e.py

# Run the WRONG solution on a different problem (should fail)
python src/arc_agi/run_code.py -p 0d3d703e -c example_solutions/ex_soln_08ed6ac7.py
```

**Observe the output:**
- `RunResult` tuple shows execution status
- `ExecutionOutcome` objects show transformation for each example
- Success/failure tracked for each training example

### Understanding RunResult

The `execute_transform()` function returns:

```python
RunResult(
    code_did_execute=True,              # Code ran without syntax errors
    code_ran_on_all_inputs=True,        # Executed on all training examples
    transform_ran_and_matched_for_all_inputs=True,  # All matched expected output
    transform_ran_and_matched_at_least_once=True,   # At least one matched
    transform_ran_and_matched_score=4   # Number of correct transformations
)
```

## Part 3: Single-Pass LLM Method (method1)

### Step 4: Run Basic Experiment

Now let's ask an LLM to solve a problem:

```bash
# Run 5 iterations on an easy problem
python src/arc_agi/methods/method1_text_prompt.py -p 0d3d703e -i 5

# Expected: 2-3 minutes, ~20% success rate
```

**While it runs:**
1. Watch the console output for progress
2. Note the timing per iteration
3. See success/failure indicators

### Step 5: Examine Results

```bash
# The script prints paths at start, e.g.:
# tail -n +0 -f experiments/exp_20251001T143022/experiment.log
# sqlite3 experiments/exp_20251001T143022/experiments.db

# Check the log file for details
tail -f experiments/exp_TIMESTAMP/experiment.log

# Query the database for successful solutions
sqlite3 experiments/exp_TIMESTAMP/experiments.db
sqlite> .schema
sqlite> SELECT code FROM experiments WHERE all_train_transformed_correctly=true;
sqlite> SELECT final_explanation FROM experiments WHERE all_train_transformed_correctly=false;
sqlite> .quit
```

**Key observations:**
- How many iterations succeeded?
- What do successful solutions look like?
- What went wrong in failed attempts?

### Understanding Automatic Retry Logic

The system automatically retries structural errors:

```
Attempt 1: LLM returns "1 2 3\n4 5 6" (data, not code)
  ❌ Structural error detected
  🔄 Retrying (attempt 2/3)...

Attempt 2: LLM returns same bad output
  ❌ Structural error detected
  🔄 Retrying (attempt 3/3)...

Attempt 3: Still failing
  ❌ Failed after 3 attempts, giving up
  ➡️ Moving to next iteration
```

**Error categories:**
- **STRUCTURAL** (auto-retry): Missing transform, wrong signature, data instead of code
- **LOGIC** (no retry): AssertionError, IndexError, ZeroDivisionError - needs debugging
- **SYNTAX** (conditional): Auto-retry if severe, otherwise prefer debugging

### Step 6: Try Different Templates

```bash
# Test with enhanced JSON template
python src/arc_agi/methods/method1_text_prompt.py -p 0d3d703e -t baseline_justjson_enhanced.j2 -i 5

# Test with CSV grid template
python src/arc_agi/methods/method1_text_prompt.py -p 0d3d703e -t baseline_wquotedgridcsv_excel_enhanced.j2 -i 5
```

**Compare results:**
- Which template performs better?
- Is the difference statistically significant?
- Use `scripts/analysis.py` to compare (see Part 5)

## Part 4: Multi-Iteration Reflexion Method (method2)

### Step 7: Run Reflexion Method

Try the reflexion method which learns from previous attempts:

```bash
# Run 20 iterations on a harder problem
python src/arc_agi/methods/method2_reflexion.py -p 9565186b -t reflexion_wquotedgridcsv_excel.j2 -i 20

# This will take longer but may improve over iterations
```

### Step 8: Watch the Learning Process

```bash
# In another terminal, watch the logs
tail -f experiments/exp_TIMESTAMP/experiment.log | grep -A 5 "EXPLANATION"
```

**Observe:**
- Growing set of explanations (up to 5 kept)
- Increasingly complex code solutions
- Self-correction as LLM sees what didn't work
- Does performance improve over iterations?

### How Reflexion Works

1. **Iteration 1**: LLM generates initial explanation and code
2. **Iteration 2**: LLM sees:
   - Previous explanation
   - Previous code
   - What failed/succeeded
3. **Iteration N**: LLM has:
   - Up to 5 previous explanations
   - History of what approaches didn't work
   - Ability to refine understanding

**Discussion:**
- Is reflexion better than fresh attempts each time?
- Does the LLM actually learn from mistakes?
- What's the optimal number of iterations?

## Part 5: Statistical Analysis

### Step 9: Compare Two Experiments

Use Fisher exact test to determine if differences are statistically significant:

```bash
# Compare template A (32/50 success) vs template B (43/50 success)
python scripts/analysis.py 32 50 43 50

# Output shows:
# - Odds ratio
# - P-value
# - Interpretation (significant if p < 0.05)
```

**Interpretation:**
- P-value < 0.05: Difference is statistically significant
- P-value ≥ 0.05: Difference could be due to chance
- Odds ratio: How much more likely one is to succeed

## Part 6: Prompt Engineering

### Step 10: Create Your Own Template

```bash
# Copy a good baseline template
cp prompts/baseline_wquotedgridcsv_excel.j2 prompts/my_enhanced_template.j2

# Edit it to add hints, structure, or different instructions
# $EDITOR prompts/my_enhanced_template.j2

# Test your template
python src/arc_agi/methods/method1_text_prompt.py -p 9565186b -t my_enhanced_template.j2 -i 3
```

**Ideas to try:**
- Add examples of Python patterns
- Include hints about common ARC-AGI patterns
- Structure the prompt differently
- Add constraints or requirements
- Include grid analysis steps

### Step 11: Compare Against Baseline

```bash
# Run your template 10 times
python src/arc_agi/methods/method1_text_prompt.py -p 9565186b -t my_enhanced_template.j2 -i 10

# Run baseline 10 times
python src/arc_agi/methods/method1_text_prompt.py -p 9565186b -t baseline_wquotedgridcsv_excel.j2 -i 10

# Compare statistically
python scripts/analysis.py <your_successes> 10 <baseline_successes> 10
```

## Part 7: Batch Testing

### Step 12: Run Systematic Tests

For comprehensive evaluation, use the batch runner:

```bash
# Test your template across multiple problems
python scripts/run_all_problems.py -t "my_enhanced_template.j2" -p "0d3d703e,9565186b,08ed6ac7" -i 10

# Compare multiple templates
python scripts/run_all_problems.py -t "baseline_justjson_enhanced.j2,my_enhanced_template.j2" -p "0d3d703e,9565186b" -i 5 -v
```

See [BATCH_RUNNER.md](BATCH_RUNNER.md) for complete documentation.

## Discussion Questions

### Representation Optimization
- Is the baseline representation suboptimal?
- How could it be improved?
- Look in `src/arc_agi/representations.py` - what could we add?
- Should we include grid size information?
- Would scipy's connected components help or mislead?

### Prompt Structure
- Is a 1-pass prompt optimal?
- Should it be split into discrete chunks?
- Break into: analysis phase → pattern identification → code generation?
- How much structure helps vs. hinders?

### Learning vs. Fresh Attempts
- Is reflexion better than independent attempts?
- Does accumulating explanations actually help?
- At what point does the context become noise?
- Best iteration count for reflexion?

### Model Selection
- Does model choice matter more than prompt?
- DeepSeek vs. other models on ARC-AGI?
- Speed vs. quality trade-offs?

### Evaluation Rigor
- How many iterations needed for statistical significance?
- When is a template "good enough"?
- Cross-problem generalization vs. problem-specific tuning?

## Next Steps

1. **Experiment**: Try different templates and problems
2. **Analyze**: Use statistical tests to validate improvements
3. **Iterate**: Refine based on results
4. **Share**: Compare findings with the playgroup

## Resources

- [CLI Reference](CLI_REFERENCE.md) - Complete tool documentation
- [Batch Runner Guide](BATCH_RUNNER.md) - Systematic testing
- [ARC Prize Website](https://arcprize.org/) - More problems and information
- [Project Repository](https://github.com/ianozsvald/playgroup_llm_202509) - Code and updates
