# Batch Experiment Runner

Comprehensive guide to `run_all_problems.py` - the batch experiment runner for systematic ARC-AGI testing.

## Overview

The `scripts/run_all_problems.py` script provides a powerful batch experiment runner for systematic ARC-AGI testing across multiple templates and problems. It automates running experiments across all combinations of templates, problems, and iterations with comprehensive tracking and reporting.

## Features

- **Multi-template and multi-problem testing**: Run experiments across all combinations of templates and problems
- **Flexible selection**: Choose specific templates and problems by name or index
- **Comprehensive timing tracking**: Global, template-level, and individual test timing with statistics
- **Real-time progress reporting**: Live feedback with timestamps and success indicators
- **Multi-format output**: Console tables, CSV files, HTML reports, and detailed logs
- **Robust error handling**: Graceful fallback when parallel execution fails
- **Success rate analysis**: Detailed performance breakdown with visual indicators
- **Checkpoint system**: Automatically save progress and resume interrupted experiments
- **Statistical summarization**: Aggregate results across multiple experiment runs

## Command-Line Options

### Core Options

```bash
-m METHOD, --method METHOD          # Method module to use (default: method1_text_prompt)
--model MODEL                       # Model name (default: openrouter/deepseek/deepseek-chat-v3-0324)
-i ITERATIONS, --iterations N       # Number of iterations per test (default: 1)
-t TEMPLATES, --templates LIST      # Comma-separated template names or indices
-p PROBLEMS, --problems LIST        # Comma-separated problem IDs or indices
-o OUTPUT_DIR, --output-dir DIR     # Output directory (default: batch_results)
```

### Execution Control

```bash
--dry-run                           # Show what would be run without executing
-v, --verbose                       # Verbose output with detailed progress
--fail-fast                         # Stop on first error (default: continue through all)
```

### Checkpoint and Resume

```bash
--no-checkpoint                     # Disable automatic checkpoint saving
--resume                            # Force resume from last checkpoint
--no-resume                         # Start fresh, ignore existing checkpoints
```

### Analysis Mode

```bash
--summarise-experiments             # Analyze existing results and generate summary statistics
```

## Usage Examples

### Basic Usage

```bash
# Run all templates and problems (comprehensive test)
python scripts/run_all_problems.py

# Run with 10 iterations per combination
python scripts/run_all_problems.py -i 10

# Use different model
python scripts/run_all_problems.py --model openrouter/deepseek/deepseek-chat-v3-0324
```

### Selective Testing

```bash
# Run specific templates and problems with multiple iterations
python scripts/run_all_problems.py -t "baseline_justjson_enhanced.j2,reflexion_enhanced.j2" -p "0d3d703e,08ed6ac7" -i 5

# Test single template-problem combination
python scripts/run_all_problems.py -t "baseline_justjson_enhanced.j2" -p "0d3d703e" -i 1

# Use template indices instead of names
python scripts/run_all_problems.py -t "1,2" -p "1,3,5" -i 3
```

### Method Selection

```bash
# Use method2 (reflexion) instead of method1
python scripts/run_all_problems.py --method method2_reflexion -i 10

# Test different method with specific templates
python scripts/run_all_problems.py --method method2_reflexion -t "reflexion_enhanced.j2" -i 20
```

### Execution Control

```bash
# Dry run to preview what would be executed
python scripts/run_all_problems.py --dry-run --verbose

# Verbose mode for detailed progress
python scripts/run_all_problems.py -v -i 5

# Fail-fast mode - stop on first error
python scripts/run_all_problems.py --fail-fast -i 10
```

### Checkpoint and Resume

```bash
# Resume interrupted experiment (will auto-detect checkpoint)
python scripts/run_all_problems.py

# Force resume without prompting
python scripts/run_all_problems.py --resume

# Disable checkpoints (not recommended for long runs)
python scripts/run_all_problems.py --no-checkpoint -i 20

# Start fresh, ignoring any existing checkpoints
python scripts/run_all_problems.py --no-resume
```

### Analysis and Reporting

```bash
# Analyze existing experiments and generate summary statistics
python scripts/run_all_problems.py --summarise-experiments --verbose

# Summarize experiments from custom output directory
python scripts/run_all_problems.py --summarise-experiments -o my_custom_results

# Just analyze, don't run new experiments
python scripts/run_all_problems.py --summarise-experiments --dry-run
```

## Checkpoint System

### How It Works

The batch runner automatically saves checkpoints during long experiment runs:

1. **Automatic saving**: Checkpoints saved every N experiments (configurable)
2. **Progress tracking**: Records completed template-problem combinations
3. **Timing preservation**: Maintains accurate timing across interruptions
4. **Configuration storage**: Saves all experiment parameters

### Resume Workflow

When you run the script and a checkpoint exists:

```
Found checkpoint from 2025-09-29 19:40:41
Completed: 12 of 20 experiments
Resume from last position? (y/n):
```

**Options:**
- `y` - Resume from checkpoint, skip completed experiments
- `n` - Start fresh, ignore checkpoint

**Auto-resume:**
```bash
python scripts/run_all_problems.py --resume  # No prompt, just resume
```

### Retroactive Checkpoints

If an experiment was interrupted before the checkpoint feature existed:

```bash
# Create checkpoint from interrupted experiment
python scripts/create_retroactive_checkpoint.py batch_results/20250929_194041

# Then resume
python scripts/run_all_problems.py --resume
```

See [CLI_REFERENCE.md](CLI_REFERENCE.md#create_retroactive_checkpointpy---checkpoint-recovery) for details.

## Output Structure

Results are saved in timestamped directories under `batch_results/`:

```
batch_results/
└── 20251001_143022/              # Timestamped experiment run
    ├── batch_results_20251001_143022.csv      # Structured data
    ├── batch_results_20251001_143022.html     # Visual dashboard
    ├── batch_summary_20251001_143022.log      # Execution log
    ├── checkpoint.json                        # Resume point
    └── experiment_summary_latest.log          # Cross-run summary
```

### File Descriptions

**CSV File (`batch_results_*.csv`)**
- Template name
- Problem ID
- Success rates (all-correct, at-least-one)
- Timing data
- LLM usage statistics
- Structured format for analysis in pandas/Excel

**HTML Report (`batch_results_*.html`)**
- Visual dashboard with color-coded results
- Performance indicators (🎯 ✅ ⚠️ ❌)
- Sortable tables
- Quick visual assessment of results

**Summary Log (`batch_summary_*.log`)**
- Detailed execution trace
- Timing breakdown by template
- Success/failure counts
- Configuration summary
- Human-readable format

**Checkpoint File (`checkpoint.json`)**
- Completed experiments list
- Timing information
- Configuration (templates, problems, iterations)
- Resume state

**Cross-Run Summary (`experiment_summary_latest.log`)**
- Aggregated statistics across multiple runs
- Template performance trends
- Problem difficulty analysis
- Best template recommendations

## Performance Indicators

Results use color-coded indicators for quick assessment:

- 🎯 **Excellent (≥80%)**: Grade A performance - template works very well
- ✅ **Good (50-79%)**: Grade B performance - template is promising
- ⚠️ **Partial**: Some success but inconsistent - needs refinement
- ❌ **Poor (<50%)**: Grade F - template needs significant improvement

## Analysis Mode

The `--summarise-experiments` flag enables analysis-only mode:

```bash
python scripts/run_all_problems.py --summarise-experiments --verbose
```

**What it does:**
1. Scans all experiment directories in output folder
2. Aggregates results across runs
3. Generates performance rankings
4. Creates persistent summary statistics
5. Identifies best templates per problem
6. Produces recommendation lookup tables

**Output files:**
- `experiment_summary_latest.log` - Aggregated statistics
- `template_performance_trends.csv` - Template analysis over time
- `problem_difficulty_trends.csv` - Problem difficulty categorization
- `best_template_lookup.json` - Automated recommendations

## Tips and Best Practices

### Planning Experiments

1. **Start small**: Test with `--dry-run` to preview execution
2. **Use indices**: When testing subsets, indices are faster than names
3. **Estimate time**: 1 iteration ≈ 30-60 seconds per combination
4. **Enable checkpoints**: Always use checkpoints for runs >30 minutes

### During Execution

1. **Monitor progress**: Use `-v` for detailed real-time feedback
2. **Check logs**: Watch timing to estimate completion
3. **Interrupt safely**: Ctrl-C is safe, checkpoint will be saved
4. **Resume promptly**: Resume interrupted experiments soon to maintain timing accuracy

### After Experiments

1. **Check HTML report**: Quick visual assessment of results
2. **Analyze CSV**: Import into pandas/Excel for detailed analysis
3. **Compare runs**: Use Fisher exact test (see [CLI_REFERENCE.md](CLI_REFERENCE.md#analysispy---statistical-analysis))
4. **Archive results**: Keep successful experiment directories for reference

### Optimization

1. **Parallel processing**: Uses joblib for parallel LLM calls
2. **Fail-fast for debugging**: Use `--fail-fast` when testing new templates
3. **Batch similar tests**: Group related experiments in single run
4. **Custom output dirs**: Use `-o` to organize experiments by purpose

### Common Patterns

**Quick template test:**
```bash
python scripts/run_all_problems.py -t "new_template.j2" -p "0d3d703e" -i 3
```

**Full evaluation:**
```bash
python scripts/run_all_problems.py -i 10 --model openrouter/deepseek/deepseek-chat-v3-0324
```

**Debugging:**
```bash
python scripts/run_all_problems.py --dry-run --verbose -t "1" -p "1"
```

**Production run:**
```bash
python scripts/run_all_problems.py -i 50 -v &
# Monitor with: tail -f batch_results/latest/batch_summary_*.log
```

## Troubleshooting

### Checkpoint Issues

**Problem**: "Checkpoint mismatch" error
**Solution**: Configuration changed since checkpoint. Use `--no-resume` to start fresh.

**Problem**: Can't find checkpoint
**Solution**: Checkpoint is in the output directory. Use `-o` to specify correct directory.

### Performance Issues

**Problem**: Very slow execution
**Solution**: Check network/API connectivity. Consider reducing iterations or problems.

**Problem**: Out of memory
**Solution**: Reduce parallel processing or run fewer combinations at once.

### Result Issues

**Problem**: All experiments failing
**Solution**: Check API keys, test single experiment first, verify template syntax.

**Problem**: Inconsistent results
**Solution**: Increase iterations for statistical significance (30+ recommended).

## See Also

- [CLI_REFERENCE.md](CLI_REFERENCE.md) - All command-line tools reference
- [WALKTHROUGH.md](WALKTHROUGH.md) - Step-by-step tutorial
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Recent feature additions
