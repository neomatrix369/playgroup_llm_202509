# playgroup_llm_202509

Created for playgroup attendees on 2025-09 in London.

License - MIT.

* https://the-playgroup.slack.com/
* https://github.com/ianozsvald/playgroup_llm_202509

# Setup

```
# I'm assuming you havea  Python 3.12 environment setup already
# you'll see further below in 'Ian's notes' I use conda to make a plain 3.12 env, then I make this venv (on my linux machine)
python -m venv .venv
. .venv/bin/activate # activate local env
pip install -r requirements.txt

# now you _should_ be able to run pytest and prompt.py
pytest
python prompt.py --help # just check it runs
python run_code.py --help # just check it runs
```

# Walkthrough (during playgroup)

```
# visit https://arcprize.org/play?task=0d3d703e
python prompt.py -t baseline_justjson.j2 -p 0d3d703e # render this problem as a prompt
python prompt.py -t baseline_wquotedgridcsv_excel.j2 -p 0d3d703e # render this problem as a prompt

```

```
python run_code.py --help
python run_code.py -p 0d3d703e -c example_solutions/ex_soln_0d3d703e.py # run good solution on the right problem
python run_code.py -p 0d3d703e -c example_solutions/ex_soln_08ed6ac7.py # run the wrong solution on a different problem
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

```
# run the basic method with the default prompt for 5 iterations
python method1_text_prompt.py --help # see the arg description
python method1_text_prompt.py -p 0d3d703e -i 5  # maybe 2-3 minutes and 20% correctness?
# this is equivalent to the fully formed version which selects the prompt and model to run
# python method1_text_prompt.py -p 0d3d703e -t baseline_justjson.j2 -m openrouter/deepseek/deepseek-chat-v3-0324 -i 5
# you could try looking at experiment.log logfile (detailed at the top of stdout when you run method1)
# you could try opening sqlite3 (if installed), detailed under the logfile line
sqlite3> .schema
sqlite3> select code from experiments where all_train_transformed_correctly=true;
sqlite3> select final_explanation from experiments where all_train_transformed_correctly=false;
Note if it got an explanation right, but wrote bad code (e.g. with a SyntaxError), then it won't transform correctly

# In method1's run_experiment function we receive an object after trying a proposed solution
# rr_train is a tuple of (RunResult, ExecutionOutcome, exception_message)
# rr_train[0].code_ran_on_all_inputs will be True if all train examples ran regardless of output quality
# rr_train[0].transform_ran_and_matched_for_all_inputs will be True if all train inputs were transformed correctly
# rr_train[1] gives an ExecutionOutcome object, for each initial/final pair it shows what the transform function generated
```

## Batch Experiment Runner

The `run_all_problems.py` script provides a comprehensive batch experiment runner for systematic ARC-AGI testing across multiple templates and problems.

### Features

- **Multi-template and multi-problem testing**: Run experiments across all combinations of templates and problems
- **Flexible selection**: Choose specific templates and problems by name or index
- **Comprehensive timing tracking**: Global, template-level, and individual test timing
- **Real-time progress reporting**: Live feedback with timestamps and success indicators
- **Multi-format output**: Console tables, CSV files, HTML reports, and detailed logs
- **Robust error handling**: Graceful fallback when parallel execution fails
- **Success rate analysis**: Detailed performance breakdown with visual indicators

### Usage Examples

```bash
# Run all templates and problems (comprehensive test)
python run_all_problems.py

# Run specific templates and problems with multiple iterations
python run_all_problems.py -t "baseline_justjson_enhanced.j2,reflexion_enhanced.j2" -p "0d3d703e,08ed6ac7" -i 5

# Test with different method module and model
python run_all_problems.py --method method2_reflexion --model openrouter/deepseek/deepseek-chat-v3-0324

# Dry run to preview what would be executed
python run_all_problems.py --dry-run --verbose

# Quick test of a single template-problem combination
python run_all_problems.py -t "baseline_justjson_enhanced.j2" -p "0d3d703e" -i 1
```

### Output Structure

Results are saved in timestamped directories under `batch_results/` containing:
- **CSV file**: Structured data for analysis (`batch_results_TIMESTAMP.csv`)
- **HTML report**: Visual dashboard with color-coded results (`batch_results_TIMESTAMP.html`)
- **Summary log**: Detailed execution log with timing breakdown (`batch_summary_TIMESTAMP.log`)

### Performance Indicators

- 🎯 **Excellent (≥80%)**: Grade A performance
- ✅ **Good (50-79%)**: Grade B performance
- ⚠️ **Partial**: Some success but inconsistent
- ❌ **Poor (<50%)**: Grade F, needs improvement

### Recent Improvements

The batch runner has been enhanced with robust error handling:
- **Fixed file path handling**: Resolves "'str' object has no attribute 'name'" errors
- **Improved execution resilience**: Fallback to serial execution when parallel processing fails
- **Better dependency management**: Added missing scikit-image dependency
- **Enhanced logging**: Proper logger initialization across modules 

Now compare this to the EXPT (experiments) results - Ian on screen - better prompt sort of gets us further.

How much further could we go?

### run method2 on a harder problem, observe the logs

```
%run method2_reflexion.py -t reflexion_wquotedgridcsv_excel.j2 -p 9565186b -i 20
now open the logs and follow them - watch the growing set of (5) explanations and more-complex code solutions
is this a good direction?
```

### hinting - use method1 again on a copy of a prompt

```
in prompts/ copy e.g. baseline_wquotedgridcsv_excel.j2 (a good one)
we'll discuss what we could add...
python method1_text_prompt.py -p 9565186b -t your_hinted_prompt.j2 -i 3
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
