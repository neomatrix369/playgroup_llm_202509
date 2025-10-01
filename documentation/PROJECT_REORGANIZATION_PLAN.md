# Project Reorganization Plan

## Current Structure Issues

The project has grown organically with many files in the root directory:
- Main modules mixed with test files in root
- Some organized subdirectories (analysis/, core/, domain/, output/)
- Test files not in dedicated tests/ directory
- No clear separation between src code and configuration

## Target Structure (Python Best Practices)

```
playgroup_llm_202509/
├── src/
│   └── arc_agi/                    # Main package
│       ├── __init__.py
│       ├── analysis/               # Analysis modules (KEEP existing)
│       │   ├── __init__.py
│       │   ├── difficulty_classifier.py
│       │   ├── experiment_aggregator.py
│       │   ├── experiment_summarizer.py
│       │   ├── performance_grader.py
│       │   └── statistics_aggregator.py
│       ├── core/                   # Core functionality (KEEP existing)
│       │   ├── __init__.py
│       │   ├── checkpoint_manager.py
│       │   ├── experiment_argument_parser.py
│       │   ├── experiment_coordinator.py
│       │   ├── experiment_executor.py
│       │   └── timing_tracker.py
│       ├── domain/                 # Domain models (KEEP existing)
│       │   ├── __init__.py
│       │   ├── experiment_config.py
│       │   └── experiment_state.py
│       ├── output/                 # Output generation (KEEP existing)
│       │   ├── __init__.py
│       │   ├── console_display.py
│       │   ├── formatters.py
│       │   └── report_writer.py
│       ├── methods/                # Experiment methods (NEW)
│       │   ├── __init__.py
│       │   ├── method1_text_prompt.py     # FROM ROOT
│       │   ├── method2_reflexion.py       # FROM ROOT
│       │   └── variants/                  # FROM variant_method1_text_prompt_in2part/
│       │       └── method1_in2part2.py
│       ├── validation/             # Code validation (NEW)
│       │   ├── __init__.py
│       │   ├── code_sanitizer.py          # Extract from run_code.py
│       │   ├── code_validator.py          # Extract from run_code.py
│       │   └── regeneration_checker.py    # Extract from run_code.py
│       ├── execution/              # Code execution (NEW)
│       │   ├── __init__.py
│       │   └── transform_executor.py      # Extract from run_code.py
│       ├── db.py                   # FROM ROOT - Database operations
│       ├── litellm_helper.py       # FROM ROOT - LLM interface
│       ├── prompt.py               # FROM ROOT - Prompt management
│       ├── representations.py      # FROM ROOT - Data representations
│       ├── run_code.py             # FROM ROOT - Main execution (refactor later)
│       └── utils.py                # FROM ROOT - Utilities
├── tests/                          # All test files (NEW DIRECTORY)
│   ├── __init__.py
│   ├── test_prompt.py              # FROM ROOT
│   ├── test_refactoring.py         # FROM ROOT
│   ├── test_representations.py     # FROM ROOT
│   ├── test_retry_logic.py         # FROM ROOT
│   ├── test_run_code.py            # FROM ROOT
│   ├── test_statistics_aggregator.py  # FROM ROOT
│   └── test_utils.py               # FROM ROOT
├── scripts/                        # Executable scripts (NEW)
│   ├── __init__.py
│   ├── run_all_problems.py         # FROM ROOT - Main batch runner
│   ├── create_retroactive_checkpoint.py  # FROM ROOT
│   └── analysis.py                 # FROM ROOT - Analysis script
├── data/                           # Data files (RENAME)
│   └── arc_data/                   # FROM ROOT arc_data/
├── results/                        # Results output (RENAME)
│   └── batch_results/              # FROM ROOT batch_results/
├── experiments/                    # Experiment databases (KEEP)
│   └── *.db
├── prompts/                        # Prompt templates (KEEP)
│   └── *.j2
├── example_solutions/              # Example solutions (KEEP)
│   └── *.py
├── documentation/                  # Documentation (KEEP)
│   ├── PROJECT_REORGANIZATION_PLAN.md
│   └── REFACTORING_PLAN.md
├── requirements.txt                # Dependencies (KEEP)
├── pytest.ini                      # Test configuration (CREATE)
├── setup.py or pyproject.toml      # Package setup (CREATE)
└── README.md                       # Main documentation (KEEP)
```

## Migration Strategy

### Phase 1: Create Directory Structure
1. Create `src/arc_agi/` directory
2. Create `tests/` directory
3. Create `scripts/` directory
4. Create `src/arc_agi/methods/` directory
5. Create `src/arc_agi/validation/` directory
6. Create `src/arc_agi/execution/` directory

### Phase 2: Move Existing Organized Directories
```bash
git mv analysis/ src/arc_agi/
git mv core/ src/arc_agi/
git mv domain/ src/arc_agi/
git mv output/ src/arc_agi/
```

### Phase 3: Move Test Files
```bash
git mv test_*.py tests/
```

### Phase 4: Move Main Modules to Package
```bash
git mv db.py src/arc_agi/
git mv litellm_helper.py src/arc_agi/
git mv prompt.py src/arc_agi/
git mv representations.py src/arc_agi/
git mv run_code.py src/arc_agi/
git mv utils.py src/arc_agi/
```

### Phase 5: Move Method Modules
```bash
git mv method1_text_prompt.py src/arc_agi/methods/
git mv method2_reflexion.py src/arc_agi/methods/
git mv variant_method1_text_prompt_in2part/ src/arc_agi/methods/variants/
```

### Phase 6: Move Scripts
```bash
git mv run_all_problems.py scripts/
git mv analysis.py scripts/
git mv create_retroactive_checkpoint.py scripts/
```

### Phase 7: Rename Data Directories
```bash
git mv arc_data/ data/arc_data/
git mv batch_results/ results/batch_results/
```

### Phase 8: Create __init__.py Files
Create empty `__init__.py` files in all package directories.

### Phase 9: Update All Imports
Update imports across all files:
- `from run_code import` → `from arc_agi.run_code import`
- `from utils import` → `from arc_agi.utils import`
- `import method1_text_prompt` → `from arc_agi.methods import method1_text_prompt`
- etc.

### Phase 10: Create Package Configuration
Create `pyproject.toml` or `setup.py` for package installation.

Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

### Phase 11: Update README
Update all path references in README.md.

## Test Coverage Assessment (Current State)

### ✅ Well-Tested Modules:
1. **run_code.py** (11 tests)
   - Code execution, validation, sanitization
   - Transform execution on various inputs
   - Error handling and edge cases

2. **utils.py** (11 tests)
   - Code block extraction (7 tests)
   - Explanation extraction (7 tests)
   - Multi-strategy fallback

3. **Retry Logic** (6 tests)
   - Regeneration decision logic
   - Retry workflow for structural errors
   - Success case (no retry)

4. **statistics_aggregator.py** (7 tests)
   - Template statistics
   - Problem statistics
   - Experiment statistics
   - Best template recommendations

5. **representations.py** (5 tests)
   - Grid representations
   - CSV formats
   - Excel-style output

6. **prompt.py** (1 test)
   - Basic template rendering

7. **Refactoring components** (5 tests)
   - Performance grading
   - Difficulty classification
   - Model immutability

**Total: 50 tests covering core functionality**

### ❌ Missing Test Coverage:

#### High-Level Workflows (End-to-End):
1. **run_all_problems.py** - NO TESTS
   - Batch experiment orchestration
   - Template/problem iteration
   - Result aggregation
   - Checkpoint/resume functionality

2. **method1_text_prompt.py** - NO TESTS
   - Full experiment iteration workflow
   - LLM interaction
   - Result recording

3. **method2_reflexion.py** - NO TESTS
   - Reflexion iteration loop
   - Explanation refinement
   - Multi-iteration code improvement

#### Module-Level Coverage Gaps:
4. **db.py** - NO TESTS
   - Database operations
   - Experiment recording

5. **litellm_helper.py** - NO TESTS
   - LLM API calls
   - Error handling
   - Response parsing

6. **Checkpoint system** - NO TESTS
   - Checkpoint creation
   - Resume from checkpoint
   - State restoration

7. **ExperimentArgumentParser** - NO TESTS
   - Argument parsing
   - Validation

8. **ExperimentExecutor** - NO TESTS
   - Experiment execution orchestration

9. **ExperimentCoordinator** - NO TESTS
   - Coordination logic

10. **Output modules** - NO TESTS
    - HTML/CSV report generation
    - Console display formatting

### Recommended Test Additions:

#### Priority 1 (Critical Workflows):
- [ ] End-to-end test for `run_all_problems.py` workflow
- [ ] Integration test for method1 with mocked LLM
- [ ] Integration test for method2 reflexion loop
- [ ] Checkpoint save/resume integration test

#### Priority 2 (Module Coverage):
- [ ] Unit tests for `db.py` operations
- [ ] Unit tests for `litellm_helper.py` with mocked API
- [ ] Unit tests for ExperimentArgumentParser
- [ ] Unit tests for ExperimentExecutor

#### Priority 3 (Output Validation):
- [ ] Tests for report generation (HTML/CSV)
- [ ] Tests for console formatting
- [ ] Tests for output file creation

## Benefits of Reorganization

1. **Clarity**: Clear separation of concerns (src, tests, scripts, data)
2. **Discoverability**: Easy to find related functionality
3. **Testing**: Dedicated tests/ directory with proper test discovery
4. **Packaging**: Can install as `pip install -e .` for development
5. **Standards**: Follows Python community best practices
6. **Imports**: Clean import structure (`from arc_agi.core import ...`)
7. **Scalability**: Easy to add new modules in appropriate locations

## Risks and Mitigation

### Risks:
1. Breaking existing workflows/scripts
2. Import errors across the codebase
3. Path references in configuration files

### Mitigation:
1. Use `git mv` to preserve history
2. Comprehensive import update pass
3. Run full test suite after each phase
4. Update documentation with new paths
5. Keep backward compatibility aliases if needed

## Execution Timeline

- Phase 1-3: 15 minutes (directory creation and moves)
- Phase 4-7: 20 minutes (file moves with git)
- Phase 8: 10 minutes (create __init__.py files)
- Phase 9: 30-60 minutes (update imports - most time consuming)
- Phase 10: 15 minutes (configuration files)
- Phase 11: 10 minutes (documentation updates)

**Total estimated time: 2-3 hours**

## Rollback Plan

If issues arise:
```bash
git reset --hard HEAD  # Discard all changes
git clean -fd          # Remove untracked files
```

All changes should be in a single commit or feature branch for easy rollback.