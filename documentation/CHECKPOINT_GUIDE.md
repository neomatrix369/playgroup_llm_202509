# Checkpoint & Resume Guide

## What It Does

The checkpoint system automatically saves your progress during batch experiments. If your experiment is interrupted (Ctrl+C, crash, etc.), you can resume exactly where you left off.

---

## Basic Usage

### 1. Normal Run (Auto-Checkpoint Enabled)
```bash
python run_all_problems.py
```
- Automatically saves progress after each experiment
- On restart, prompts you to resume if a checkpoint exists

### 2. Resume After Interruption
If you interrupted an experiment, just run it again:
```bash
python run_all_problems.py
```

You'll see:
```
🔄 CHECKPOINT FOUND - PREVIOUS RUN WAS INTERRUPTED
================================================================================

📊 Checkpoint Details:
   Output directory: batch_results/20250929_194041
   Progress: 5/20 experiments (25.0%)
   Time elapsed: 1h 15m
   Last completed: ('baseline_justjson_enhanced.j2', '0d3d703e')

▶️  Would resume from:
   Test 6/20: baseline_justjson_enhanced.j2 + 08ed6ac7

================================================================================
Resume from checkpoint? [Y/n]:
```

Press **Enter** or type **Y** to continue from where you stopped.

---

## Command-Line Flags

### Force Resume (No Prompt)
```bash
python run_all_problems.py --resume
```
Automatically resumes without asking.

### Start Fresh (Ignore Checkpoint)
```bash
python run_all_problems.py --no-resume
```
Starts a new run, ignoring any existing checkpoint.

### Disable Checkpointing
```bash
python run_all_problems.py --no-checkpoint
```
Turns off automatic checkpoint saving (not recommended).

---

## How It Works

1. **Auto-Save**: After each experiment completes, progress is saved to:
   ```
   batch_results/YYYYMMDD_HHMMSS/checkpoint.json
   ```

2. **On Startup**: The system checks for a checkpoint file in the output directory.

3. **Resume**: If found, you can choose to resume. The system will:
   - Skip already-completed experiments
   - Continue from the next uncompleted experiment
   - Restore all results and timing data

4. **On Completion**: Checkpoint file is archived (not deleted) for reference.

---

## Examples

### Example 1: Interrupted Run
```bash
# Start experiment
python run_all_problems.py -t "baseline_justjson_enhanced.j2" -p "0d3d703e,08ed6ac7,178fcbfb"

# ... runs for 1 hour, completes 1/3 tests ...
# Press Ctrl+C to interrupt

# Later, resume:
python run_all_problems.py
# Prompts to resume → Press Y
# Skips first test, continues from test 2/3
```

### Example 2: Multiple Interruptions
```bash
python run_all_problems.py
# Completes 5/20 tests, interrupted

python run_all_problems.py --resume
# Resumes, completes 5 more (now 10/20), interrupted again

python run_all_problems.py --resume
# Resumes from test 11/20, completes all
```

---

## Checkpoint File Location

Checkpoints are saved in your output directory:
```
batch_results/
  └── 20250929_194041/           # Your experiment run
      ├── checkpoint.json         # ← Checkpoint file
      ├── experiment_run_*.log
      └── experiments.db
```

---

## FAQ

**Q: What happens if I change templates or problems between runs?**  
A: The checkpoint is specific to the original configuration. If you change parameters, start fresh with `--no-resume`.

**Q: Can I manually edit the checkpoint file?**  
A: Not recommended. It's JSON, but changing it may cause errors.

**Q: Does checkpointing slow down experiments?**  
A: No. Checkpoint saves are very fast (milliseconds).

**Q: What if the checkpoint file is corrupted?**  
A: The system will warn you and start fresh automatically.

**Q: Can I see what's in the checkpoint?**  
A: Yes! It's a JSON file. Open it with any text editor:
```bash
cat batch_results/20250929_194041/checkpoint.json
```

---

## Troubleshooting

### Checkpoint Not Working?
1. Check you're not using `--no-checkpoint`
2. Verify the output directory exists
3. Check file permissions

### Can't Resume?
1. Use `--no-resume` to start fresh
2. Delete the checkpoint file manually if corrupted:
   ```bash
   rm batch_results/YYYYMMDD_HHMMSS/checkpoint.json
   ```

---

## Quick Reference

| Command | What It Does |
|---------|-------------|
| `python run_all_problems.py` | Normal run with auto-checkpoint |
| `python run_all_problems.py --resume` | Force resume without prompt |
| `python run_all_problems.py --no-resume` | Start fresh, ignore checkpoint |
| `python run_all_problems.py --no-checkpoint` | Disable checkpointing |

---

**That's it!** The checkpoint system works automatically. Just run your experiments normally, and if interrupted, run again to resume.
