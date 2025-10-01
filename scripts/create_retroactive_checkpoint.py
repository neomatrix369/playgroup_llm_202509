#!/usr/bin/env python3
"""
Create retroactive checkpoint from an interrupted experiment.

This script reconstructs a checkpoint.json file from existing experiment logs and database,
allowing you to resume experiments that were interrupted before the checkpoint feature existed.

Usage:
    python create_retroactive_checkpoint.py batch_results/20250929_194041

The script will:
1. Parse experiment logs for timing and configuration
2. Extract results from the SQLite database
3. Create a valid checkpoint.json file
4. Generate resume instructions
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def parse_log_file(log_path):
    """Parse experiment log to extract timing and configuration."""
    with open(log_path, "r") as f:
        lines = f.readlines()

    # Extract start time from first line
    # Format: 🕒 [2025-09-29 19:40:41] Logging to: ...
    start_line = lines[0]
    start_time_str = start_line.split("[")[1].split("]")[0]
    start_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")

    # Find completed experiments and checkpoint time
    completed = []
    checkpoint_time_str = start_time_str  # Default to start time

    for line in lines:
        if "✅ Problem completed:" in line:
            # Extract problem ID
            parts = line.split("✅ Problem completed:")[1].strip().split()
            problem = parts[0]
            # Get timestamp
            timestamp_str = line.split("[")[1].split("]")[0]
            checkpoint_time_str = timestamp_str
            completed.append(problem)
        elif "Starting template branch:" in line:
            # Extract template name
            # Format: 🌿 Starting template branch: baseline_justjson_enhanced.j2 (3 problems)
            template = line.split("Starting template branch:")[1].split("(")[0].strip()

    checkpoint_dt = datetime.strptime(checkpoint_time_str, "%Y-%m-%d %H:%M:%S")

    return {
        "start_time": start_dt.timestamp(),
        "checkpoint_time": checkpoint_dt.timestamp(),
        "template": template if "template" in locals() else None,
        "completed_problems": completed,
    }


def analyze_database(db_path):
    """Extract results from the experiment database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get success statistics
    cursor.execute("SELECT COUNT(*) FROM experiments WHERE all_train_transformed_correctly = 1;")
    all_correct = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM experiments;")
    total_iterations = cursor.fetchone()[0]

    conn.close()

    return {
        "total_iterations": total_iterations,
        "all_correct": all_correct,
        "success_rate": all_correct / total_iterations if total_iterations > 0 else 0,
    }


def create_checkpoint(experiment_dir, template, problems, completed_problems, db_stats, timing):
    """Create a checkpoint object from reconstructed data."""
    # Create completed experiments list
    completed_experiments = [[template, p] for p in completed_problems]

    # Calculate duration for completed experiments
    duration = int(timing["checkpoint_time"] - timing["start_time"])

    # Create results data for completed experiments
    results_data = []
    for problem in completed_problems:
        results_data.append(
            {
                "template": template,
                "problem": problem,
                "all_correct_rate": db_stats["success_rate"],
                "at_least_one_correct_rate": db_stats["success_rate"],
                "individual_duration": duration,
                "total_runs": db_stats["total_iterations"],
                "all_correct": db_stats["all_correct"],
                "at_least_one_correct": db_stats["all_correct"],
            }
        )

    # Create checkpoint
    checkpoint = {
        "output_dir": str(experiment_dir),
        "templates": [template],
        "problems": problems,
        "completed_experiments": completed_experiments,
        "results_data": results_data,
        "failed_experiments": [],
        "start_time": timing["start_time"],
        "checkpoint_time": timing["checkpoint_time"],
        "total_combinations": len(problems),
        "args_dict": {
            "iterations": db_stats["total_iterations"],
            "model": "openrouter/deepseek/deepseek-chat-v3-0324",
            "template": None,
            "problem": None,
            "dry_run": False,
            "verbose": False,
            "fail_fast": False,
        },
        "version": "1.0",
    }

    return checkpoint


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_retroactive_checkpoint.py <experiment_directory>")
        print("\nExample:")
        print("  python create_retroactive_checkpoint.py batch_results/20250929_194041")
        sys.exit(1)

    experiment_dir = Path(sys.argv[1])

    if not experiment_dir.exists():
        print(f"❌ Error: Directory not found: {experiment_dir}")
        sys.exit(1)

    print("=" * 80)
    print("🔧 RETROACTIVE CHECKPOINT CREATOR")
    print("=" * 80)
    print(f"\n📁 Experiment directory: {experiment_dir}")

    # Find log file
    log_files = list(experiment_dir.glob("experiment_run_*.log"))
    if not log_files:
        print(f"❌ Error: No log file found in {experiment_dir}")
        sys.exit(1)

    log_file = log_files[0]
    print(f"📄 Log file: {log_file.name}")

    # Find database
    db_file = experiment_dir / "experiments.db"
    if not db_file.exists():
        print(f"❌ Error: No database found: {db_file}")
        sys.exit(1)

    print(f"🗄️  Database: {db_file.name}")

    # Parse log
    print("\n🔍 Parsing log file...")
    timing = parse_log_file(log_file)
    print(f"   Start time: {datetime.fromtimestamp(timing['start_time'])}")
    print(f"   Checkpoint time: {datetime.fromtimestamp(timing['checkpoint_time'])}")
    print(f"   Template: {timing['template']}")
    print(f"   Completed problems: {timing['completed_problems']}")

    # Analyze database
    print("\n📊 Analyzing database...")
    db_stats = analyze_database(db_file)
    print(f"   Total iterations: {db_stats['total_iterations']}")
    print(f"   Successful: {db_stats['all_correct']}")
    print(f"   Success rate: {db_stats['success_rate'] * 100:.1f}%")

    # Get problems from user or use defaults
    print("\n❓ Enter problem IDs (comma-separated)")
    print("   Leave empty to use defaults from log")
    problems_input = input("   Problems: ").strip()

    if problems_input:
        problems = [p.strip() for p in problems_input.split(",")]
    else:
        # Use defaults based on completed + typical next problems
        problems = timing["completed_problems"] + ["08ed6ac7", "178fcbfb"]
        problems = list(dict.fromkeys(problems))  # Remove duplicates, preserve order

    print(f"   Using problems: {problems}")

    # Create checkpoint
    print("\n💾 Creating checkpoint...")
    checkpoint = create_checkpoint(
        experiment_dir, timing["template"], problems, timing["completed_problems"], db_stats, timing
    )

    # Save checkpoint
    checkpoint_path = experiment_dir / "checkpoint.json"
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"✅ Checkpoint created: {checkpoint_path}")

    # Show summary
    completed_count = len(timing["completed_problems"])
    total_count = len(problems)
    print("\n📊 Summary:")
    print(
        f"   Progress: {completed_count}/{total_count} ({completed_count / total_count * 100:.1f}%)"
    )
    print(f"   Remaining: {total_count - completed_count} experiments")

    # Show resume command
    print("\n🚀 To resume this experiment:")
    print("\npython run_all_problems.py \\")
    print(f'  -t "{timing["template"]}" \\')
    print(f'  -p "{",".join(problems)}" \\')
    print(f"  -i {db_stats['total_iterations']} \\")
    print(f"  -o {experiment_dir} \\")
    print("  --resume")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
