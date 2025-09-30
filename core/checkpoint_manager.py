"""
Checkpoint and resumption system for batch experiments.

Enables experiments to be interrupted and resumed without losing progress.
Saves checkpoints after each completed experiment (template + problem combination).

Following SOLID principles:
- Single Responsibility: Handles checkpoint save/load/resume logic only
- Open/Closed: Easy to extend checkpoint format without modification
- Dependency Inversion: Works with any result/progress data structure
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class ExperimentCheckpoint:
    """
    Value object representing a checkpoint state.
    
    Encapsulates all information needed to resume an experiment run.
    """
    
    def __init__(
        self,
        output_dir: str,
        templates: List[str],
        problems: List[str],
        completed_experiments: List[Tuple[str, str]],
        results_data: List[Dict[str, Any]],
        failed_experiments: List[Dict[str, Any]],
        start_time: float,
        checkpoint_time: float,
        total_combinations: int,
        args_dict: Dict[str, Any]
    ):
        self.output_dir = output_dir
        self.templates = templates
        self.problems = problems
        self.completed_experiments = completed_experiments  # List of (template, problem) tuples
        self.results_data = results_data
        self.failed_experiments = failed_experiments
        self.start_time = start_time
        self.checkpoint_time = checkpoint_time
        self.total_combinations = total_combinations
        self.args_dict = args_dict
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for JSON serialization."""
        return {
            'output_dir': self.output_dir,
            'templates': self.templates,
            'problems': self.problems,
            'completed_experiments': self.completed_experiments,
            'results_data': self.results_data,
            'failed_experiments': self.failed_experiments,
            'start_time': self.start_time,
            'checkpoint_time': self.checkpoint_time,
            'total_combinations': self.total_combinations,
            'args_dict': self.args_dict,
            'version': '1.0'
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentCheckpoint':
        """Create checkpoint from dictionary."""
        return cls(
            output_dir=data['output_dir'],
            templates=data['templates'],
            problems=data['problems'],
            completed_experiments=[tuple(exp) for exp in data['completed_experiments']],
            results_data=data['results_data'],
            failed_experiments=data['failed_experiments'],
            start_time=data['start_time'],
            checkpoint_time=data['checkpoint_time'],
            total_combinations=data['total_combinations'],
            args_dict=data['args_dict']
        )
    
    def get_progress_summary(self) -> str:
        """Get human-readable progress summary."""
        completed = len(self.completed_experiments)
        total = self.total_combinations
        percentage = (completed / total * 100) if total > 0 else 0
        
        elapsed = self.checkpoint_time - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        return (
            f"Progress: {completed}/{total} experiments ({percentage:.1f}%)\n"
            f"Time elapsed: {hours}h {minutes}m\n"
            f"Last completed: {self.completed_experiments[-1] if self.completed_experiments else 'None'}"
        )
    
    def get_next_experiment(self) -> Optional[Tuple[str, str, int]]:
        """
        Get the next experiment to run.
        
        Returns:
            Tuple of (template, problem, test_number) or None if all complete
        """
        completed_set = set(self.completed_experiments)
        test_number = len(self.completed_experiments) + 1
        
        for template in self.templates:
            for problem in self.problems:
                if (template, problem) not in completed_set:
                    return (template, problem, test_number)
        
        return None


class CheckpointManager:
    """
    Manages checkpoint saving and loading for experiment resumption.
    
    Provides automatic checkpointing after each experiment and
    easy resumption from interrupted runs.
    """
    
    CHECKPOINT_FILENAME = "checkpoint.json"
    
    def __init__(self, output_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory where checkpoints will be saved
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_path = self.output_dir / self.CHECKPOINT_FILENAME
    
    def save_checkpoint(self, checkpoint: ExperimentCheckpoint) -> None:
        """
        Save checkpoint to disk.
        
        Args:
            checkpoint: ExperimentCheckpoint to save
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write checkpoint with pretty formatting for human readability
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
    
    def load_checkpoint(self) -> Optional[ExperimentCheckpoint]:
        """
        Load checkpoint from disk if it exists.
        
        Returns:
            ExperimentCheckpoint if found, None otherwise
        """
        if not self.checkpoint_path.exists():
            return None
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
            return ExperimentCheckpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠️  Warning: Could not load checkpoint: {e}")
            return None
    
    def checkpoint_exists(self) -> bool:
        """Check if a checkpoint file exists."""
        return self.checkpoint_path.exists()
    
    def delete_checkpoint(self) -> None:
        """Delete the checkpoint file (e.g., after successful completion)."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
    
    def archive_checkpoint(self) -> None:
        """Archive the checkpoint with timestamp (for debugging)."""
        if self.checkpoint_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"checkpoint_completed_{timestamp}.json"
            archive_path = self.output_dir / archive_name
            self.checkpoint_path.rename(archive_path)
    
    @staticmethod
    def find_latest_checkpoint(batch_results_dir: Path = Path("batch_results")) -> Optional[Path]:
        """
        Find the most recent checkpoint file across all batch result directories.
        
        Args:
            batch_results_dir: Root directory containing batch result folders
            
        Returns:
            Path to the most recent checkpoint file, or None if not found
        """
        if not batch_results_dir.exists():
            return None
        
        checkpoints = []
        for run_dir in batch_results_dir.iterdir():
            if run_dir.is_dir():
                checkpoint_file = run_dir / CheckpointManager.CHECKPOINT_FILENAME
                if checkpoint_file.exists():
                    checkpoints.append(checkpoint_file)
        
        if not checkpoints:
            return None
        
        # Sort by modification time, most recent first
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    @staticmethod
    def prompt_resume_from_checkpoint(checkpoint: ExperimentCheckpoint) -> bool:
        """
        Prompt user whether to resume from checkpoint.
        
        Args:
            checkpoint: The checkpoint to potentially resume from
            
        Returns:
            True if user wants to resume, False otherwise
        """
        print("\n" + "=" * 80)
        print("🔄 CHECKPOINT FOUND - PREVIOUS RUN WAS INTERRUPTED")
        print("=" * 80)
        print(f"\n📊 Checkpoint Details:")
        print(f"   Output directory: {checkpoint.output_dir}")
        print(f"   {checkpoint.get_progress_summary()}")
        
        next_exp = checkpoint.get_next_experiment()
        if next_exp:
            template, problem, test_num = next_exp
            print(f"\n▶️  Would resume from:")
            print(f"   Test {test_num}/{checkpoint.total_combinations}: {template} + {problem}")
        
        print("\n" + "=" * 80)
        response = input("Resume from checkpoint? [Y/n]: ").strip().lower()
        return response in ('', 'y', 'yes')
