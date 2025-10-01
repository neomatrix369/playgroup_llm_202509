import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

# sqlite3 exp_fold/experiments.db
# .tables # shows experiments table
# .schema experiments # shows the schema
# select * from experiments; # shows all entries
# .exit

# get the final explanation or code where the code correctly transformed all train inputs
# select final_explanation from experiments where all_train_transformed_correctly = True;
# select code from experiments where all_train_transformed_correctly = True;
# get an average of how often the code correctly transformed all train inputs
# select avg(all_train_transformed_correctly) from experiments;


def make_db(db_path: str, db_filename: str = "experiments.db") -> None:
    """
    Create an empty SQLite database with the specified schema.

    Args:
        db_path: Path where the SQLite database should be created

    Schema:
        - iteration (integer): Iteration number
        - final_explanation (string): Final explanation text
        - code (string): Generated code
        - conversation_json (string): JSON string of conversation data
        - all_train_transformed_correctly (bool): Boolean flag for training transformation success
    """
    # Ensure the directory exists
    db_full_path = Path(db_path) / db_filename
    os.makedirs(os.path.dirname(db_full_path), exist_ok=True)

    # Connect to the database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_full_path)
    cursor = conn.cursor()

    # Create the table with the specified schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            iteration INTEGER,
            final_explanation TEXT,
            code TEXT,
            conversation_json TEXT,
            all_train_transformed_correctly BOOLEAN
        )
    """)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    # print(f"Database created successfully at: {db_full_path}")
    return db_full_path


def record_run(
    db_filename: str,
    iteration: int,
    final_explanation: str,
    code: str,
    conversation: List[Dict[str, Any]],
    all_train_transformed_correctly: bool,
) -> None:
    """
    Record a run in the database.

    Args:
        db_filename: Path to the SQLite database file
        iteration: Iteration number
        final_explanation: Final explanation text
        code: Generated code
        conversation: List of dictionaries representing the conversation
        all_train_transformed_correctly: Boolean flag for training transformation success
    """
    # Convert conversation list to JSON string
    conversation_json = json.dumps(conversation)

    # Connect to the database
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    # Insert the record
    cursor.execute(
        """
        INSERT INTO experiments 
        (iteration, final_explanation, code, conversation_json, all_train_transformed_correctly)
        VALUES (?, ?, ?, ?, ?)
    """,
        (
            iteration,
            final_explanation,
            code,
            conversation_json,
            all_train_transformed_correctly,
        ),
    )

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    # print(f"Record inserted successfully for iteration {iteration}")


if __name__ == "__main__":
    from utils import make_experiment_folder

    # Example usage
    exp_folder = make_experiment_folder()
    # db_path = Path(exp_folder) / "experiments.db"
    db_filename = make_db(exp_folder)
    print(f"Database created at: {db_filename}")
    record_run(db_filename, 0, "Test explanation", "print('Hello, world!')", [], True)
