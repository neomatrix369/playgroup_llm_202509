import argparse
import base64
import json
import logging
import os
import re
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import PurePath

import numpy as np

import analysis
from db import make_db


def setup_logging(experiment_folder):
    logging.basicConfig(
        level=logging.INFO,  # Set minimum log level
        format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)s - %(funcName)20s() ] - %(message)s",  # Log format
        filename=f"{experiment_folder}/experiment.log",  # File to write logs to
        filemode="w",
        force=True,  # else repeated use of logger in ipython results in no subsequent logs!
    )
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    logger.info(f"Experiment folder: {experiment_folder}")
    return logger


def initial_log(logger, args):
    """Initial log messages - args, githash for current commit"""
    logger.info(f"{args=}")
    githash = os.popen("git rev-parse HEAD").read().strip()
    logger.info(f"{githash=}")


def make_message_part(text, role):
    """Create another message part in OpenAI's format"""
    assert role in ["user", "assistant"]
    return {"content": [{"type": "text", "text": text}], "role": role}


def get_examples(pattern_name):
    """Load arc examples, return the requested example"""
    # path = "/media/ian/data/llms/kaggle/202406_arc/ARC-AGI/data/training"
    # path = "/media/ian/data/llms/kaggle/202503_arc_2025/arc-prize-2025"
    path = "./arc_data/arc-prize-2025"
    training_name = "arc-agi_training_challenges.json"
    filename = PurePath(path, training_name)
    # fully_filename = PurePath(path, pattern_name)
    # load the json
    with open(filename) as f:
        example = json.load(f)
    single_ex = example[pattern_name]
    return single_ex


def break_if_not_git_committed():
    """Exit if git not checked in"""
    if os.popen("git ls-files -m -o --exclude-from=.gitignore").read().strip() != "":
        print("!! UNCOMMITTED CHANGES !!")
        sys.exit(1)


RunResult = namedtuple(
    "run_result",
    [
        "code_did_execute",
        "code_ran_on_all_inputs",
        "transform_ran_and_matched_for_all_inputs",
        "transform_ran_and_matched_at_least_once",
        "transform_ran_and_matched_score",
    ],
)


class ExecutionOutcome:
    def __init__(self, initial, final, generated_final, was_correct):
        self.initial = np.array(initial)
        self.final = np.array(final)
        try:
            arr = np.array(generated_final)
            assert arr.ndim == 2
            self.generated_final = arr
        except (ValueError, AssertionError):
            # if the generated final is not a 2d grid, we can't make it an array
            self.generated_final = None
        self.was_correct = was_correct
        # generated_final_message = "..." or None?

    def __repr__(self):
        return f"""initial:\n{self.initial}\ndesired final:\n{self.final}\ntransformed initial:
{self.generated_final}\nmatches desired final: {self.was_correct}\n"""


def extract_from_code_block(text):
    "Extract the first code block in a text string"
    try:
        # result = re.search(r"```\s(.*?)\s```", text, re.DOTALL).group(1)
        # this also gets ///python
        re_groups = re.search(r"```[a-zA-Z]*\s(.*?)\s```", text, re.DOTALL)
        # group(0) is the whole match, group(1) is the first capture group
        result = re_groups.group(1)
    except AttributeError:
        result = None
    return result


def extract_explanation(text):
    """Extract content between EXPLANATION tags."""
    pattern = r"<EXPLANATION>(.*?)</EXPLANATION>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return ""  # return empty string if no explanation found



def add_argument_parser(
    problem_name=False,
    template_name=False,
    iterations=False,
    model_name=False,
    code_filename=False,
):
    parser = argparse.ArgumentParser(description=__doc__)  # read __doc__ attribute
    if problem_name:
        parser.add_argument(
            "-p",
            "--problem_name",
            type=str,
            nargs="?",
            help="name of an ARC AGI problem e.g. 9565186b"
            " (default: %(default)s))",  # help msg 2 over lines with default
            default="9565186b",
        )  # some default
    if template_name:
        parser.add_argument(
            "-t",
            "--template_name",
            type=str,
            nargs="?",
            help="template to use in ./prompts/"
            " (default: %(default)s))",  # help msg 2 over lines with default
            default="baseline_justjson.j2",
        )
    if iterations:
        parser.add_argument(
            "-i",
            "--iterations",
            type=int,
            nargs="?",
            help="number of iterations to run",
            default=1,
        )
    if model_name:
        parser.add_argument(
            "-m",
            "--model_name",
            type=str,
            nargs="?",
            help="openrouter model name (default: %(default)s)",
            # DeepSeek r3 March 2024 release (their latest)
            # https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free
            # default="openrouter/deepseek/deepseek-chat-v3-0324:free",
            # https://openrouter.ai/meta-llama/llama-4-scout/api
            default="openrouter/deepseek/deepseek-chat-v3-0324",
        )
    if code_filename:
        parser.add_argument(
            "-c",
            "--code_filename",
            type=str,
            help="Name of the file e.g. /tmp/solution.py to run",
            default="/tmp/solution.py",
        )
    return parser


def make_experiment_folder(root_folder="experiments"):
    """Create a new experiment folder with ISO8601 timestamp format
    e.g. `exp_20250919T064451` in local time"""
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    folder_name = f"exp_{timestamp}"
    full_path = PurePath(root_folder) / folder_name
    os.makedirs(full_path, exist_ok=True)
    return str(full_path)


def do_first_setup(args=None):
    # create an argparse with default args
    # we can manually add any as needed
    parser = add_argument_parser(
        problem_name=True, template_name=True, iterations=True, model_name=True
    )
    args = parser.parse_args()
    print(args)
    experiment_folder = make_experiment_folder()
    print(f"tail -n +0 -f {experiment_folder}/experiment.log")
    print(f"sqlite3 {experiment_folder}/experiments.db")
    logger = setup_logging(experiment_folder)
    initial_log(logger, args)
    start_dt = datetime.now()
    logger.info("Started experiment")

    db_filename = make_db(experiment_folder)
    logger.info(f"Database created at: {db_filename}")

    # load a single problem
    problems = get_examples(args.problem_name)
    return args, experiment_folder, logger, start_dt, db_filename, problems


def do_last_report(rr_trains, llm_responses, experiment_folder, start_dt):
    analysis.summarise_results(rr_trains)
    analysis.summarise_llm_responses(llm_responses)
    end_dt = datetime.now()
    dt_delta = end_dt - start_dt
    print(f"Experiment took {dt_delta}")
    print(f"Full logs in:\n{experiment_folder}/experiment.log")


if __name__ == "__main__":
    input = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    import representations

    print(representations.make_excel_description_of_example(input))

    exp_folder = make_experiment_folder()
    logger = setup_logging(exp_folder)
    initial_log(logger, None)
