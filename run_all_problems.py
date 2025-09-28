"""Run a set of problems, report on the results"""

import importlib
from collections import Counter
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

import utils
from utils import (
    do_first_setup,
)

# from prompt import get_func_dict, make_prompt
# from run_code import execute_transform

load_dotenv()


if __name__ == "__main__":
    # parser = utils.add_argument_parser(
    #    template_name=True, model_name=True, iterations=True
    # )

    # parser.add_argument(
    #    "--module_name",
    #    type=str,
    #    nargs="?",
    #    help="module to import to run run_all_problems"
    #    " (default: %(default)s))",  # help msg 2 over lines with default
    #    default="method5_single_prompt",
    # )
    # args = parser.parse_args()
    ##print(args)

    start_dt_outer = datetime.now()
    # logger.info(f"{args=}")

    # Slightly harder problems
    # https://arcprize.org/play?task=1caeab9d
    # https://arcprize.org/play?task=1cf80156
    # https://arcprize.org/play?task=1e0a9b12
    # https://arcprize.org/play?task=1e32b0e9
    # https://arcprize.org/play?task=0607ce86
    # https://arcprize.org/play?task=06df4c85
    # all_problems_to_run = ['1caeab9d', '1cf80156', '1e0a9b12', '1e32b0e9', '0607ce86', '06df4c85']

    # Easier problems
    # https://arcprize.org/play?task=0d3d703e  # fixed colour mapping, 3x3 grid min
    # https://arcprize.org/play?task=08ed6ac7  # coloured in order of height, 9x9 grid min
    # https://arcprize.org/play?task=178fcbfb  # dots form coloured lines, 9x9 grid min
    # https://arcprize.org/play?task=9565186b  # most frequent colour wins, 3x3 grid min
    # https://arcprize.org/play?task=0a938d79  # dots form repeated coloured lines, 9x22 grid min
    # https://arcprize.org/play?task=1a07d186  # dots attach to same coloured line, 14x15 grid min
    all_problems_to_run = [
        "0d3d703e",
        "08ed6ac7",
        "178fcbfb",
        "9565186b",
        "0a938d79",
    ]
    #    "1a07d186",
    # ]
    result_rr_trains = []  # list of lists of rr_trains for each problem

    # module_name = args.module_name
    module_name = "method1_text_prompt"
    # module_name = "method2_reflexion"
    print(f"HARDCODED to use {module_name} <-------------")
    method_module = importlib.import_module(module_name)
    entry_point = method_module.run_experiment_for_iterations

    print(f"Using module: {method_module.__file__}")
    print(f"Using entry point: {entry_point.__name__}")
    print(f"Using entry point: {entry_point.__doc__}")

    all_llm_responses = []

    for problem_to_run in all_problems_to_run:
        print(f"Running problem: {problem_to_run}")
        # ignore the default problems
        args, experiment_folder, logger, start_dt, db_filename, _ = do_first_setup()
        # load a single problem
        problems = utils.get_examples(problem_to_run)
        method_module.logger = logger
        llm_responses, rr_trains = entry_point(
            db_filename,
            model=args.model_name,
            iterations=args.iterations,
            problems=problems,
            template_name=args.template_name,
        )
        all_llm_responses.extend(llm_responses)
        result_rr_trains.append(rr_trains)

    end_dt = datetime.now()
    dt_delta = end_dt - start_dt_outer

    # Create a list to store results for DataFrame
    results_data = []

    for problem_to_run, rr_trains in zip(all_problems_to_run, result_rr_trains):
        assert len(rr_trains) == args.iterations, (
            f"Expected exactly {args.iterations} rr_train per problem, got {len(rr_trains)}"
        )

        # Count successes for this problem
        all_correct = 0
        at_least_one_correct = 0

        for rr_train in rr_trains:
            ran_all_train_problems_correctly = rr_train[
                0
            ].transform_ran_and_matched_for_all_inputs
            ran_at_least_one_train_problem_correctly = rr_train[
                0
            ].transform_ran_and_matched_at_least_once

            if ran_all_train_problems_correctly:
                all_correct += 1
            if ran_at_least_one_train_problem_correctly:
                at_least_one_correct += 1

            # indicator = "✅" if ran_all_train_problems_correctly else "❌"
            # print(
            #    f"{indicator} On {problem_to_run} {ran_all_train_problems_correctly=} {ran_at_least_one_train_problem_correctly=}"
            # )

        # Add results to our data list
        results_data.append(
            {
                "problem": problem_to_run,
                "total_runs": args.iterations,
                "all_correct": all_correct,
                "at_least_one_correct": at_least_one_correct,
                "all_correct_rate": all_correct / args.iterations,
                "at_least_one_correct_rate": at_least_one_correct / args.iterations,
            }
        )

    # Create and display the DataFrame
    results_df = pd.DataFrame(results_data)
    # Reorder columns to put problem and all_correct_rate first
    column_order = [
        "problem",
        "all_correct_rate",
        "at_least_one_correct_rate",
        "total_runs",
        "all_correct",
        "at_least_one_correct",
    ]
    results_df = results_df[column_order]
    print("\nResults Summary:")
    print(results_df)
    print(f"Experiment took {dt_delta}")

    cnt_provider = Counter([response.provider for response in all_llm_responses])
    print(f"Provider counts: {cnt_provider}")

    all_token_usages = [
        llm_response.usage.total_tokens for llm_response in all_llm_responses
    ]
    print(f"Max token usage on a call was {max(all_token_usages)}")
    print(
        f"Median token usage on a call was {sorted(all_token_usages)[int(len(all_token_usages) / 2)]}"
    )

    # Save results to CSV
    # results_df.to_csv('experiment_results.csv', index=False)
    # print("\nResults saved to experiment_results.csv")
