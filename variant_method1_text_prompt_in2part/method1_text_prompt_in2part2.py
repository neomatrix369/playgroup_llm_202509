# GOAL
# Can you make a generic prompt that correctly describes the rules governgin how
# the initial grid turns into the final grid?
# BONUS can you make it write code that solves this?


from dotenv import load_dotenv
from tqdm import tqdm

from db import record_run
from litellm_helper import call_llm, check_litellm_key, disable_litellm_logging
from prompt import get_func_dict, make_prompt
from run_code import execute_transform

# from litellm import completion
from utils import (
    do_first_setup,
    do_last_report,
    extract_explanation,
    extract_from_code_block,
    make_message_part,
)

disable_litellm_logging()

load_dotenv()

# method1_text_prompt_in2part2.py -t baseline_wquotedgridcsv_excel_for2parts.j2 -p 08ed6ac7 -i 5

prompt_for_2nd_part = """

After this write a solution in Python code that follows the following format. You must accept an `initial` np.ndarray of numbers as input and return a `final` np.ndarray of numbers. Each number is in the range [0...9] and the grid is rectangular. The grid size of `final` may be different to the size of `initial`.

```python
import numpy as np
def transform(initial):
    assert isinstance(initial, np.ndarray)
    ... # you need to write code to generate `final`
    assert isinstance(final, np.ndarray)
    return final
```
"""


def run_experiment(
    db_filename: str,
    iteration_n: int,
    model,
    problems,
    messages,
    rr_trains,
    llm_responses,
):
    # in this variant we ask for explanation first, not code
    response, content = call_llm(model, messages)
    llm_responses.append(response)
    logger.info(f"Content: {content}")
    # content will be the explanation only
    messages_plus_response = messages + [make_message_part(content, "assistant")]
    # now ask it to build on the explanation to write code
    messages_plus_response.append(make_message_part(prompt_for_2nd_part, "user"))
    response, content = call_llm(model, messages_plus_response)
    llm_responses.append(response)
    logger.info(f"Content: {content}")
    code_as_string = extract_from_code_block(content)
    
    # Handle case where no code block was found (after trying multiple extraction strategies)
    if code_as_string is None:
        code_as_string = ""
        logger.warning("No code block found in LLM response after trying multiple extraction strategies")
        logger.debug(f"LLM response content: {content[:500]}...")  # Log first 500 chars for debugging

    train_problems = problems["train"]
    rr_train = execute_transform(code_as_string, train_problems)
    explanation = extract_explanation(content)
    record_run(
        db_filename,
        iteration_n,
        explanation,
        code_as_string,
        messages_plus_response,
        rr_train[0].transform_ran_and_matched_for_all_inputs,
    )
    logger.info(f"RR train: {rr_train}")
    rr_trains.append(rr_train)


def run_experiment_for_iterations(
    db_filename: str, model, iterations, problems, template_name
):
    """method1_text_prompt's run experiment"""
    llm_responses = []
    rr_trains = []

    # make a prompt before calling LLM
    func_dict = get_func_dict()
    initial_prompt = make_prompt(
        template_name, problems, target="train", func_dict=func_dict
    )
    logger.info(f"Prompt: {initial_prompt}")

    messages = [make_message_part(initial_prompt, "user")]

    for n in tqdm(range(iterations)):
        run_experiment(
            db_filename,
            n,
            model,
            problems,
            messages,
            rr_trains,
            llm_responses,
        )
    return llm_responses, rr_trains


if __name__ == "__main__":
    # We can force the code to _only run_ if fully checked it
    # if BREAK_IF_NOT_CHECKED_IN:
    #    utils.break_if_not_git_committed()

    args, experiment_folder, logger, start_dt, db_filename, problems = do_first_setup()
    check_litellm_key(args)  # note this will check any provider

    llm_responses, rr_trains = run_experiment_for_iterations(
        db_filename=db_filename,
        model=args.model_name,
        iterations=args.iterations,
        problems=problems,
        template_name=args.template_name,
    )

    do_last_report(rr_trains, llm_responses, experiment_folder, start_dt)
