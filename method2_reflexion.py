# GOAL
# Reflexion asks a model to reflect and improve
# This variant removes the code implementations that hadn't helped but
# keeps the explanations that were wrong, to steer it
# Is there another way to get closer to the right solution?


# from litellm import completion
from dotenv import load_dotenv
from tqdm import tqdm
import logging

from db import record_run
from litellm_helper import call_llm, check_litellm_key, disable_litellm_logging
from prompt import get_func_dict, make_prompt
from run_code import execute_transform, should_request_regeneration
from utils import (
    do_first_setup,
    do_last_report,
    extract_from_code_block,
    make_message_part,
)

disable_litellm_logging()

load_dotenv()

# Initialize logger - will be overridden by run_all_problems.py if used from there
logger = logging.getLogger(__name__)


prompt_for_python_code = """
## Action to write a solution in Python

After this write a solution in Python code that follows the following format. You must accept an `initial` np.ndarray of numbers as input and return a `final` np.ndarray of numbers. Each number is in the range [0...9] and the grid is rectangular. The grid size of `final` may be different to the size of `initial`.

You can use only pure Python, numpy.

```python
import numpy as np
def transform(initial):
    assert isinstance(initial, np.ndarray)
    ... # you need to write code to generate `final`
    assert isinstance(final, np.ndarray)
    return final
```
"""

prompt_for_explanation = """
## Action to explain what you see

Given the above examples and guidance, write several bullet points that explain the rules that convert the input patterns to the output patterns. Do note write any code, just explain the rules in a block that is marked like the following and ends with:
<EXPLANATION>
...
</EXPLANATION>
"""


def call_for_new_explanation(model, messages, llm_responses):
    """Call the LLM with the description and any past explanations, ask for new explanation"""
    response, explanation_response_as_text = call_llm(model, messages)
    llm_responses.append(response)
    return explanation_response_as_text


def call_then_extract_code(model, messages, llm_responses):
    messages.append(make_message_part(prompt_for_python_code, "user"))
    response, code_response_as_text = call_llm(model, messages)
    llm_responses.append(response)
    code_as_string = extract_from_code_block(code_response_as_text)
    
    # Handle case where no code block was found (after trying multiple extraction strategies)
    if code_as_string is None:
        code_as_string = ""
        logger.warning("No code block found in LLM response after trying multiple extraction strategies")
        logger.debug(f"LLM response content: {code_response_as_text[:500]}...")  # Log first 500 chars for debugging
    
    logger.info("After calling the LLM, we get code back")
    logger.info(code_as_string)
    return code_as_string


def add_previous_explanations_to_messages(
    messages_to_get_explanation, previous_explanations
):
    """Add previous incorrect explanations to the message list to guide the LLM away from repeating them."""
    explanation_prompt_pre = (
        """Previously you wrote the following explanations, each of these are WRONG. """
        """You must not repeat these explanations, you must come up with something radically different."""
    )
    logger.info(f"Explanation prompt pre: {explanation_prompt_pre}")
    logger.info(f"Previous {len(previous_explanations)} explanations:")
    for pe in previous_explanations:
        logger.info(pe)
    messages_to_get_explanation.append(
        make_message_part(explanation_prompt_pre, "user")
    )
    for previous_explanation in previous_explanations:
        messages_to_get_explanation.append(
            make_message_part(
                "Incorrect explanation:\n" + previous_explanation, "assistant"
            )
        )
    explanation_prompt_post = (
        """Never repeat the above explanations, you must come up with something """
        """radically different. Start by explaining why this might be wrong."""
    )
    logger.info(f"Explanation prompt post: {explanation_prompt_post}")
    messages_to_get_explanation.append(
        make_message_part(explanation_prompt_post, "user")
    )


def ask_for_code_and_execute(
    model, prompt_to_describe_problem, explanation_response_as_text, llm_responses
):
    MAX_RETRY_ATTEMPTS = 3
    train_problems = problems["train"]

    # Build a new prompt using the most recent explanation (and not any historic bad explanations)
    messages_to_get_code = []
    messages_to_get_code.append(make_message_part(prompt_to_describe_problem, "user"))
    messages_to_get_code.append(make_message_part(prompt_for_explanation, "user"))
    messages_to_get_code.append(
        make_message_part(explanation_response_as_text, "assistant")
    )

    # Retry loop for structural errors
    for retry_attempt in range(MAX_RETRY_ATTEMPTS):
        code_as_string = call_then_extract_code(model, messages_to_get_code, llm_responses)

        # run the code
        rr_train, execution_outcomes, exception_message = execute_transform(code_as_string, train_problems)

        # Check if code regeneration is recommended
        should_regen = False
        if exception_message:
            should_regen, regen_reason = should_request_regeneration(exception_message, code_as_string)
            if should_regen:
                logger.warning(f"Code regeneration recommended: {regen_reason} (attempt {retry_attempt + 1}/{MAX_RETRY_ATTEMPTS})")
                logger.debug(f"Exception: {exception_message}")
            else:
                logger.info(f"Debugging preferred over regeneration: {regen_reason}")

        # If structural error and retries remaining, try again with same prompt
        if should_regen and retry_attempt < MAX_RETRY_ATTEMPTS - 1:
            retry_msg = f"🔄 Retrying with same prompt (attempt {retry_attempt + 2}/{MAX_RETRY_ATTEMPTS})..."
            logger.info(retry_msg)
            print(retry_msg)  # Also print to console for visibility
            continue

        # Either success, non-structural error, or out of retries - break and record
        if should_regen and retry_attempt == MAX_RETRY_ATTEMPTS - 1:
            failure_msg = f"❌ Failed after {MAX_RETRY_ATTEMPTS} attempts, giving up on this reflexion iteration"
            logger.error(failure_msg)
            print(failure_msg)  # Also print to console for visibility
        break

    logger.info("After executing the code, we get rr_train:")
    logger.info(rr_train)
    return rr_train, code_as_string, messages_to_get_code, execution_outcomes, exception_message


def run_experiment(
    db_filename: str,
    iteration_n: int,
    model,
    problems,
    template_name,
    rr_trains,
    llm_responses,
):
    func_dict = get_func_dict()
    REFLEXION_ITERATIONS = 5
    if REFLEXION_ITERATIONS != 5:
        # useful debug message to ian
        print(f"---REFLEXION_ITERATIONS is {REFLEXION_ITERATIONS}, not 5")
    # we need to store the previous explanations to enable reflexion
    previous_explanations = []
    # we can force the previous explanation part for debugging
    # previous_explanations = ["""<EXPLANATION>This is a word substitution puzzle where numbers are switched around</EXPLANATION>"""]
    for reflexion_n in tqdm(range(REFLEXION_ITERATIONS), leave=False):
        logger.info(
            f"Reflexion iteration {reflexion_n} on experiment iteration {iteration_n}"
        )
        prompt_to_describe_problem = make_prompt(
            template_name, problems, target="train", func_dict=func_dict
        )
        messages_to_get_explanation = [
            make_message_part(prompt_to_describe_problem, "user")
        ]
        logger.info(f"Prompt to describe problem: {prompt_to_describe_problem}")
        # if we've iterated and failed, list the previous (incorrect) explanations
        if previous_explanations:
            add_previous_explanations_to_messages(
                messages_to_get_explanation, previous_explanations
            )
        # Next ask for a new explanation
        messages_to_get_explanation.append(
            make_message_part(prompt_for_explanation, "user")
        )

        explanation_response_as_text = call_for_new_explanation(
            model, messages_to_get_explanation, llm_responses
        )
        logger.info("New explanation:")
        logger.info(explanation_response_as_text)
        previous_explanations.append(explanation_response_as_text)

        rr_train, code_as_string, messages_to_get_code, execution_outcomes, exception_message = ask_for_code_and_execute(
            model,
            prompt_to_describe_problem,
            explanation_response_as_text,
            llm_responses,
        )

        success = rr_train.transform_ran_and_matched_for_all_inputs
        if success:
            logger.info(f"------> Success on reflexion iteration {reflexion_n}")
            break
        # if we failed, we need to iterate, so just loop again

    # take the last execution result, move on
    # TODO we don't yet store reflexion_n
    record_run(
        db_filename,
        iteration_n,
        explanation_response_as_text,
        code_as_string,
        messages_to_get_code,
        success,
    )
    rr_trains.append(rr_train)
    return rr_train


def run_experiment_for_iterations(
    db_filename: str, model, iterations, problems, template_name
):
    """run experiment for many iterations"""
    llm_responses = []
    rr_trains = []

    successes = 0
    for n in tqdm(range(iterations)):
        logger.info(f"Running iteration {n}")
        run_experiment(
            db_filename,
            n,
            model,
            problems,
            template_name,
            rr_trains,
            llm_responses,
        )
        logger.info(f"rr_trains: {len(rr_trains)}")
        rr_train = rr_trains[-1]
        success = rr_train[0].transform_ran_and_matched_for_all_inputs
        successes += success
        if success:
            print(f"Successes: {successes} on {n=} of {iterations=}")
    assert len(rr_trains) == iterations
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
