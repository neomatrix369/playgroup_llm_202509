# GOAL
# Can you make a generic prompt that correctly describes the rules governgin how
# the initial grid turns into the final grid?
# BONUS can you make it write code that solves this?


from dotenv import load_dotenv
from tqdm import tqdm
import logging

from db import record_run
from litellm_helper import call_llm, check_litellm_key, disable_litellm_logging
from prompt import get_func_dict, make_prompt
from run_code import execute_transform, should_request_regeneration

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

# Initialize logger - will be overridden by run_all_problems.py if used from there
logger = logging.getLogger(__name__)


def run_experiment(
    db_filename: str,
    iteration_n: int,
    model,
    problems,
    messages,
    rr_trains,
    llm_responses,
):
    response, content = call_llm(model, messages)
    llm_responses.append(response)
    logger.info(f"Content: {content}")
    messages_plus_response = messages + [make_message_part(content, "assistant")]

    code_as_string = extract_from_code_block(content)
    
    # Handle case where no code block was found (after trying multiple extraction strategies)
    if code_as_string is None:
        code_as_string = ""
        logger.warning("No code block found in LLM response after trying multiple extraction strategies")
        logger.debug(f"LLM response content: {content[:500]}...")  # Log first 500 chars for debugging

    train_problems = problems["train"]
    rr_train, execution_outcomes, exception_message = execute_transform(code_as_string, train_problems)

    # Check if code regeneration is recommended
    if exception_message:
        should_regen, regen_reason = should_request_regeneration(exception_message, code_as_string)
        if should_regen:
            logger.warning(f"Code regeneration recommended: {regen_reason}")
            logger.debug(f"Exception: {exception_message}")
        else:
            logger.info(f"Debugging preferred over regeneration: {regen_reason}")

    explanation = extract_explanation(content)
    record_run(
        db_filename,
        iteration_n,
        explanation,
        code_as_string,
        messages_plus_response,
        rr_train.transform_ran_and_matched_for_all_inputs,
    )
    logger.info(f"RR train: {rr_train}")
    rr_trains.append((rr_train, execution_outcomes, exception_message))


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
