"""Test retry logic for code regeneration on structural errors."""
import pytest
from unittest.mock import Mock, patch, call
from run_code import should_request_regeneration


def test_should_request_regeneration_structural_errors():
    """Test that structural errors return should_regen=True"""
    # Test 1: Data instead of code
    exception_msg = "Code validation failed: Code looks like data/output"
    should_regen, reason = should_request_regeneration(exception_msg, "1 2 3")
    assert should_regen is True
    assert "STRUCTURAL" in reason

    # Test 2: Missing transform function
    exception_msg = "name 'transform' is not defined"
    should_regen, reason = should_request_regeneration(exception_msg, "def foo(): pass")
    assert should_regen is True
    assert "STRUCTURAL" in reason

    # Test 3: Wrong function signature
    exception_msg = "transform() missing 1 required positional argument"
    should_regen, reason = should_request_regeneration(exception_msg, None)
    assert should_regen is True
    assert "STRUCTURAL" in reason


def test_should_request_regeneration_logic_errors():
    """Test that logic errors return should_regen=False"""
    # Test 1: AssertionError
    exception_msg = "execute_transform caught: assertion failed of type(e)=<class 'AssertionError'>"
    should_regen, reason = should_request_regeneration(exception_msg, None)
    assert should_regen is False
    assert "LOGIC" in reason

    # Test 2: IndexError
    exception_msg = "execute_transform caught: list index out of range of type(e)=<class 'IndexError'>"
    should_regen, reason = should_request_regeneration(exception_msg, None)
    assert should_regen is False
    assert "LOGIC" in reason

    # Test 3: ZeroDivisionError
    exception_msg = "execute_transform caught: division by zero of type(e)=<class 'ZeroDivisionError'>"
    should_regen, reason = should_request_regeneration(exception_msg, None)
    assert should_regen is False
    assert "LOGIC" in reason


def test_should_request_regeneration_syntax_errors():
    """Test that syntax errors return should_regen=True"""
    exception_msg = "SyntaxError: invalid syntax"
    should_regen, reason = should_request_regeneration(exception_msg, None)
    assert should_regen is True
    assert "SYNTAX" in reason


def test_should_request_regeneration_no_error():
    """Test that no error returns should_regen=False"""
    should_regen, reason = should_request_regeneration(None, None)
    assert should_regen is False
    assert reason == "No error"

    should_regen, reason = should_request_regeneration("", None)
    assert should_regen is False
    assert reason == "No error"


@patch('method1_text_prompt.call_llm')
@patch('method1_text_prompt.execute_transform')
def test_retry_logic_structural_error(mock_execute_transform, mock_call_llm):
    """Test that structural errors trigger retry up to 3 times"""
    from method1_text_prompt import run_experiment
    from utils import RunResult, ExecutionOutcome

    # Mock LLM to always return bad code (data instead of code)
    mock_call_llm.return_value = (Mock(), "```python\n1 2 3\n4 5 6\n```")

    # Mock execute_transform to return structural error
    bad_result = RunResult(
        code_did_execute=False,
        code_ran_on_all_inputs=False,
        transform_ran_and_matched_for_all_inputs=False,
        transform_ran_and_matched_at_least_once=False,
        transform_ran_and_matched_score=0
    )
    mock_execute_transform.return_value = (
        bad_result,
        [],
        "Code validation failed: Code looks like data/output"
    )

    # Run experiment
    rr_trains = []
    llm_responses = []
    messages = [{"role": "user", "content": "test prompt"}]
    problems = {"train": [{"input": [[1, 2]], "output": [[3, 4]]}]}

    with patch('method1_text_prompt.extract_from_code_block', return_value="1 2 3"):
        with patch('method1_text_prompt.extract_explanation', return_value="test explanation"):
            with patch('method1_text_prompt.record_run'):
                run_experiment(
                    "test.db",
                    0,
                    "test-model",
                    problems,
                    messages,
                    rr_trains,
                    llm_responses
                )

    # Verify LLM was called 3 times (retry logic)
    assert mock_call_llm.call_count == 3, f"Expected 3 LLM calls, got {mock_call_llm.call_count}"
    assert mock_execute_transform.call_count == 3, f"Expected 3 execute_transform calls, got {mock_execute_transform.call_count}"


@patch('method1_text_prompt.call_llm')
@patch('method1_text_prompt.execute_transform')
def test_retry_logic_success_first_try(mock_execute_transform, mock_call_llm):
    """Test that successful code executes only once (no retry)"""
    from method1_text_prompt import run_experiment
    from utils import RunResult

    # Mock LLM to return good code
    mock_call_llm.return_value = (Mock(), "```python\ndef transform(x): return x\n```")

    # Mock execute_transform to return success
    good_result = RunResult(
        code_did_execute=True,
        code_ran_on_all_inputs=True,
        transform_ran_and_matched_for_all_inputs=True,
        transform_ran_and_matched_at_least_once=True,
        transform_ran_and_matched_score=1
    )
    mock_execute_transform.return_value = (good_result, [], None)

    # Run experiment
    rr_trains = []
    llm_responses = []
    messages = [{"role": "user", "content": "test prompt"}]
    problems = {"train": [{"input": [[1, 2]], "output": [[1, 2]]}]}

    with patch('method1_text_prompt.extract_from_code_block', return_value="def transform(x): return x"):
        with patch('method1_text_prompt.extract_explanation', return_value="test"):
            with patch('method1_text_prompt.record_run'):
                run_experiment(
                    "test.db",
                    0,
                    "test-model",
                    problems,
                    messages,
                    rr_trains,
                    llm_responses
                )

    # Verify LLM was called only once (no retry needed)
    assert mock_call_llm.call_count == 1, f"Expected 1 LLM call, got {mock_call_llm.call_count}"
    assert mock_execute_transform.call_count == 1, f"Expected 1 execute_transform call, got {mock_execute_transform.call_count}"