import json
import logging
import shutil
import subprocess
import tempfile
import threading
import traceback
from pathlib import Path
from time import time
from typing import Any, Literal, Optional, Union, overload

from attrs import frozen

from func_correct.loaded_problem import ToEval, get_converter


@frozen
class Result:
    """Result of a single test case."""

    correct: bool = False
    reason: Optional[str] = "not ran"


@frozen
class Results:
    """Results of the execution of a solution."""

    results: list[Result]  # list of length n_expected_results, (even when there is a failure)
    failure: Optional[str] = None  # str if exception


@frozen
class ExecutionResult:
    """Result of a single test case."""

    code: Union[ToEval, Any] = None
    reason: Optional[str] = "not ran"


@frozen
class ExecutionResults:
    """Results of the execution of a solution."""

    results: list[ExecutionResult]  # list of length n_expected_results, (even when there is a failure)
    failure: Optional[str] = None  # str if exception


SPLIT_FACTOR = 5
MIN_NB_CODES_ALLOWED_SPLIT = 20

this_folder = str(Path(__file__).parent)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler(f"{this_folder}/exec.log", "w"))


@overload
def sandboxed_check_correctness(
    codes: list[str], n_expected_results: list[int], timeout_per_res: float, just_run: Literal[True]
) -> list[ExecutionResults]:
    ...


@overload
def sandboxed_check_correctness(
    codes: list[str], n_expected_results: list[int], timeout_per_res: float, just_run: Literal[False]
) -> list[Results]:
    ...


# Make mypy happy
@overload
def sandboxed_check_correctness(
    codes: list[str], n_expected_results: list[int], timeout_per_res: float, just_run: bool
) -> Union[list[Results], list[ExecutionResults]]:
    ...


def sandboxed_check_correctness(
    codes: list[str], n_expected_results: list[int], timeout_per_res: float = 0.1, just_run: bool = False
) -> Union[list[Results], list[ExecutionResults]]:
    """Return the results of the sandboxed execution of the codes.

    Each code should populate a local variable results____ of the format [{correct: bool, reason: Optional[str]}]
    with the results of the tests, of length corresponding to the entry in n_expected_result."""

    if len(codes) != len(n_expected_results):
        raise ValueError(f"len(codes) != len(n_expected_results): {len(codes)} != {len(n_expected_results)}")

    current_folder = Path(__file__).parent.absolute()
    tmp_dir = Path().home() / ".tmp" / "exec_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=tmp_dir) as temp_dir:
        folder = Path(f"{temp_dir}/exec/scripts_to_test")
        folder.mkdir(parents=True, exist_ok=True)

        # Copy the dockerfile and the unsafe_code_execution.py file
        shutil.copy(f"{current_folder}/exec/Dockerfile", f"{temp_dir}/exec")
        shutil.copy(f"{current_folder}/exec/unsafe_code_exec.py", f"{temp_dir}/exec")

        # write config to file
        json.dump(
            {"timeout": timeout_per_res, "n_expected_results": n_expected_results, "just_run": just_run},
            open(f"{temp_dir}/exec/config.json", "w"),
        )

        # write python files
        for i, code in enumerate(codes):
            Path(f"{folder}/{i}.py").write_text(code)

        # build a docker image, don't show stdout:
        # cmd = f"sudo docker build -t exec{id} {temp_dir}/exec"
        # subprocess.run(cmd.split(), stdout=subprocess.DEVNULL)

        # execute the docker image with runsc and get the stdout
        max_time = int(10 + timeout_per_res * 1.1 * sum(n_expected_results))

        thread_ident = threading.get_ident()
        st = time()
        logger.debug(f"Run {len(codes)} codes with a timeout of {max_time:.3f} seconds (thread {thread_ident})")
        cmd = f"sudo timeout --kill-after=10s {max_time}s docker run --rm --runtime=runsc --cpus=2 --memory=512m $(sudo docker build -q {temp_dir}/exec)"
        output = subprocess.run(cmd, capture_output=True, shell=True)

    logger.debug(
        f"Ran {len(codes)} codes with a timeout of {max_time:.3f} seconds done in {time() - st:.3f} seconds (thread {thread_ident})"
    )

    string_out = output.stdout.decode("utf-8")
    string_err = output.stderr.decode("utf-8")
    if string_err:
        logger.error(f"Something went wrong with the sandboxed execution:\n{string_err}")

    try:
        # just look at the last line to avoid looking at garbage if the IO capture failed
        if just_run:
            out = get_converter().loads(string_out.splitlines()[-1], list[ExecutionResults])
        else:
            out = get_converter().loads(string_out.splitlines()[-1], list[Results])

        if len(out) != len(codes):
            raise ValueError(f"len(out) != len(codes): {len(out)} != {len(codes)}")

        return out
    except Exception:
        if len(codes) < MIN_NB_CODES_ALLOWED_SPLIT:
            # print(f"WARNING: Something went wrong with the sandboxed execution:\n{repr(string_out)}", flush=True)
            logger.exception(f"Something went wrong with the sandboxed execution:\n{repr(string_out)}")
            for i, code in enumerate(codes):
                logger.debug(f"Code {i}:\n{code} (expected {n_expected_results[i]} results)")

            exec_str = f"BATCH_ERR: {traceback.format_exc()}"

            if just_run:
                return [
                    ExecutionResults([ExecutionResult()] * n_expected_results[i], failure=exec_str)
                    for i in range(len(codes))
                ]
            else:
                return [Results([Result()] * n_expected_results[i], failure=exec_str) for i in range(len(codes))]
        else:
            logger.warning(f"Splitting the codes to try again:\n{repr(string_out)}")
            # Iteratively split the codes and try again
            splits = [
                (codes[i : i + len(codes) // SPLIT_FACTOR], n_expected_results[i : i + len(codes) // SPLIT_FACTOR])
                for i in range(0, len(codes), len(codes) // SPLIT_FACTOR)
            ]
            r: Union[list[Results], list[ExecutionResults]] = []  # type: ignore
            for codes_split, n_expect_split in splits:
                r += sandboxed_check_correctness(codes_split, n_expect_split, timeout_per_res, just_run)  # type: ignore
            return r


# sudo docker run --rm --runtime=runsc --ulimit cpu=1 $(sudo docker build -q exec)
