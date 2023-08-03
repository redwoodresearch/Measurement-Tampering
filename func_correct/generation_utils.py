import re
import threading
import time
from time import sleep
from typing import Optional, TypeVar

import attrs
import openai
from tqdm import tqdm

from func_correct.prompting import tokenizer


@attrs.frozen
class GenerationParams:
    temperature: float
    model: str


TOKENS_PER_MODEL = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
}
MAX_RPM_PER_MODEL = {
    "gpt-3.5-turbo": 3500,
    "gpt-4": 200,
}
MAX_TPM_PER_MODEL = {
    "gpt-3.5-turbo": 90_000,
    "gpt-4": 40_000,
}

MAX_TOKENS_MARGIN = 20


def strip_triple_backquotes(s) -> Optional[str]:
    """Remove code inside fences.

    Support both raw fences and python fences (with triple backquotes).

    Return None if code fence is not closed.
    """

    closed_fence_result = re.search(r"(```(?:python\n)?)([\s\S]*?)(```)", s)

    if closed_fence_result is not None:
        return closed_fence_result.group(2)
    else:
        unclosed_fence_result = re.search(r"```(.*?)", s)

        if unclosed_fence_result is not None:
            return None
        else:
            return s


def nb_tokens_in_prompt(prompt_messages: list[dict[str, str]]) -> int:
    return sum(len(tokenizer.encode(s["content"])) for s in prompt_messages) + MAX_TOKENS_MARGIN


def generate(
    prompt_messages: list[dict[str, str]],
    nb_solutions: int = 5,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.7,
    model: str = "gpt-3.5-turbo",
    max_attemps: int = 10,
    only_accept_finished: bool = True,
    sleep_time: float = 0,
    max_sleep_time_when_fail: float = 30,
):
    if max_new_tokens is None:
        assert model in TOKENS_PER_MODEL

    # chat models
    if model == "gpt-3.5-turbo" or model == "gpt-4":
        for attempt in range(max_attemps):
            try:
                st = time.time()
                max_tokens_possible = TOKENS_PER_MODEL[model] - nb_tokens_in_prompt(prompt_messages)
                if max_new_tokens is None:
                    max_new_tokens = max_tokens_possible
                else:
                    max_new_tokens = min(max_new_tokens, max_tokens_possible)

                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=prompt_messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    n=nb_solutions,
                )
                generations = [
                    choice["message"]["content"]
                    for choice in completion["choices"]
                    if (choice["finish_reason"] == "stop") or (not only_accept_finished)
                ]
                request_time = time.time() - st
                if request_time < sleep_time:
                    sleep(sleep_time - request_time)
                break
            except Exception as e:
                generations = []
                if attempt == max_attemps - 1:
                    print(f"Failed {max_attemps} times, last error: {e}")
                sleep(max_sleep_time_when_fail * 2 ** (-(max_attemps - attempt - 1)))
    else:
        raise ValueError(f"Model {model} not supported")

    return generations


T = TypeVar("T")


def threaded_generations(
    prompts_messages: list[tuple[T, list[dict[str, str]]]],
    nb_solutions: int = 1,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.7,
    model: str = "gpt-3.5-turbo",
    max_attemps: int = 10,
    only_accept_finished: bool = True,
    n_threads: int = 5,
) -> list[tuple[T, list[str]]]:
    threads_results: list[list[tuple[T, list[str]]]] = [[] for _ in range(n_threads)]

    pbar = tqdm(total=len(prompts_messages))
    lock = threading.Lock()

    active_threads = n_threads

    def thread_task(thread_id: int):
        nonlocal active_threads

        start = thread_id * len(prompts_messages) // n_threads
        end = (thread_id + 1) * len(prompts_messages) // n_threads
        prompts_to_process = prompts_messages[start:end]
        thread_results = []
        for obj, prompt in prompts_to_process:
            effective_max_rpm = min(MAX_RPM_PER_MODEL[model], MAX_TPM_PER_MODEL[model] / nb_tokens_in_prompt(prompt))

            sleep_time = (60 / effective_max_rpm) * active_threads

            generations = generate(
                prompt, nb_solutions, max_new_tokens, temperature, model, max_attemps, only_accept_finished, sleep_time
            )
            thread_results.append((obj, generations))

            with lock:
                pbar.update(1)

        threads_results[thread_id] = thread_results
        with lock:
            active_threads -= 1

    threads = []
    for thread_id in range(n_threads):
        thread = threading.Thread(target=thread_task, args=(thread_id,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    pbar.close()

    return sum(threads_results, [])
