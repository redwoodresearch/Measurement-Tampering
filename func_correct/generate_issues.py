# %%

import json
import os
import random
import re
import threading
from typing import Any, Optional

import openai
from tqdm import tqdm  # type: ignore

from func_correct.loaded_problem import DATA_DIR, ISSUE_GENERATION_METHOD, LoadedProblem, PythonTestCases
from func_correct.prompting import (
    example_code_for_issue_assistant_response,
    example_code_for_issue_prompt,
    example_potential_issues_assistant_response,
    example_potential_issues_prompt,
    example_potential_issues_prompt_size,
    get_code_for_issues_prompt,
    get_converter,
    get_issues_prompt,
    tokenizer,
)
from func_correct.prompting_extended import get_prompt

# %%

if not os.environ["OPENAI_API_KEY"]:
    openai.api_key_path = os.path.expanduser("~/.openai_api_key")

# %%

json_converter = get_converter()

# %%


with open(f"{DATA_DIR}/raw_functions_v3.jsonl", "r") as f:
    all_problems = [json_converter.loads(l, LoadedProblem) for l in f]
responses_code_for_issues_path = f"{DATA_DIR}/responses_code_for_issues.jsonl"
responses_potential_issues_path = f"{DATA_DIR}/responses_potential_issues.jsonl"


# %%

sub = [
    p
    for p in all_problems
    if isinstance(
        p.test_cases, PythonTestCases
    )  # TODO: we should eventually generate non-python issues (but not currently high priority)
    and p.solutions
]
print(f"{len(sub)} problems with python test cases and solutions, out of {len(all_problems)} total")
print(f"Loaded from {DATA_DIR}/raw_functions_v3.jsonl")
# %%

is_turbo = True

# %%
ProblemAndPrompts = list[tuple[LoadedProblem, list[dict[str, str]]]]

problem_and_prompts: ProblemAndPrompts = []
# for p in sub[:50]:
print(f"Generating using method {ISSUE_GENERATION_METHOD}")

# %%
for p in sub:
    actual_prompt, _, _ = get_issues_prompt(
        p, skip_base_prompt=True, target_len=1200 if is_turbo else 3000, max_len=1600 if is_turbo else 5000
    )
    if actual_prompt is not None:
        problem_and_prompts.append(
            (
                p,
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": example_potential_issues_prompt},
                    {"role": "assistant", "content": example_potential_issues_assistant_response},
                    {"role": "user", "content": actual_prompt},
                ],
            )
        )


pbar_lock = threading.Lock()


def get_completions(
    all_problems_and_prompts: ProblemAndPrompts,
    k: int,
    n_threads: int,
    model: str = "gpt-3.5-turbo",
    max_total_len: int = 4096,
    out_file: str = "temp.jsonl",
    aux_info: Optional[list[Any]] = None,
    pbar: Optional[tqdm] = None,
):
    start = k * len(all_problems_and_prompts) // n_threads
    end = (k + 1) * len(all_problems_and_prompts) // n_threads
    tmp_responses: list = []
    tmp_aux_info: list = []
    tmp_problems_and_prompts: ProblemAndPrompts = []
    if aux_info is not None:
        assert len(aux_info) == len(all_problems_and_prompts)
        aux_info = aux_info[start:end]
    all_problems_and_prompts = all_problems_and_prompts[start:end]
    print(f"{k=} {n_threads=} {start=} {end=}")

    def dump_and_clear(item: str):
        with open(out_file, "a") as f:
            for (problem, prompt), response, aux_info in zip(tmp_problems_and_prompts, tmp_responses, tmp_aux_info):
                f.write(
                    json.dumps(
                        {"problem_id": problem.task_id, "prompt": prompt, "response": response, "aux_info": aux_info}
                    )
                    + "\n"
                )
        print(f"thread {k} {item} done")
        tmp_responses.clear()
        tmp_problems_and_prompts.clear()

    for i in range(len(all_problems_and_prompts)):
        p, prompt = all_problems_and_prompts[i]
        try:
            total_len = sum(len(tokenizer.encode(s["content"])) for s in prompt) + 20
            max_tokens = max_total_len - total_len
            response = openai.ChatCompletion.create(
                model=model,
                messages=prompt,
                temperature=0,
                max_tokens=max_tokens,
            )
            tmp_responses.append(response)
            tmp_aux_info.append(aux_info[i] if aux_info is not None else None)
            tmp_problems_and_prompts.append(all_problems_and_prompts[i])

            if i % 10 == 0 and i > 0:
                dump_and_clear(str(i))

            if k == 0:
                print(f"{p.task_id=}")
                print()
                print(response["choices"][0]["message"]["content"])

            if pbar is not None:
                with pbar_lock:
                    if pbar is not None:
                        pbar.update(1)

        except Exception as e:
            print("error")
            print(e)
            print(f"{p.task_id=}")
            pass

    dump_and_clear("fin")


# %%
if ISSUE_GENERATION_METHOD == "extended":
    # truncate
    with open(responses_code_for_issues_path, "w") as f:
        ...

    problem_and_prompts = [(p, get_prompt(p)) for p in sub]

    n_threads = min(len(problem_and_prompts) // 10, 100)

    pbar = tqdm(total=len(problem_and_prompts))

    threads = [
        threading.Thread(
            target=get_completions,
            args=(
                problem_and_prompts,
                k,
                n_threads,
                "gpt-3.5-turbo" if is_turbo else "gpt-4",
                4096 if is_turbo else 8192,
                responses_code_for_issues_path,
            ),
            kwargs={"pbar": pbar},
        )
        for k in range(n_threads)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    pbar.close()

    print("done!")

    exit()
elif ISSUE_GENERATION_METHOD != "base":
    raise ValueError(f"Unknown issue generation method {ISSUE_GENERATION_METHOD}")

# %%


# truncate
with open(responses_potential_issues_path, "w") as f:
    ...


# n_threads = 50
n_threads = min(len(problem_and_prompts) // 10, 100)

threads = [
    threading.Thread(
        target=get_completions,
        args=(
            problem_and_prompts,
            k,
            n_threads,
            "gpt-3.5-turbo" if is_turbo else "gpt-4",
            4096 if is_turbo else 8192,
            responses_potential_issues_path,
        ),
    )
    for k in range(n_threads)
]

# %%

for t in threads:
    t.start()

# %%

for t in threads:
    t.join()

# %%

print("done!")

# %%

with open(responses_potential_issues_path, "r") as f:
    all_potential_issues_responses = [json.loads(l) for l in f]

# %%

# len(problem_and_prompts)
# len(all_potential_issues_responses)

# %%

# with open(f"{code_elk_setting_dir}/responses_potential_issues.jsonl", "r") as f:
#     all_potential_issues_responses = [json.loads(l) for l in f]

# %%

task_id_to_problem = {p.task_id: p for p in sub}

# %%

issue_descs_by_problem: dict[int, list[str]] = {}

for response in all_potential_issues_responses:
    all_response_lines = response["response"]["choices"][0]["message"]["content"].splitlines()
    try:
        initial_list_index = all_response_lines.index("Initial list:")
    except ValueError:
        continue

    first_after = (
        [
            i
            for i, l in enumerate(all_response_lines[initial_list_index:])
            if l.strip() == "" or l == "Reasoning for which to pick:"
        ]
        + [len(all_response_lines[initial_list_index:]) - 1]  # handle case where output was truncated
    )[0]
    theoretical_list_lines = all_response_lines[initial_list_index + 1 : initial_list_index + first_after]
    all_list_items: dict[int, str] = {}
    for maybe_list_line in theoretical_list_lines:
        match = re.match(r"(\d+)\. (.*)", maybe_list_line)
        assert match is not None
        all_list_items[int(match.group(1))] = match.group(2)

    try:
        final_list_index = all_response_lines.index("Final list:")
        final_list = [int(x.strip()) for x in all_response_lines[final_list_index + 1].split(",")]
    except ValueError:
        final_list = None

    if final_list is not None:
        extra_items = sorted({1, 2, 3, 4, 5} - set(final_list))[: max(5 - len(final_list), 0)]
        full_list = final_list + extra_items

        all_issue_descs = [all_list_items[i] for i in full_list if i in all_list_items]
    else:
        # cut last item because we might be cut off
        all_issue_descs = list(all_list_items.values())[:-1][:5]

    if len(all_issue_descs) != 0:
        issue_descs_by_problem[response["problem_id"]] = all_issue_descs

# %%

is_turbo = True

# %%

problem_and_code_for_issue_prompts: ProblemAndPrompts = []
aux_info_for_code_for_issue = []
for problem_id, issues in issue_descs_by_problem.items():
    p = task_id_to_problem[problem_id]
    solution_count = len(p.solutions)
    running_solution_start = 0
    # leave 5 extra solutions?
    solutions_per = min(max((solution_count - 5) // len(issues), 1), 3)
    for issue_desc in issues:
        actual_prompt, _, solutions_used = get_code_for_issues_prompt(
            p,
            issue_desc,
            start_solutions=running_solution_start,
            solution_limit=solutions_per,
            skip_base_prompt=True,
            target_len=1200 if is_turbo else 3000,
            max_len=2000 if is_turbo else 5000,
        )
        if actual_prompt is not None:
            problem_and_code_for_issue_prompts.append(
                (
                    p,
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": example_code_for_issue_prompt},
                        {"role": "assistant", "content": example_code_for_issue_assistant_response},
                        {"role": "user", "content": actual_prompt},
                    ],
                )
            )
            aux_info_for_code_for_issue.append(
                dict(
                    issue_desc=issue_desc,
                    solution_range=(running_solution_start, running_solution_start + solutions_used),
                )
            )
            running_solution_start += solutions_used
            running_solution_start = min(running_solution_start, solution_count - 1)

# %%

assert len(problem_and_code_for_issue_prompts) < 20_000

# %%


# truncate
with open(responses_code_for_issues_path, "w") as f:
    ...


# %%

n_threads = min(len(problem_and_code_for_issue_prompts) // 10, 50)

threads = [
    threading.Thread(
        target=get_completions,
        args=(
            problem_and_code_for_issue_prompts,
            k,
            n_threads,
            "gpt-3.5-turbo" if is_turbo else "gpt-4",
            4096 if is_turbo else 8192,
            responses_code_for_issues_path,
            aux_info_for_code_for_issue,
        ),
    )
    for k in range(n_threads)
]

# %%

for t in threads:
    t.start()

# %%

for t in threads:
    t.join()

# %%

print("done!")

# %%
