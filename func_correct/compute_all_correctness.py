# %%
import json
import logging
from pathlib import Path

import attrs
import ray.data
from cloudpathlib import CloudPath
from fire import Fire

from func_correct.evaluate_correctness import LoadedProblemAndResults, evaluate_correctness
from func_correct.exec_code import Result
from func_correct.loaded_problem import DATA_DIR, TestCaseId, get_converter
from func_correct.loaded_problem_extra import LoadedProblemExtra

EXCLUSION_REASONS = {"not ran", "TIMEOUT", "ModuleNotFoundError", "OSError"}
# 'not ran' is when there is a failure

this_folder = str(Path(__file__).parent)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler(f"{this_folder}/compute_all_correctness.log", "w"))

# %%

CACHE_CHECKPOINT_FREQ = 5000

recover_checkpoints: bool = True

# %%

converter = get_converter()

with open(f"{DATA_DIR}/loaded_problem_extra_v2.jsonl", "r") as f:
    problems_extra = [converter.loads(l, LoadedProblemExtra) for l in f]

# %%

# %%
# elements are (test_id, implementation) tuples
excluded_test_impl_pairs: set[tuple[TestCaseId, str]] = set([])

cache_folder = Path(f"{DATA_DIR}/cache")
cache_folder.mkdir(exist_ok=True, parents=True)

problems_and_answers = []
for p_extra in problems_extra:
    problem = p_extra.loaded_problem

    solutions = p_extra.loaded_problem.solutions
    solutions += [mgs.solution for mgs in p_extra.generic_generated_solutions]
    solutions += [issue.generated_solution for issue in p_extra.issue_items]

    tests = [problem.test_cases]
    tests += [issue.flawed_test_cases for issue in p_extra.issue_items]
    tests += [issue.checking_test_cases for issue in p_extra.issue_items]

    for test in tests:
        problems_and_answers.append((attrs.evolve(problem, test_cases=test), solutions))

batches = [
    problems_and_answers[i : i + CACHE_CHECKPOINT_FREQ]
    for i in range(0, len(problems_and_answers), CACHE_CHECKPOINT_FREQ)
]
results_lists: list[LoadedProblemAndResults] = []
for i, batch in enumerate(batches):
    print(f"Processing batch {i} of {len(batches)}")
    if recover_checkpoints and Path(f"{cache_folder}/cache_filan_datum_{i}.json").exists():
        print("Found in cache, using it")
        r = converter.loads(
            Path(f"{cache_folder}/cache_filan_datum_{i}.json").read_text(), list[LoadedProblemAndResults]
        )
    else:
        r = evaluate_correctness(batch)
        Path(f"{cache_folder}/cache_filan_datum_{i}.json").write_text(converter.dumps(r))
    results_lists += r

# %%

# keys are (task_id, test_id, implementation) tuples; values are Result objects as dicts
problem_test_impl_to_result: dict[tuple[int, TestCaseId, str], dict] = {}

for rls in results_lists:
    task_id = rls.problem.task_id
    test_ids = rls.problem.test_cases.ids
    for impl, results in rls.results_list:
        assert len(results.results) == len(test_ids)

        # if results.failure is not None:
        #     logger.error(f"Correctness failure:\n{results.failure}")
        #     logger.debug(f"Task ID: {task_id}\nimpl:\n{impl}\ntest:\n{rls.problem.test_cases}")

        for r, test_id in zip(results.results, test_ids):
            key = (task_id, test_id, impl)

            if r.reason is not None and any(reason in r.reason for reason in EXCLUSION_REASONS):
                excluded_test_impl_pairs.add((test_id, impl))
            else:
                problem_test_impl_to_result[key] = attrs.asdict(r)
# %%
json.dump(open(f"{DATA_DIR}/excluded_test_impl_pairs.json", "w"), excluded_test_impl_pairs)
json.dump(open(f"{DATA_DIR}/problem_test_impl_to_result.json", "w"), problem_test_impl_to_result)
