# %%
import attrs

from func_correct.evaluate_correctness import complete_test_cases
from func_correct.loaded_problem import DATA_DIR, LoadedProblem, get_converter, prt

# %%
converter = get_converter()

with open(f"{DATA_DIR}/extended_functions_with_solutions.jsonl", "r") as f:
    problems = [converter.loads(line, LoadedProblem) for line in f]

pb_and_completed_tests = complete_test_cases([(pb, pb.test_cases) for pb in problems])

new_problems = [attrs.evolve(pb, test_cases=completed_tests) for pb, completed_tests in pb_and_completed_tests]
# %%
n_no_test_cases_before = sum(1 for pb in problems if not pb.test_cases.n)
avg_test_cases = sum(pb.test_cases.n for pb in problems) / len(problems)
n_no_test_cases_after = sum(1 for pb in new_problems if not pb.test_cases.n)
avg_test_cases_after = sum(pb.test_cases.n for pb in new_problems) / len(new_problems)
print(f"No test cases before: {n_no_test_cases_before}/{len(problems)} average per problem: {avg_test_cases:.2f}")
print(
    f"No test cases after:  {n_no_test_cases_after}/{len(new_problems)} average per problem: {avg_test_cases_after:.2f}"
)

with open(f"{DATA_DIR}/raw_functions_v3.jsonl", "w") as f:
    for problem in new_problems:
        if problem.test_cases.n:
            f.write(converter.dumps(problem) + "\n")

# %%
if __name__ == "__main__":
    prt(pb_and_completed_tests[::1000])

# %%
