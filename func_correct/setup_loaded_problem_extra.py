# %%
# %load_ext autoreload
# %autoreload 2

import ast
import json
import re
# %%
from collections import Counter, defaultdict
from typing import Any

import attrs
from black import FileMode, format_str
from black.parsing import InvalidInput as BlackInvalidInput
from blib2to3.pgen2.tokenize import TokenError  # type: ignore
from tqdm import tqdm

from func_correct.evaluate_correctness import complete_test_cases
from func_correct.generate_solutions import LoadedProblemWithGeneratedAnswers
from func_correct.loaded_problem import (
    DATA_DIR,
    ISSUE_GENERATION_METHOD,
    LoadedProblem,
    PythonTestCases,
    TestCases,
    ToEval,
)
from func_correct.loaded_problem_extra import LoadedProblemExtra, ModelGeneratedSolution, ProblemIssueItem
from func_correct.prompting import get_converter

# %%


def is_correct_python_code(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def is_indented_correct_python_code(code: str) -> bool:
    indented_code = "\n".join(["def f():"] + ["    " + line for line in code.splitlines()])
    return is_correct_python_code(indented_code)


def check_unique_pb_ids(problems: list[LoadedProblem]):
    assert len(problems) == len(set(p.task_id for p in problems))


def check_problems_fnname(problem: LoadedProblemExtra):
    fn_names_found = set()
    if isinstance(problem.loaded_problem.test_cases, PythonTestCases):
        fn_names_found.add(problem.loaded_problem.test_cases.fn_name)
    for issue in problem.issue_items:
        if isinstance(issue.flawed_test_cases, PythonTestCases):
            fn_names_found.add(issue.flawed_test_cases.fn_name)
        if isinstance(issue.checking_test_cases, PythonTestCases):
            fn_names_found.add(issue.checking_test_cases.fn_name)
    assert (
        len(fn_names_found) <= 1
    ), f"Found {len(fn_names_found)} different function names for problem {problem.loaded_problem.task_id}."


# %%

json_converter = get_converter()

# %%

with open(f"{DATA_DIR}/raw_functions_v3.jsonl", "r") as f:
    all_problems = [json_converter.loads(line, LoadedProblem) for line in f]

check_unique_pb_ids(all_problems)

# %%

# NOTE: only these cases for now!
sub = [
    p
    for p in all_problems
    # if p.difficulty in ["apps/introductory", "mbpp"] and isinstance(p.test_cases, PythonTestCases) # TODO @Fabien remove me when possible
    # and "class Solution" not in p.solutions[0] # does nothing
]

sub_dict = {p.task_id: p for p in sub}
# %%
print("Number of LoadedProblem objects:", len(sub))
# %%

with open(f"{DATA_DIR}/responses_code_for_issues.jsonl", "r") as f:
    all_code_for_issue_responses = [json.loads(line) for line in f]

print(f"Loaded {len(all_code_for_issue_responses)} issues. from {DATA_DIR}/responses_code_for_issues.jsonl")
# %%

task_id_to_problem = {p.task_id: p for p in sub}

# %%
len_before_syntax = len(sub)
solutions_before_syntax = sum(len(p.solutions) for p in sub)

sub = [
    attrs.evolve(
        p, solutions=[s for s in p.solutions if is_correct_python_code(s) and is_indented_correct_python_code(s)]
    )
    for p in sub
]
sub = [p for p in sub if p.solutions]

len_after_syntax = len(sub)
solutions_after_syntax = sum(len(p.solutions) for p in sub)

print(f"Removed {solutions_before_syntax - solutions_after_syntax} solutions which did not have correct syntax.")
print(f"Removed {len_before_syntax - len_after_syntax} problems which did not have solutions with correct syntax.")

# %%


# %%


class CollectCallArgs(ast.NodeVisitor):
    target_fn_name: str
    all_args: list[tuple[ToEval, ...]]

    def __init__(self, target_fn_name: str) -> None:
        super().__init__()
        self.target_fn_name = target_fn_name
        self.all_args = []

    def visit_Call(self, node: ast.Call) -> Any:
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == self.target_fn_name:
            self.all_args.append(tuple(ToEval(format_str(ast.unparse(x), mode=FileMode()).strip()) for x in node.args))

        return super().generic_visit(node)


# %%


issue_items_all: defaultdict[int, list[ProblemIssueItem]] = defaultdict(list)

nb_out_of_context = 0
nb_assert_in_solution = 0
nb_not_right_amount_of_fences = 0

# fn, flawed_test_cases, checking_test_cases, problem_id, aux_info
all_responses: list[tuple[str, str, str, int, Any]] = []
for response in tqdm(all_code_for_issue_responses, desc="Processing responses"):
    model_response = response["response"]["choices"][0]["message"]["content"]
    problem_id = response["problem_id"]
    aux_info = response["aux_info"]

    out_of_context = response["response"]["choices"][0]["finish_reason"] != "stop"
    nb_out_of_context += out_of_context

    if ISSUE_GENERATION_METHOD == "base":
        model_response_lines = model_response.splitlines()

        def parse_out_python(header: str):
            start_line = model_response_lines.index(header)
            end_line = [i for i, s in enumerate(model_response_lines[start_line + 2 :]) if s.strip() == "```"][0]
            python_lines = model_response_lines[start_line + 2 : start_line + 2 + end_line]
            return "\n".join(python_lines)

        try:
            impl = format_str(parse_out_python("Incorrect implementation:"), mode=FileMode())
        except (ValueError, IndexError, BlackInvalidInput, TokenError):
            continue

        try:
            flawed_test_suite_str = parse_out_python("Flawed test suite:")
        except (ValueError, IndexError):
            flawed_test_suite_str = ""  # empty tests
        try:
            additional_tests_str = parse_out_python("Additional tests:")
        except (ValueError, IndexError):
            additional_tests_str = ""  # empty tests
        all_responses.append((impl, flawed_test_suite_str, additional_tests_str, problem_id, aux_info))
    elif ISSUE_GENERATION_METHOD == "extended":
        issue_splits = model_response.split("Step 2 - Way")[1:]
        if out_of_context:
            # the last one is probably broken, remove it
            issue_splits = issue_splits[:-1]

        # of the form
        issues_descs = re.findall(f"Way \d+: (.*?)\n", model_response, re.MULTILINE)

        for issue_text, desc in zip(issue_splits, issues_descs):
            get_code_in_blocks = re.findall(r"```python(.*?)```", issue_text, re.DOTALL)
            if len(get_code_in_blocks) != 3:
                nb_not_right_amount_of_fences += 1
                continue
            additional_tests_str = get_code_in_blocks[0].strip()
            flawed_test_suite_str = get_code_in_blocks[1].strip()
            try:
                impl = format_str(get_code_in_blocks[2].strip(), mode=FileMode())
            except (ValueError, IndexError, BlackInvalidInput, TokenError):
                continue

            if "assert" in impl and "..." in impl:
                nb_assert_in_solution += 1
                continue

            aux_info = {"issue_desc": desc, "solution_range": (0, 1)}

            all_responses.append((impl, flawed_test_suite_str, additional_tests_str, problem_id, aux_info))
    else:
        raise ValueError(f"Unknown issue generation method: {ISSUE_GENERATION_METHOD}")

print(f"Out of context: {nb_out_of_context} / {len(all_code_for_issue_responses)}")
if ISSUE_GENERATION_METHOD == "extended":
    print(f"Not right amount of fences: {nb_not_right_amount_of_fences} / {len(all_code_for_issue_responses) * 3}")
    print(f"Assert in solution: {nb_assert_in_solution} / {len(all_code_for_issue_responses) * 3}")

nb_out_of_dict = sum(1 for r in all_responses if r[3] not in task_id_to_problem)
print(f"Total responses: {len(all_responses)}, out of dict: {nb_out_of_dict}")
all_responses = [r for r in all_responses if r[3] in task_id_to_problem]


for impl, flawed_test_suite_str, additional_tests_str, problem_id, aux_info in tqdm(
    all_responses, desc="Parsing tests"
):
    # NOTE: this special cases no method python test cases!
    test_cases = task_id_to_problem[problem_id].test_cases
    assert isinstance(test_cases, PythonTestCases)

    collect_flawed_test_calls = CollectCallArgs(test_cases.fn_name)
    try:
        collect_flawed_test_calls.visit(ast.parse(flawed_test_suite_str))
        all_flawed_test_inputs = collect_flawed_test_calls.all_args
    except SyntaxError:
        all_flawed_test_inputs = []

    collect_checking_test_calls = CollectCallArgs(test_cases.fn_name)
    try:
        collect_checking_test_calls.visit(ast.parse(additional_tests_str))
        all_checking_test_inputs = collect_checking_test_calls.all_args
    except SyntaxError:
        all_checking_test_inputs = []

    issue_items_all[problem_id].append(
        ProblemIssueItem(
            desc=aux_info["issue_desc"],
            solution_range=tuple(aux_info["solution_range"]),  # type: ignore
            generated_solution=impl,
            # NOTE: empty outputs for now
            flawed_test_cases=PythonTestCases(
                is_solution_class=False, fn_name=test_cases.fn_name, inputs=all_flawed_test_inputs, outputs=[]
            ),
            checking_test_cases=PythonTestCases(
                is_solution_class=False, fn_name=test_cases.fn_name, inputs=all_checking_test_inputs, outputs=[]
            ),
        )
    )
# %%


def clean(s: str) -> str:
    # NOTE: cursed, should be removed eventually TODO
    if s.startswith("python\n"):
        return s[len("python\n") :]
    return s


generated_answers: dict[int, list[ModelGeneratedSolution]] = {}
with open(f"{DATA_DIR}/raw_functions_v3_with_generated_answers.jsonl", "r") as f:
    for line in f:
        p = json_converter.loads(line, LoadedProblemWithGeneratedAnswers)
        generated_answers[p.problem.task_id] = [
            ModelGeneratedSolution(p.generation_params.model, p.generation_params.temperature, clean(s))
            for s in p.generated_answers
        ]

# %%


def remove_wrong_syntax_inputs(test_cases: TestCases) -> TestCases:
    if not isinstance(test_cases, PythonTestCases):
        return test_cases
    new_inputs = []
    for inp, arg_str in zip(test_cases.inputs, test_cases.arg_strings()):
        if is_correct_python_code("f" + arg_str):
            new_inputs.append(inp)
    return attrs.evolve(test_cases, inputs=new_inputs)


number_of_inputs = {"flawed": 0, "checking": 0}

problems_and_tests_f = []
problems_and_tests_c = []
j = 0
for k, issues in issue_items_all.items():
    for issue in issues:
        number_of_inputs["flawed"] += len(issue.flawed_test_cases.inputs)
        number_of_inputs["checking"] += len(issue.checking_test_cases.inputs)
        problems_and_tests_f.append((sub_dict[k], remove_wrong_syntax_inputs(issue.flawed_test_cases)))
        problems_and_tests_c.append((sub_dict[k], remove_wrong_syntax_inputs(issue.checking_test_cases)))
        j += 1
# %%
problems_and_tests_f_ = complete_test_cases(problems_and_tests_f)
# %%ll
problems_and_tests_c_ = complete_test_cases(problems_and_tests_c)
# %%

# %%
number_of_inputs_good_syntax = {"flawed": 0, "checking": 0}
number_of_outputs_produced = {"flawed": 0, "checking": 0}
test_cases_without_outputs = {"flawed": 0, "checking": 0}
number_of_syntax_err = 0

i = 0
for k in issue_items_all.keys():
    new_issues = []
    for issue in issue_items_all[k]:
        if not is_correct_python_code(issue.generated_solution):
            number_of_syntax_err += 1
        else:
            new_issues.append(
                attrs.evolve(
                    issue,
                    flawed_test_cases=problems_and_tests_f_[i][1],
                    checking_test_cases=problems_and_tests_c_[i][1],
                )
            )

            number_of_inputs_good_syntax["flawed"] += len(issue.flawed_test_cases.inputs)
            number_of_inputs_good_syntax["checking"] += len(issue.checking_test_cases.inputs)
            number_of_outputs_produced["flawed"] += len(problems_and_tests_f_[i][1].outputs)
            number_of_outputs_produced["checking"] += len(problems_and_tests_c_[i][1].outputs)
            test_cases_without_outputs["flawed"] += len(problems_and_tests_f_[i][1].outputs) == 0
            test_cases_without_outputs["checking"] += len(problems_and_tests_c_[i][1].outputs) == 0

        i += 1  # don't forget me (not continue nor break allowed)
    issue_items_all[k] = new_issues

assert i == len(problems_and_tests_f_) == len(problems_and_tests_c_) == j

n_issue_items_all = sum(len(l) for l in issue_items_all.values())

# Health stats:
for typ in ["flawed", "checking"]:
    print(f"Number of {typ} individual test cases:      {number_of_inputs[typ]}")
    print(f"Number of {typ} test cases good syntax:     {number_of_inputs[typ]}")
    print(f"Number of {typ} outputs produced:           {number_of_outputs_produced[typ]}")
    print(f"Number of {typ} test cases without outputs: {test_cases_without_outputs[typ]} / {n_issue_items_all}")
print(f"Flawed solutions with syntax errors: {number_of_syntax_err} / {n_issue_items_all + number_of_syntax_err}")

# %%


problems_extra: list[LoadedProblemExtra] = []

counts_number_of_generated_solutions: dict[str, list[int]] = {"before syntax filter": [], "after syntax filter": []}

for problem in sub:
    generated = generated_answers.get(problem.task_id, [])
    counts_number_of_generated_solutions["before syntax filter"].append(len(generated))
    generated = [g for g in generated if is_correct_python_code(g.solution)]
    counts_number_of_generated_solutions["after syntax filter"].append(len(generated))

    problems_extra.append(
        LoadedProblemExtra(
            loaded_problem=problem,
            issue_items=issue_items_all[problem.task_id],
            generic_generated_solutions=generated,
        )
    )

for p in problems_extra:
    check_problems_fnname(p)

for name, l in counts_number_of_generated_solutions.items():
    print(name)
    for k, v in sorted(Counter(l).items()):
        print(f"{v/len(l):.4%} problems with {k} generic generated solutions")

# %%


def test_case_valid(test):
    return len(test.inputs) == len(test.outputs)


def valid_data(loaded_problem_extra: LoadedProblemExtra):
    if not test_case_valid(loaded_problem_extra.loaded_problem.test_cases):
        return False
    if not loaded_problem_extra.loaded_problem.solutions:
        return False
    for issue in loaded_problem_extra.issue_items:
        if not test_case_valid(issue.flawed_test_cases):
            return False
        if not test_case_valid(issue.checking_test_cases):
            return False
    return True


print("Number of LoadedProblemExtra objects:                ", len(problems_extra))
problems_extra = [p for p in problems_extra if valid_data(p)]
print("Number of LoadedProblemExtra objects with valid data:", len(problems_extra))

with open(f"{DATA_DIR}/loaded_problem_extra_v2.jsonl", "w") as f:
    for p in problems_extra:
        f.write(json_converter.dumps(p, LoadedProblemExtra) + "\n")
# %%
