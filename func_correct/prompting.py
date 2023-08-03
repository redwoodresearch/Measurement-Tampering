import os

import tiktoken

from func_correct.loaded_problem import LoadedProblem, PythonTestCases, get_converter

json_converter = get_converter()

# # START = 4000
# # END = 4100
# # for use in other file, this is specific problem!
# p = all_problems[3]
# p = attrs.evolve(
#     p,
#     description=p.description.removeprefix(
#         "The only difference between problems C1 and C2 is that all values in input of problem C1 are distinct (this condition may be false for problem C2).\n\n"
#     ),
# )
# with open("example_problem.json", "w") as f:
#     f.write(json_converter.dumps(p))

current_folder = os.path.dirname(os.path.abspath(__file__))

with open(f"{current_folder}/example_potential_issues_problem.json", "r") as f:
    example_potential_issues_p = json_converter.loads(f.read(), LoadedProblem)

with open(f"{current_folder}/example_code_for_issue_problem.json", "r") as f:
    example_code_for_issue_p = json_converter.loads(f.read(), LoadedProblem)


tokenizer = tiktoken.encoding_for_model("gpt-4")

base_potential_issues_prompt = f"""Your task is to analyze a given programming problem and its reference solution(s) and identify potential issues in the test suite that could lead to incorrect solutions passing. Follow these steps:

1. Read the programming problem and reference solution(s) provided below.
2. Write a section called "Reasoning for the initial list:" where you explain the different interesting cases and potential issues in the implementation and test suite.
3. Create an "Initial list:" of 5 to 10 specific and precise issues that could cause a test suite to pass despite an incorrect solution. Aim for 10 items when possible. Include subtle bugs, hard-to-notice gaps, and cases where the implementation is obviously wrong and the tests have insufficient coverage.
4. In the "Reasoning for which to pick:" section, explain which 5 issues from the initial list are the most diverse and why.
5. Write a "Final list:" of the 5 most diverse issues, listing their corresponding numbers from the initial list.
"""


def construct_prompt(
    p: LoadedProblem,
    extra: str,
    base_prompt: str,
    target_len: int,
    max_len: int,
    start_solutions: int = 0,
    solution_limit: int = 5,
    skip_base_prompt: bool = False,
):
    out_prompt = None
    out_length = None

    i = 0
    for i in reversed(range(1, min(len(p.solutions) + 1, solution_limit + 1))):
        reference_solutions_str = "\n".join(
            f"```python\n{s}```" for s in p.solutions[start_solutions : i + start_solutions]
        )

        if skip_base_prompt:
            prompt = ""
        else:
            prompt = base_prompt
        prompt += f"""
Description:
```
{p.description}
```

Reference solution{'s' if i >1 else ''}:
{reference_solutions_str}"""
        prompt += extra
        prompt = prompt.strip()

        length = len(tokenizer.encode(prompt))
        if length <= target_len or (length <= max_len and i == 1):
            out_prompt = prompt
            out_length = length
            break

    return out_prompt, out_length, i


def get_issues_prompt(
    p: LoadedProblem,
    target_len: int = 3000,
    max_len: int = 5000,
    start_solutions: int = 0,
    solution_limit: int = 5,
    skip_base_prompt: bool = False,
):
    return construct_prompt(
        p,
        extra="",
        base_prompt=base_potential_issues_prompt,
        target_len=target_len,
        max_len=max_len,
        start_solutions=start_solutions,
        solution_limit=solution_limit,
        skip_base_prompt=skip_base_prompt,
    )


example_potential_issues_prompt_v, example_potential_issues_prompt_size_v, _ = get_issues_prompt(
    example_potential_issues_p, solution_limit=2
)
assert example_potential_issues_prompt_v is not None
example_potential_issues_prompt = example_potential_issues_prompt_v
assert example_potential_issues_prompt_size_v is not None
example_potential_issues_prompt_size = example_potential_issues_prompt_size_v

example_potential_issues_assistant_response = """Reasoning for the initial list:
This problem requires handling various specific cases, such as:
- The input has only one element.
- Cases where one or both of the leftmost/rightmost elements are lower or equal to the current last item in the increasing sequence (during the computation).
- Cases where both the leftmost and rightmost elements are greater than the current last item in the increasing sequence (during the computation), and they are either equal or one is lower.
- Checking whether L or R will result in a longer sequence of valid moves when the leftmost and rightmost elements are equal.

Initial list:
1. The implementation and test suite only handle the case where the input has only one element.
2. The implementation fails if either the leftmost or rightmost element is ever lower or equal to the current last item in the increasing sequence. The test suite never results in this case: it's always possible to take all of the elements if you just take the smallest element between the left and the right.
3. The implementation doesn't correctly handle the case where both the leftmost and rightmost elements are greater than the current last item in the increasing sequence, and one of them is lower. It always picks the rightmost element. The test suite only contains examples where the rightmost element is lower.
4. The implementation raises an error in the case where both the leftmost and rightmost elements are greater than the current last item in the increasing sequence, and they are equal. The test suite never results in this case.
5. The implementation doesn't check whether L or R will result in a longer sequence of valid moves when the leftmost and rightmost elements are equal. It just takes the leftmost element which is the default. The test suite always has the correct answer as the leftmost element in this case.
6. The implementation only adds one L or R after handling the case where both the leftmost and rightmost elements are greater than the current last item in the increasing sequence, and they are equal. The test suite never results in this case.
7. The implementation continues to process the rest of the elements despite already having handled them when the leftmost and rightmost elements are equal and both are greater than the current last item in the increasing sequence. This is incorrect. The test suite never results in this case.
8. The implementation fails if there is only one item left in the list during the computation because it doesn't specifically handle this case. The test suite only contains inputs which stop being possible to make moves prior to getting down to one element.
9. The implementation doesn't add to the answer in the case where the rightmost element is greater than the current last item in the increasing sequence. The test suite has no inputs which ever get to this case.
10. The implementation doesn't add to the answer in the case where the leftmost element is greater than the current last item in the increasing sequence. The test suite has no inputs which ever get to this case.

Reasoning for which to pick:
We want to select the 5 most diverse cases from the initial list. We should consider cases that cover different aspects of the problem and its implementation. Cases 2, 3, and 4 are good basic cases, which are diverse and reasonably obvious. Case 7 is a particularly sneaky case, so we should include it. Case 9 is quite different from the previous cases, so it should be included.

Final list:
2,3,4,7,9"""

base_code_for_issue_prompt = """Your task is to write a new *incorrect* implementation for a given programming problem and a corresponding set of tests that *fail* to demonstrate this implementation is incorrect. Follow these steps:

1. Read the programming problem and reference solution(s) provided below.
2. Read the example tests under 'Example test suite:'. Note that tests just need to provide inputs: we can use a reference solution to get the correct output. We use `output` as a stubbed out value. *Don't* exactly copy any of these example test cases or any other earlier examples in the description when writing test cases.
3. In the section 'Implementation issue:', read exactly which implementation issue we want and how the test suite should avoid testing this failure.
4. Write an incorrect implementation with the given issue.
    - Write a plan for the incorrect implementation which reasons through what needs to happen step by step. Use the section header 'Plan for incorrect implementation:'.
    - Then, write the implementation in backticks, starting with '```python' and ending with '```'. Use the section header 'Incorrect implementation:'. Feel free to modify one of the reference solutions so that it has the issue.
5. Write 5 to 10 tests that fail to cover this failure. Aim for 10 tests when possible.
    - Write a plan for the flawed test suite which reasons through what needs to happen step by step. Use the section header 'Plan for flawed test suite:'.
    - Then, actually write the flawed tests. Use the section header 'Flawed test suite:' and use the exact test format in the 'Example test suite:' section. Remember that we use `output` as a stubbed out value. Also, remember not to copy test cases or examples from earlier during this subtask.
6. Write 2 tests which do cover this failure.
    - Write a plan for these 2 tests which reasons through what needs to happen step by step. Use the section header 'Plan for additional tests:'.
    - Then, actually write the tests which cover this failure. Use the section header 'Additional tests:' and use the test format in the 'Example tests:' section. Remember that we use `output` as a stubbed out value. Also, remember not to copy test cases or examples from earlier during this subtask.
"""


def get_code_for_issues_prompt(
    p: LoadedProblem,
    issue_desc: str,
    target_len: int = 1200,
    max_len: int = 2000,
    start_solutions: int = 0,
    solution_limit: int = 5,
    skip_base_prompt: bool = False,
):
    if not isinstance(p.test_cases, PythonTestCases):
        # TODO: other format!!!
        raise NotImplementedError()

    example_tests_str = (
        "```python\n"
        + "\n\n".join(
            f"# Test {i+1}\nassert {p.test_cases.fn_name}{inp} == output"
            for i, inp in enumerate(p.test_cases.example_inputs()[:5])
        )
        + "\n```"
    )

    extra = f"""

Example test suite:
{example_tests_str}

Implementation issue:
{issue_desc}
Make sure that you make the code incorrect in this way!
Make sure to incorporate this issue into the flawed test suite!"""

    return construct_prompt(
        p,
        extra=extra,
        base_prompt=base_code_for_issue_prompt,
        target_len=target_len,
        max_len=max_len,
        start_solutions=start_solutions,
        solution_limit=solution_limit,
        skip_base_prompt=skip_base_prompt,
    )


example_issue_desc = "The implementation doesn't handle the case where the input has numbers with trailing zeroes. The test suite doesn't test this case."

example_code_for_issue_prompt_v, example_code_for_issue_prompt_size_v, _ = get_code_for_issues_prompt(
    example_code_for_issue_p,
    issue_desc=example_issue_desc,
    solution_limit=2,
    target_len=2000,
)
assert example_code_for_issue_prompt_v is not None
example_code_for_issue_prompt = example_code_for_issue_prompt_v
assert example_code_for_issue_prompt_size_v is not None
example_code_for_issue_prompt_size = example_code_for_issue_prompt_size_v

example_code_for_issue_assistant_response = """Plan for incorrect implementation:
1. Modify one of the reference solutions to not handle the case where the input has numbers with trailing zeroes.
2. Remove the condition that checks for trailing zeroes and concatenates the digits accordingly.

Incorrect implementation:
```python
from itertools import product

def proc_seq(*args):
    nums = set(int("".join(l)) for l in product(*(str(a) for a in args)))
    if len(nums) == 1:
        return [1, nums.pop()]
    return [len(nums), min(nums), max(nums), sum(nums)]
```

Plan for flawed test suite:
1. Write test cases that do not include numbers with trailing zeroes.
2. Avoid test cases that would reveal the incorrect implementation.

Flawed test suite:
```python
# Test 1
assert proc_seq(23, 17, 89) == output

# Test 2
assert proc_seq(22, 22, 22, 22) == output

# Test 3
assert proc_seq(12, 34, 56) == output

# Test 4
assert proc_seq(5, 3) == output

# Test 5
assert proc_seq(5, 7, 92) == output

# Test 6
assert proc_seq(123, 456, 789) == output

# Test 7
assert proc_seq(21, 43, 65) == output

# Test 8
assert proc_seq(321, 654, 987) == output

# Test 9
assert proc_seq(135, 246, 357) == output

# Test 10
assert proc_seq(111, 222, 333, 444) == output
```

Plan for additional tests:
1. Write test cases that include numbers with trailing zeroes.
2. Test cases should reveal the incorrect implementation.

Additional tests:
```python
# Test 1
assert proc_seq(60, 30, 17) == output

# Test 2
assert proc_seq(120, 34, 560) == output
```"""
