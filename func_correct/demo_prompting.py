# %%

import os

from func_correct.loaded_problem import DATA_DIR, LoadedProblem, get_converter
from func_correct.prompting import (
    example_code_for_issue_assistant_response,
    example_code_for_issue_prompt,
    example_potential_issues_assistant_response,
    example_potential_issues_prompt,
    get_code_for_issues_prompt,
    get_issues_prompt,
)

# %%

json_converter = get_converter()

# %%


with open(f"{DATA_DIR}/raw_functions_v3.jsonl", "r") as f:
    all_problems = [json_converter.loads(l, LoadedProblem) for l in f]

# %%

with open("out_example_issues_prompt.txt", "w") as f:
    f.write(example_potential_issues_prompt)
with open("out_example_issues_response.txt", "w") as f:
    f.write(example_potential_issues_assistant_response)

with open("out_example_code_for_issue_prompt.txt", "w") as f:
    f.write(example_code_for_issue_prompt)
with open("out_example_code_for_issue_response.txt", "w") as f:
    f.write(example_code_for_issue_assistant_response)


# %%

p = all_problems[-2410]

# %%


prompt, _, _ = get_issues_prompt(p, skip_base_prompt=True)
assert prompt is not None
with open("out_temp.txt", "w") as f:
    f.write(prompt)

# %%

issue_desc = "The implementation doesn't specifically handle the case where the input has a value of n that results in a Fibonacci number with a digit that appears more than once. So, it fails to find the maximum count of each digit. The implementation just selects the first digit because this is the default elsewhere in the implementation. In the test suite, it's always correct to select the first digit in this case."

# %%


prompt, _, _ = get_code_for_issues_prompt(p, issue_desc=issue_desc, skip_base_prompt=True)
assert prompt is not None
with open("out_temp.txt", "w") as f:
    f.write(prompt)

# %%

import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4")
len(tokenizer.encode(example_code_for_issue_assistant_response))
