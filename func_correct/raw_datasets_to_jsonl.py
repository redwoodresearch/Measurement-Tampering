# mypy: ignore-errors
# %%
import ast
import json
import os

import attrs
import cattrs
from black import FileMode, InvalidInput, format_str
from datasets import load_dataset
from tqdm import tqdm

from func_correct.loaded_problem import (
    DATA_DIR,
    InputOutputTestCases,
    LoadedProblem,
    PythonTestCases,
    ToEval,
    get_converter,
)

# %%

json_converter = get_converter()


# %%


# %%


# %%

apps_data = load_dataset("codeparrot/apps")
mbpp_data = load_dataset("mbpp")
# human_eval_data = load_dataset("openai_humaneval")

# %%

# print(human_eval_data["test"][0]["prompt"])

# %%

START = -10000000
END = 10000000
STRIDE = 1

# max_format_num = 6
max_format_num = 100000

# START = 100
# END = 150

# %%
all_problems: list[LoadedProblem] = []
task_id = 0

# def to_lines(x: Any) -> list[str]

for t_t in ["test", "train"]:
    for datum_id, x in enumerate(tqdm(apps_data[t_t])):
        if datum_id < START or datum_id >= END or datum_id % STRIDE != 0:
            continue
        solutions: list[str] = []
        if x["solutions"] != "":
            for i, s in enumerate(json.loads(x["solutions"])):
                assert isinstance(s, str)
                # TODO: format all, not just first 6?
                if i < max_format_num:
                    try:
                        solutions.append(format_str(s, mode=FileMode()))
                    except (InvalidInput, IndentationError):
                        continue  # legacy continues without task_id incr
                else:
                    solutions.append(s)
        base_code = "\n".join("# " + l for l in x["question"].splitlines())

        if x["input_output"] != "":
            inp_out = json.loads(x["input_output"])
            if "fn_name" in inp_out:
                if any("class Solution" in s for s in solutions):
                    # cursed
                    continue  # legacy continues without task_id incr

                # undo cursed formatting
                if not all(isinstance(y, list) and len(y) == 1 for y in inp_out["outputs"]):
                    print("WARNING: weird output format (skipped)\n", inp_out["outputs"])
                    continue  # legacy continues without task_id incr
                outputs = [y[0] for y in inp_out["outputs"]]

                test_cases = PythonTestCases(
                    is_solution_class=False,
                    fn_name=inp_out["fn_name"],
                    inputs=[tuple(y) for y in inp_out["inputs"]],
                    outputs=outputs,
                )
            else:
                if len(inp_out["inputs"]) == len(inp_out["outputs"]) != 0:
                    if isinstance(inp_out["inputs"][0], list):
                        inp_out["inputs"] = ["\n".join(y) for y in inp_out["inputs"]]
                    if isinstance(inp_out["outputs"][0], list):
                        inp_out["outputs"] = ["\n".join(y) for y in inp_out["outputs"]]

                test_cases = InputOutputTestCases(
                    inputs=inp_out["inputs"],
                    outputs=inp_out["outputs"],
                )
        else:
            test_cases = InputOutputTestCases(inputs=[], outputs=[])

        all_problems.append(
            LoadedProblem(
                task_id,
                dataset=f"apps/{t_t}",
                description=x["question"],
                solutions=solutions,
                base_code=base_code,
                base_code_fmt_example=base_code,  # apps have really good explanations and examples
                test_cases=test_cases,
                difficulty=f"apps/{x['difficulty']}",
            )
        )
        task_id += 1
# %%
for t_t in ["test", "train", "validation", "prompt"]:
    for datum_id, x in enumerate(tqdm(mbpp_data[t_t])):
        if datum_id < START or datum_id >= END:
            continue
        fmt_sol_code = format_str(x["code"], mode=FileMode())
        text = x["text"]
        assert "\n" not in text
        p = "\n    "
        inputs = []
        outputs = []
        fn_name = None
        for y_str in x["test_list"]:
            y = ast.parse(y_str)
            assert isinstance(y, ast.Module)
            assertion = y.body[0]
            assert isinstance(assertion, ast.Assert)
            assert_test = assertion.test
            assert isinstance(assert_test, ast.Compare)
            l, r = assert_test.left, assert_test.comparators[0]
            r_str = ToEval(format_str(ast.unparse(r), mode=FileMode()).strip())
            assert isinstance(l, ast.Call)
            fn = l.func
            assert isinstance(fn, ast.Name)

            new_fn_name = fn.id
            assert fn_name is None or new_fn_name == fn_name
            fn_name = new_fn_name
            args_strings = tuple(ToEval(format_str(ast.unparse(arg), mode=FileMode()).strip()) for arg in l.args)
            inputs.append(args_strings)
            outputs.append(r_str)

        assert fn_name is not None
        base_code = f"# {text}"
        test_cases = PythonTestCases(is_solution_class=False, fn_name=fn_name, inputs=inputs, outputs=outputs)
        explanation = test_cases.explanation(max_nb_examples=2)
        commented_explanation = "\n".join("# " + l for l in explanation.splitlines())
        base_code_fmt_example = f"{base_code}\n{commented_explanation}"

        all_problems.append(
            LoadedProblem(
                task_id,
                dataset=f"mbpp/{t_t}",
                description=text,
                solutions=[fmt_sol_code],
                base_code=base_code,
                base_code_fmt_example=base_code_fmt_example,
                test_cases=test_cases,
                difficulty=f"mbpp",
            )
        )
        task_id += 1

# %%

with open(f"{DATA_DIR}/raw_functions_v3.jsonl", "w") as f:
    for p in all_problems:
        f.write(json_converter.dumps(p, LoadedProblem) + "\n")

# %%


# there are always examples
def has_example(x) -> bool:
    return (
        "example" in x["question"].lower()
        or "sample" in x["question"].lower()
        or "should return" in x["question"].lower()
    )


without_examples = len([e for e in apps_data["train"] if not has_example(e)])
print(f"percentage without examples {without_examples / len(apps_data['train']):.4%}")
# %%
