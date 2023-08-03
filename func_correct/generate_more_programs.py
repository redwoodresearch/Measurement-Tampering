# %%
import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
from attr import frozen
from matplotlib import pyplot as plt
from matplotlib.artist import get
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange

from func_correct.final_datum import solution_tests_to_input_datum
from func_correct.generation_utils import threaded_generations
from func_correct.loaded_problem import (
    AI_WATERMARK,
    DATA_DIR,
    LoadedProblem,
    PythonTestCases,
    ToEval,
    get_converter,
)
from func_correct.prompting import tokenizer

converter = get_converter()

actually_generate_ratings = True
actually_generate_round_1 = True
actually_run_round_1_xp = True
actually_generate_round_2 = True
actually_generate_filtered_closure = True
actually_generate_extensions = True
actually_generate_test_cases = True

# %%
problems: list[LoadedProblem] = []
with open(f"{DATA_DIR}/raw_functions_v3.jsonl", "r") as f:
    for line in f:
        problems.append(converter.loads(line, LoadedProblem))
# %%
mbpp_problems = [p for p in problems if p.dataset.startswith("mbpp")]
for i in range(10):
    print(mbpp_problems[i].description)
# %%
# Rated by 'how easy and how much is it the kind of thing I want'
# I want mistakes which only affect some test cases, not all. Math is not good.
prompt_ratings = [
    ("Write a function to sort a given matrix in ascending order according to the sum of its rows.", "2"),
    ("Write a function to check if the given number is woodball or not.", "5 (outside knowledge)"),
    ("Write a function to count the most common words in a dictionary.", "4 (unclear)"),
    ("Write a python function to find the volume of a triangular prism.", "5 (outside knowledge)"),
    ("Write a function to split a string at lowercase letters.", "3"),
    (
        "Write a function to find sequences of lowercase letters joined with an underscore.",
        "4 (complicated due to edge cases)",
    ),
    (
        "Write a python function to remove first and last occurrence of a given character from the string.",
        "2 (nice, many ways to do it)",
    ),
    ("Write a function to find the perimeter of a square.", "4 (outside knowledge)"),
    ("Write a function to remove characters from the first string which are present in the second string.", "3"),
    ("Write a function to find whether a given array of integers contains any duplicate element.", "3"),
]


# %%
def get_prompt(problem_descriptions: list[str]):
    lines = [
        "Rate the difficulty of the following python programming problems from 1 to 5.",
        "1 is the easiest and 5 is the hardest.",
        "Hard problem involve outside knowledge, or have ambiguous instructions.",
        "Easy problems are very clear, have multiple ways of being solved, and don't require outside knowledge.",
    ]
    prompt_rating_texts = [t for t, _ in prompt_ratings]
    prompt_rating_ratings = [r for _, r in prompt_ratings]
    for i, prompt in enumerate(prompt_rating_texts + problem_descriptions):
        lines.append(f"{i+1}. {prompt}")
    for i, rating in enumerate(prompt_rating_ratings):
        lines.append(f"{i+1}. {rating}")

    prompt = [{"role": "user", "content": "\n".join(lines)}]

    return prompt


all_prompts = []
n_problems = len(mbpp_problems)
nb_problem_per_prompt = 40


# %%
if not actually_generate_ratings:
    with open(f"{DATA_DIR}/mbpp_ratings.json", "r") as f:
        ratings_tuples = converter.loads(f.read(), list[tuple[LoadedProblem, int]])
else:
    for i in range(0, n_problems, nb_problem_per_prompt):
        problems = mbpp_problems[i : i + nb_problem_per_prompt]
        desc = [p.description for p in problems]
        all_prompts.append((problems, get_prompt(desc)))
    answers = threaded_generations(all_prompts, nb_solutions=1, temperature=0, model="gpt-3.5-turbo", n_threads=30)
    ratings_tuples = []
    for problems, completions in answers:
        try:
            assert len(completions) == 1
            completion = completions[0]
            ratings = [int(r.split(". ")[1][0]) for r in completion.split("\n")]
            assert len(ratings) == nb_problem_per_prompt
            for problem, rating in zip(problems, ratings):
                ratings_tuples.append((problem, rating))
        except Exception as e:
            print(e)
    with open(f"{DATA_DIR}/mbpp_ratings.json", "w") as f:
        f.write(converter.dumps(ratings_tuples))
# %%
random_tuples = random.sample(ratings_tuples, 20)
for problem, rating in random_tuples:
    print(rating, problem.description)

easy_mbpp_problems = [p for p, r in ratings_tuples if r <= 2]
print("Number of easy problems:", len(easy_mbpp_problems), "out of", len(mbpp_problems))
# %%


def get_cosine_similarities(strings):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(strings)
    cosine_similarities = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(cosine_similarities, 0)  # Set the diagonal to 0 as they are self-similarities
    return cosine_similarities


def neirest_neighbor_similarities(strings):
    return get_cosine_similarities(strings).max(axis=1)


mbpp_problems_descs = [p.description for p in mbpp_problems]
if actually_run_round_1_xp:
    plt.hist(neirest_neighbor_similarities(mbpp_problems_descs), bins=100, range=(0, 1), density=True, alpha=0.5)
    plt.hist(neirest_neighbor_similarities(mbpp_problems_descs * 2), bins=100, range=(0, 1), density=True, alpha=0.5)
# %%
from collections import Counter, defaultdict

print(Counter([p.test_cases.n for p in easy_mbpp_problems]))
print(Counter([len(p.description.splitlines()) for p in easy_mbpp_problems]))
# %%

import collections
import functools
import itertools

ideas = [
    "functions with many arguments",
    "functions with optional arguments",
    "recursive functions",
    "functions with loops",
    "index manipulation",
    "string manipulation",
    "list manipulation",
    "dictionary manipulation",
    "functions with many if statements",
    "switch statements",
    "functions with many return statements",
    "nested functions",
    "nested loops",
    "functions with many local variables",
]

# get all functions from modules

for module in [collections, functools, itertools]:
    ideas += [f"functions using {x}" for x in dir(module) if callable(getattr(module, x)) if not x.startswith("_")]

INSTRUCTIONS = """You are a creative and accurate problem and program generator. Complete the users prompts by respecting the format they use. The problems you generate are very diverse and original."""


def comment(lines):
    return ["# " + line for line in lines]


def get_completion_prompt(problems: list[LoadedProblem], ideas: tuple[str, str], additional_lines: list[str] = []):
    idea1, idea2 = ideas
    lines = [
        "20 problems, solutions, and test cases.",
        f"Some are about {idea1}, some are about {idea2}, some are about both.",
        "The answer to each problem is a function which is then tested in 3 test cases.",
        "You can use Python's standard library, and you can introduce new functions.",
        "The assert statements should only invoke the function described by the problem.",
        *additional_lines,
    ]
    for i, problem in enumerate(problems):
        lines.append(f"* Problem {i+1}:")
        lines.append("# Description (1 to 5 lines):")
        lines += comment(problem.description.replace(AI_WATERMARK, "").splitlines())
        lines.append("# Solution:")
        lines += problem.solutions[0].splitlines()
        lines.append("# 3 test cases:")
        assert isinstance(problem.test_cases, PythonTestCases)
        for assert_statement in problem.test_cases.get_assertions():
            lines.append(assert_statement)

    return [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": "\n".join(lines)}]


prompt = get_completion_prompt(easy_mbpp_problems[:3], random.sample(ideas, 2))
for m in prompt:
    print(m["role"])
    print(m["content"])
    print(len(tokenizer.encode(m["content"])))
# %%

if actually_generate_round_1:
    answers = threaded_generations(
        [(None, prompt)], nb_solutions=1, temperature=1, model="gpt-3.5-turbo", n_threads=30, only_accept_finished=False
    )

    print(len(tokenizer.encode(answers[0][1][0])))
    print(answers[0][1][0])
import ast

# %%
import re

from black import FileMode, format_str


def fmt(s):
    return format_str(s, mode=FileMode()).strip()


class CollectCallArgs(ast.NodeVisitor):
    all_args: Optional[tuple[ToEval, ...]]

    def __init__(self) -> None:
        super().__init__()
        self.all_args = None
        self.fn_name = None

    def visit_Call(self, node: ast.Call) -> Any:
        fn = node.func
        if isinstance(fn, ast.Name):
            self.all_args = tuple(ToEval(fmt(ast.unparse(x))) for x in node.args)
            self.fn_name = fn.id
            return

        return super().generic_visit(node)


def input_fn_to_args(input_fn_str):
    visitor = CollectCallArgs()
    visitor.visit(ast.parse(input_fn_str))
    assert visitor.all_args is not None
    assert visitor.fn_name is not None
    return visitor.all_args, visitor.fn_name


assert input_fn_to_args("f(1, 2)") == ((ToEval("1"), ToEval("2")), "f")
assert input_fn_to_args("garb()") == ((), "garb")
assert input_fn_to_args("f(1)")[0] == (ToEval("1"),)
assert input_fn_to_args("f((1, 2, 3), 'hello', (25), {3})")[0] == (
    ToEval("(1, 2, 3)"),
    ToEval('"hello"'),
    ToEval("25"),
    ToEval("{3}"),
)


def extract_problems(text: str, id: int) -> list[LoadedProblem]:
    problems = text.split("* Problem")[1:]
    extracted_problems = []

    for problem in problems:
        try:
            description = re.search(r"# Description \(1 to 5 lines\):\n(.+)\n# Solution:", problem, re.DOTALL).group(1)
            solution = fmt(re.search(r"# Solution:\n(.+)\n# 3 test cases", problem, re.DOTALL).group(1))
            test_cases_raw = re.findall(r"assert \((.+)\)", problem) + re.findall(r"assert\((.+)\)", problem)

            description = "\n".join(
                l.replace("# ", "") for l in description.splitlines()
            )  # Remove "# " at the beginning of each line

            fn_names = set()

            inputs = []
            outputs = []

            for test_case in test_cases_raw:
                input_fn = re.search(r"(.+) == (.+)", test_case).group(1)
                output = ToEval(fmt(re.search(r"== (.+)", test_case).group(1)))
                inputs_args, fn_name = input_fn_to_args(input_fn)
                inputs.append(inputs_args)
                outputs.append(output)
                fn_names.add(fn_name)

            assert len(fn_names) == 1, f"Multiple function names in test cases {fn_names}"
            assert len(description.splitlines()) >= 1, "No line in description"
            assert len(test_cases_raw) == 3, "Not 3 test cases"

            # extracted_problems.append(
            #     {"description": description, "solution": solution, "test_cases": test_cases, "fn_name": fn_names.pop()}
            # )
            description = AI_WATERMARK + description
            base_code = f"# {description}"
            test_cases = PythonTestCases(is_solution_class=False, fn_name=fn_name, inputs=inputs, outputs=outputs)
            explanation = test_cases.explanation(max_nb_examples=2)
            commented_explanation = "\n".join("# " + l for l in explanation.splitlines())
            base_code_fmt_example = f"{base_code}\n{commented_explanation}"
            extracted_problems.append(
                LoadedProblem(
                    id,
                    "gen/mbpp_easy/v1",
                    description,
                    [solution],
                    base_code,
                    base_code_fmt_example,
                    "mbpp_easy",
                    test_cases,
                )
            )
        except Exception as e:
            # print("Error while extracting problem:", str(e))
            pass
    return extracted_problems


# %%
if actually_generate_round_1:
    r = extract_problems(answers[0][1][0], 0)
    print(r[3].base_code_fmt_example)
# %%
threads = 50
completion_per_thread = 3
n_rounds = 0
start_run_id = 1_000_000


@frozen
class LoadedProblemInspired:
    problem: LoadedProblem
    inspiration_ds: list[int]
    ideas: tuple[str, str]


# %%
new_problems = []
if actually_generate_round_1:
    new_problems = []
    with open(f"{DATA_DIR}/generated_mbpp_easy.jsonl", "w") as f:
        for i_round in range(n_rounds):
            prompts = []
            for i in range(threads * completion_per_thread):
                problems = random.sample(easy_mbpp_problems, 3)
                idea = random.sample(ideas, 2)
                p_ids = [p.task_id for p in problems]
                prompts.append(((p_ids, idea), get_completion_prompt(problems, idea)))
            print("Generating...")
            answers = threaded_generations(
                prompts,
                nb_solutions=1,
                temperature=1,
                model="gpt-3.5-turbo",
                n_threads=max(1, threads),
                only_accept_finished=False,
                max_attemps=1,
            )
            print("End generation")
            for (p_ids, idea), completions in answers:
                for completion in completions:
                    for problem in extract_problems(completion, len(new_problems) + start_run_id):
                        new_problems.append(problem)
                        to_write = LoadedProblemInspired(problem, p_ids, idea)
                        f.write(converter.dumps(to_write) + "\n")
            print(
                "got", len(new_problems), "new problems, expected", threads * completion_per_thread * (i_round + 1) * 17
            )
else:
    with open(f"{DATA_DIR}/generated_mbpp_easy.jsonl", "r") as f:
        for line in tqdm(f):
            problem = converter.loads(line, LoadedProblemInspired).problem
            new_problems.append(problem)
# %%
all_descriptions = [p.description for p in new_problems] + [p.description for p in easy_mbpp_problems]
similarities = get_cosine_similarities(all_descriptions)
# mask out (i,j) with i >= j: we don't want problem similar with higher id problems
similarities = np.triu(similarities)
max_higher_index_sim = similarities.max(axis=1)
which = np.argmax(similarities, axis=1)

filtered: list[LoadedProblem] = []
print_threshold = 0.7
threshold = 0.6
printed = 0

nb_filtered = []

for i in trange(len(new_problems) - 1, -1, -1):
    # for i in range(len(new_problems)):
    if max_higher_index_sim[i] < threshold:
        filtered.append(new_problems[i])
    elif printed < 10 and max_higher_index_sim[i] < print_threshold:
        print(max_higher_index_sim[i])
        print(new_problems[i].description)
        print(all_descriptions[which[i]])
        print()
        printed += 1
    nb_filtered.append(len(filtered))

print(len(filtered), "problems left")
# %%

from sklearn.linear_model import LinearRegression

if actually_run_round_1_xp:
    lr = LinearRegression(fit_intercept=False)

    f = np.square
    rf = np.sqrt

    lr.fit(np.arange(len(nb_filtered)).reshape(-1, 1), f(np.array(nb_filtered)))
    max_nb = 50_000
    print(rf(lr.predict([[max_nb]])))
    preds = lr.predict(np.arange(0, max_nb, 10).reshape(-1, 1))
    plt.scatter(np.arange(len(nb_filtered)), nb_filtered, marker=".", alpha=0.5)
    plt.plot(np.arange(0, max_nb, 10), rf(preds), color="black")
    plt.scatter([max_nb], rf(lr.predict([[max_nb]])), color="red")
# %%
if actually_run_round_1_xp:
    new_descriptions_without_info = [p.description.replace(AI_WATERMARK, "") for p in new_problems]
    plt.imshow(get_cosine_similarities(new_descriptions_without_info))
    plt.colorbar()
# %%
if actually_run_round_1_xp:
    plt.imshow(get_cosine_similarities(new_descriptions_without_info + [p.description for p in easy_mbpp_problems]))
    # line showing where the new problems start
    plt.axhline(len(new_problems), color="red")
    plt.axvline(len(new_problems), color="red")
    plt.colorbar()
# %%
if actually_run_round_1_xp:
    high = get_cosine_similarities(new_descriptions_without_info + [p.description for p in mbpp_problems]) > 0.5
    plt.imshow(high)
    # line showing where the new problems start
    plt.axhline(len(new_problems), color="red")
    plt.axvline(len(new_problems), color="red")

# %%
if actually_run_round_1_xp:
    plt.hist(
        neirest_neighbor_similarities([p.description for p in new_problems]),
        bins=100,
        range=(0, 1),
        density=True,
        alpha=0.5,
        label="new",
    )
    plt.hist(
        neirest_neighbor_similarities([p.description for p in mbpp_problems]),
        bins=100,
        range=(0, 1),
        density=True,
        alpha=0.5,
        label="mbpp",
    )
# %%
from tqdm import trange

if actually_run_round_1_xp:
    possibilities = np.arange(1_000_000)
    # density = np.ones_like(possibilities)
    density = 1 / ((possibilities * 0.1) ** 2 + 1)
    density /= density.sum()
    samples = set()
    nb_unique_samples = []
    max_n = 1_000
    for _ in trange(max_n):
        sample = np.random.choice(possibilities, p=density)
        samples.add(sample)
        nb_unique_samples.append(len(samples))
    plt.scatter(np.arange(len(nb_unique_samples)), nb_unique_samples, marker=".", alpha=0.5)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.arange(len(nb_unique_samples)).reshape(-1, 1), np.array(nb_unique_samples) ** 2)
    preds = np.sqrt(lr.predict(np.arange(0, max_n, 10).reshape(-1, 1)))
    plt.plot(np.arange(0, max_n, 10), preds, color="black")
# %%

# threads = 2
# completion_per_thread = 1
# n_rounds = 1
threads = 200
completion_per_thread = 10
n_rounds = 3
start_run_id = 2_000_000

additional_lines = [
    "The first 5 problems have a 1-line description, the next 5 have a 2-line description, etc.",
    "All problems are radically different from each other.",
]

new_problems_round_2 = []
if actually_generate_round_2:
    with open(f"{DATA_DIR}/generated_mbpp_easy_round_2.jsonl", "w") as f:
        for i_round in range(n_rounds):
            print(f"Round {i_round+1} / {n_rounds}")
            prompts = []
            for i in range(threads * completion_per_thread):
                problems = random.sample(filtered, 3)
                idea = random.sample(ideas, 2)
                p_ids = [p.task_id for p in problems]
                prompts.append(((p_ids, idea), get_completion_prompt(problems, idea, additional_lines)))
            print("Generating...")
            answers = threaded_generations(
                prompts,
                nb_solutions=1,
                temperature=1,
                model="gpt-3.5-turbo",
                n_threads=max(1, threads),
                only_accept_finished=False,
                max_attemps=1,
            )
            print("End generation")
            for (p_ids, idea), completions in answers:
                for completion in completions:
                    for problem in extract_problems(completion, len(new_problems_round_2) + start_run_id):
                        new_problems_round_2.append(problem)
                        to_write = LoadedProblemInspired(problem, p_ids, idea)
                        f.write(converter.dumps(to_write) + "\n")
            print("Sample:")
            print(new_problems_round_2[-1].description)
            print(
                "got",
                len(new_problems_round_2),
                "new problems, expected",
                threads * completion_per_thread * (i_round + 1) * 17,
            )
else:
    with open(f"{DATA_DIR}/generated_mbpp_easy_round_2.jsonl", "r") as f:
        for line in tqdm(f):
            problem = converter.loads(line, LoadedProblemInspired).problem
            new_problems_round_2.append(problem)
# %%
for p in new_problems_round_2[:30]:
    print(p.description)
    print()
# %%


def get_pb_category(pb):
    if pb.task_id >= 2_000_000:
        return "round_2"
    elif pb.task_id >= 1_000_000:
        return "round_1"
    else:
        return "mbpp"


filtered: list[LoadedProblem] = []
if not actually_generate_filtered_closure:
    with open(f"{DATA_DIR}/mbpp_easy_closure.jsonl", "r") as f:
        for line in tqdm(f):
            problem = converter.loads(line, LoadedProblem)
            filtered.append(problem)
else:
    all_problems: list[LoadedProblem] = new_problems_round_2 + new_problems + easy_mbpp_problems
    all_descriptions = [p.description.replace(AI_WATERMARK, "") for p in all_problems]
    similarities = get_cosine_similarities(all_descriptions)
    # mask out (i,j) with i >= j: we don't want problem similar with higher id problems
    similarities = np.triu(similarities)
    max_higher_index_sim = similarities.max(axis=1)
    which = np.argmax(similarities, axis=1)

    print_threshold = 0.7
    threshold = 0.6
    printed = 0

    nb_filtered = []

    for i, pb in enumerate(tqdm(all_problems)):
        # for i in range(len(new_problems_round_2)):
        if max_higher_index_sim[i] < threshold:
            filtered.append(pb)
        elif printed < 10 and max_higher_index_sim[i] < print_threshold:
            print(max_higher_index_sim[i])
            print(pb.description)
            print(all_descriptions[which[i]])
            print()
            printed += 1
        nb_filtered.append(len(filtered))

    with open(f"{DATA_DIR}/mbpp_easy_closure.jsonl", "w") as f:
        for pb in filtered:
            f.write(converter.dumps(pb) + "\n")

categories = [get_pb_category(pb) for pb in filtered]
for k, v in sorted(Counter(categories).items()):
    print(k, v)


# %%
for pb in filtered[:30]:
    print(pb.description.replace(AI_WATERMARK, ""))
    print()
# %%
INSTRUCTIONS_START = """
For each of the following programming problems, reformulate them into more precise and elaborate versions:
- Define the program's behavior more precisely.
- Allow it to accept a wider variety of inputs.
- Include special treatments for unique values.
- Clarify the program's actions in all edge cases.
Note: programs should never raise an exception.
Extended descriptions should range from 3 to 7 lines in length.
"""

examples_inputs = [
    """Write a function to count the number of words in a dictionary.""",
    """Write a function that takes a list of integers and returns the product of all the elements.""",
    """Write a function that takes a list of integers and returns the second highest number.""",
    """Write a function that receives two strings and returns True if the first string ends with the second string.""",
    """Write a function to receive a list of numbers and return the unique values of the list in order, preserving their first occurrences.""",
    """Write a function that takes a list of numbers and returns a list of the differences between each adjacent pair of numbers.""",
    """Write a function that receives a list of strings and returns the longest string in the list.""",
    """Write a function that receives a dictionary containing an item and its price, and returns the item with the highest price.""",
    """Write a function that takes a list of numbers and returns a list of the numbers that are divisible by 3 or 5.""",
    """Write a function that takes two integers and returns their sum as a string, except if the sum is 21 or more, in which case the function should return "BUST".""",
    """function that takes two dictionaries and combines them into one.""",
    """Write a function that receives two integers, and returns True if the first integer is divisible by the second integer.""",
    """Compute the value of factorial(n) for a given integer n.""",
]


example_outputs: list[tuple[str, str]] = [
    (
        """Write a function to count the number of words in the keys and values of a dictionary.
If keys or values are not strings, skip them. If a key or a value is a string, the number of words in it is the number of non-empty strings after you split by spaces and commas.""",
        """assert count_words({'hello': 'world', 'this': 'is a test'}) == 6
assert count_words({'hello, world': 123, 'this is': 'a,test', 3: 4}) == 6
assert count_words({1: 2, 3: 4, 5: 6}) == 0
assert count_words({"          ":",,,abc,,,"}) == 1
assert count_words({'hello world': 'this, is a test'}) == 6
""",
    ),
    (
        """Write a function that takes a list of integers, floats, or a combination of both and returns the product of all the elements. If the list is empty, return 1. Skip non-integer or non-float value, except for None, which behaves like 0.""",
        """assert (product_of_elements([1, 2, 3, 4]) == 24)
assert (product_of_elements([1, 2.5, 3, 0.5]) == 3.75)
assert (product_of_elements([1, 2, "a", 3, "b"]) == 6)
assert (product_of_elements([]) == 1)
assert (product_of_elements([1, 2, None, 3, -3]) == 0)
""",
    ),
    (
        """Write a function that takes a list of integers or floats and returns the second highest number.
If the list has fewer than 2 unique elements, return None.
Ignore non-integer or non-float values.""",
        """assert (second_highest([1, 2, 2, 3, 4]) == 2)
assert (second_highest([1, 2.5, 3, 0.5]) == 1)
assert (second_highest([1, 1]) == None)
assert (second_highest([1]) == None)
assert (second_highest([1, 1, "a", 3, "b", 2]) == 2)
""",
    ),
    (
        """Write a function that receives two strings and returns True if the first string ends with the second string.
If either of the two strings is empty, return False.""",
        """assert (string_ends_with("hello world", "world") == True)
assert (string_ends_with("hello world", "hello") == False)
assert (string_ends_with("hello world", "") == False)
assert (string_ends_with("", "world") == False)
assert (string_ends_with("", "") == False)
""",
    ),
    (
        """Write a function to receive a list of integers, floats, or a combination of both and return the unique values of the list in order, preserving their first occurrences.
Ignore non-integer or non-float values.""",
        """assert (unique_values([1, 2, 3, 2, 1]) == [1, 2, 3])
assert (unique_values([1, 2.5, 3, 2.5, 1]) == [1, 2.5, 3])
assert (unique_values([1, "a", 2, None, 1, "b"]) == [1, 2])
assert (unique_values([]) == [])
assert (unique_values([1, 1, 1, 1, 1]) == [1])
""",
    ),
    (
        """Write a function that takes a list of objects and returns a list of absolute values of the differences between each adjacent pair of numbers.
Non-integer and non-float values count as 0 (if "a" is next to 3, the difference is 3).
If the list is empty or has only one element, return an empty list.""",
        """assert (differences([1, 2, 3, 4]) == [1, 1, 1])
assert (differences([1, 2.5, 3, 0.5]) == [1.5, 0.5, 2.5])
assert (differences([1, 2, "a", 3, "b"]) == [1, 2, 3, 3])
assert (differences([]) == [])
assert (differences([1]) == [])
""",
    ),
    (
        """Write a function that receives a list of strings and returns the longest string in the list. If there is a tie between two or more strings, return the first one.
If the list is empty, return None.
Here, empty strings are the longest strings (but the length of other strings is unchanged).""",
        """assert (longest_string(["hello", "world", "this", "is", "a", "test"]) == "hello")
assert (longest_string(["this", "is", "a", "test", "hello", "world"]) == "hello")
assert (longest_string([]) == None)
assert (longest_string(["a", "bb", "ccc", "dd", "eeee"]) == "eeee")
assert (longest_string(["abc", ""]) == "")
""",
    ),
    (
        """Write a function that receives a dictionary containing an item (string) and its price (integer or float) and returns the item with the highest price.
If there is a tie between two or more items, return the first one. If the dictionary is empty, return None.
Negative prices are mistakes and should be ignored.""",
        """assert (highest_price_item({"banana": 1.5, "apple": 0.5, "orange": 1}) == "banana")
assert (highest_price_item({"banana": 1.5, "apple": 1.5, "orange": 1.5}) == "banana")
assert (highest_price_item({}) == None)
assert (highest_price_item({"apple": -1, "banana": -2, "orange": -3}) == None)
assert (highest_price_item({"apple": 1.5, "banana": 0.5, "pear": 3}) == "pear")
""",
    ),
    (
        """Write a function that takes a list of integers or floats and returns a list of the numbers that are divisible by 3 or 5, but not both.
Order should be preserved.
Floating point numbers should be floored (3.5 is divisible by 3).""",
        """assert (divisible_by_3_or_5([1, 2, 3, 4, 5, 6, 15]) == [3, 5, 6])
assert (divisible_by_3_or_5([1, 2.5, 3, 0.5]) == [3])
assert (divisible_by_3_or_5([]) == [])
assert (divisible_by_3_or_5([3, 3, 5, 10, 0, -3]) == [3, 3, 5, 10, -3])
assert (divisible_by_3_or_5([-2, -2.1, -3, -3.1, -14, -14.1]) == [-2.1, -3])
""",
    ),
    (
        """Write a function that takes two integers and returns their sum as a string, except if the sum is 21 or more, in which case the function should return "BUST".
If the value is exactly 21, return "BLACKJACK".""",
        """assert (sum_as_string(1, 2) == "3")
assert (sum_as_string(10, 11) == "BLACKJACK")
assert (sum_as_string(0, 0) == "0")
assert (sum_as_string(-10, -12) == "-22")
assert (sum_as_string(-10, 50) == "BUST")
""",
    ),
    (
        """Write a function that takes two dictionaries and combines them into one.
If the same key exists in both dictionaries, the value from the second dictionary should overwrite the value from the first dictionary, but if one of the two values is None, the other value should be used.""",
        """assert (combine_dicts({"a": 1, "b": 2}, {"c": 3, "d": 4}) == {"a": 1, "b": 2, "c": 3, "d": 4})
assert (combine_dicts({"a": 1, "b": 2}, {"a": 3, "d": 4}) == {"a": 3, "b": 2, "d": 4})
assert (combine_dicts({}, {"a": 1, "b": 2}) == {"a": 1, "b": 2})
assert (combine_dicts({"a": 1, "b": 2}, {"b": None}) == {"a": 1, "b": 2})
assert (combine_dicts({}, {}) == {})
""",
    ),
    (
        """Write a function that receives two integers, and returns True if the first integer is divisible by the second integer, or if the second integer is divisible by the first integer.
If one of the integers is zero, return "Cannot divide by zero".""",
        """assert (is_divisible(6, 3) == True)
assert (is_divisible(5, 2) == False)
assert (is_divisible(3, 6) == True)
assert (is_divisible(6, 0) == "Cannot divide by zero")
assert (is_divisible(-10, 5) == True)
""",
    ),
    (
        """Write a function that takes an integer n and returns the value of factorial(n).
If the input is negative, return factorial(-n).""",
        """assert (factorial(5) == 120)
assert (factorial(0) == 1)
assert (factorial(1) == 1)
assert (factorial(-1) == 1)
assert (factorial(-5) == 120)
""",
    ),
]

examples = list(zip(examples_inputs, example_outputs))


# %%
def get_variation_prompt(
    descriptions: list[str], number_of_random_inspirations: int = 6
) -> tuple[list[int], list[dict[str, str]]]:
    """Return inspirations and prompt for the variations of the given descriptions."""
    inspiration_ids = random.sample(range(len(examples)), number_of_random_inspirations)
    inspiration_examples = [examples[i] for i in inspiration_ids]

    lines = INSTRUCTIONS_START.splitlines()

    all_descriptions = [desc for desc, _ in inspiration_examples] + [
        desc.replace(AI_WATERMARK, "") for desc in descriptions
    ]
    for i, desc in enumerate(all_descriptions):
        lines.append(f"* Problem {i+1} description:")
        lines += desc.splitlines()
    for i, (_, (desc, _)) in enumerate(inspiration_examples):
        lines.append(f"* Problem {i+1} extended:")
        lines += desc.splitlines()

    return (
        inspiration_ids,
        [
            {"role": "system", "content": "You are a creative dataset generator"},
            {"role": "user", "content": "\n".join(lines)},
        ],
    )


p = get_variation_prompt([p.description for p in filtered[-15:]], 6)
print(p[1][1]["content"])
print(len(tokenizer.encode(p[1][1]["content"])))
# %%
if actually_generate_extensions:
    answers = threaded_generations(
        [p],
        nb_solutions=1,
        temperature=1,
        model="gpt-3.5-turbo",
        n_threads=30,
        only_accept_finished=False,
    )

    print(len(tokenizer.encode(answers[0][1][0])) + len(tokenizer.encode(p[1][1]["content"])))
    print(answers[0][1][0])
# %%


def extract_problems_answers(text: str) -> list[str]:
    lines = text.splitlines()
    answers = []
    current_answer = []
    for line in lines:
        if line.startswith("* Problem ") or ("Problem" in line and line.endswith(":")):
            if current_answer:
                answers.append("\n".join(current_answer))
                current_answer = []
        else:
            current_answer.append(line)
    answers.append("\n".join(current_answer))
    return answers


if actually_generate_extensions:
    extract_problems_answers(answers[0][1][0])

# %%

# threads = 2
# completion_per_thread = 1
# nb_pb_per_prompt = 20
# n_rounds = 1
# desc_to_extend = [p.description for p in filtered][:10]

threads = 200
completion_per_thread = 3
nb_pb_per_prompt = 25
n_repetition_per_desc = 20
desc_to_extend = [p.description for p in filtered] * n_repetition_per_desc

batch_size = threads * completion_per_thread * nb_pb_per_prompt
batches = [desc_to_extend[i : i + batch_size] for i in range(0, len(desc_to_extend), batch_size)]

desc_extended = []
if actually_generate_extensions:
    with open(f"{DATA_DIR}/desc_extended.jsonl", "w") as f:
        for i, batch in enumerate(batches):
            print(f"Extension Batch {i+1}/{len(batches)}")
            sub_batches = [batch[i : i + nb_pb_per_prompt] for i in range(0, len(batch), nb_pb_per_prompt)]
            prompts = [get_variation_prompt(sub_batch) for sub_batch in sub_batches]
            answers = threaded_generations(
                prompts,
                nb_solutions=1,
                temperature=1,
                model="gpt-3.5-turbo",
                n_threads=max(1, threads),
                only_accept_finished=True,
                max_attemps=1,
            )
            for inspirations, extensions in answers:
                for extension in extensions:
                    for desc in extract_problems_answers(extension):
                        if "raise" in desc:
                            continue
                        desc = AI_WATERMARK + desc
                        desc_extended.append(desc)
                        f.write(json.dumps({"description": desc, "inspirations": inspirations}) + "\n")
            print("Sample:")
            print(desc_extended[-1])
            print(
                "got",
                len(desc_extended),
                "new problems, expected",
                (i + 1) * batch_size,
            )
else:
    with open(f"{DATA_DIR}/desc_extended.jsonl", "r") as f:
        for line in tqdm(f):
            desc_extended.append(json.loads(line)["description"])


# %%
# Test case generation
def get_test_cases_prompt(
    descriptions: list[str], number_of_random_inspirations: int = 6
) -> tuple[list[int], list[dict[str, str]]]:
    """Return inspirations and test cases for the description."""
    inspiration_ids = random.sample(range(len(examples)), number_of_random_inspirations)
    inspiration_examples = [examples[i] for i in inspiration_ids]

    lines = [
        "Provide 5 test cases for each of the following programming problem.",
        "Cover the usual cases and the edge cases.",
        "Each assert statement should only invoke the function described by the problem, and be one line long.",
    ]

    all_descriptions = [desc for _, (desc, _) in inspiration_examples] + [
        desc.replace(AI_WATERMARK, "") for desc in descriptions
    ]
    for i, desc in enumerate(all_descriptions):
        lines.append(f"* Problem {i+1} description:")
        lines += desc.splitlines()
    for i, (_, (_, tests)) in enumerate(inspiration_examples):
        lines.append(f"* Problem {i+1} test cases:")
        lines += tests.splitlines()

    return (
        inspiration_ids,
        [
            {"role": "system", "content": "You are a creative dataset generator"},
            {"role": "user", "content": "\n".join(lines)},
        ],
    )


print(get_test_cases_prompt([p.description for p in easy_mbpp_problems[:20]], 6)[1][1]["content"])
# %%

threads = 200
completion_per_thread = 3
nb_pb_per_prompt = 12

batch_size = threads * completion_per_thread * nb_pb_per_prompt
batches = [desc_extended[i : i + batch_size] for i in range(0, len(desc_extended), batch_size)]

desc_extended_and_tests = []
if actually_generate_test_cases:
    with open(f"{DATA_DIR}/desc_extended_and_tests.jsonl", "w") as f:
        for i, batch in enumerate(batches):
            print(f"Test Batch {i+1}/{len(batches)}")
            sub_batches = [batch[i : i + nb_pb_per_prompt] for i in range(0, len(batch), nb_pb_per_prompt)]
            prompts = [get_test_cases_prompt(sub_batch) for sub_batch in sub_batches]
            prompts_with_data = [(descs, prompt) for descs, (_, prompt) in zip(sub_batches, prompts)]
            answers = threaded_generations(
                prompts_with_data,
                nb_solutions=1,
                temperature=0,
                model="gpt-3.5-turbo",
                n_threads=max(1, threads),
                only_accept_finished=True,
                max_attemps=1,
            )
            for descs, extensions in answers:
                for extension in extensions:
                    extracted_tests = extract_problems_answers(extension)
                    if len(extracted_tests) != len(descs):
                        continue

                    for desc, tests in zip(descs, extracted_tests):
                        r = {"description": desc, "tests": tests}
                        desc_extended_and_tests.append(r)
                        f.write(json.dumps(r) + "\n")
            print("Sample:")
            print(desc_extended_and_tests[-1])
            print(
                "got",
                len(desc_extended_and_tests),
                "new problems, expected",
                (i + 1) * batch_size,
            )
else:
    with open(f"{DATA_DIR}/desc_extended_and_tests.jsonl", "r") as f:
        for line in tqdm(f):
            desc_extended_and_tests.append(json.loads(line))
# %%
start_id = 3_000_000

filtered_first_lines = [p.description.split(".")[0] for p in filtered]
extended_first_lines = [p["description"].split(".")[0].replace(AI_WATERMARK, "") for p in desc_extended_and_tests]
first_lines = filtered_first_lines + extended_first_lines

vectorizer = TfidfVectorizer(stop_words="english")
vectorizer = vectorizer.fit(first_lines[::10])
# %%
# Similarity matrix between filtered and extended problems
similarities = cosine_similarity(vectorizer.transform(filtered_first_lines), vectorizer.transform(extended_first_lines))
print(similarities.shape)
# %%
closests = np.argmax(similarities, axis=0)
print(closests.shape)
# %%
for i in range(10):
    extended_desc = desc_extended_and_tests[i]["description"]
    parent = filtered[closests[i]]
    print("Extended:\n", extended_desc)
    print(f"Parent: {parent.task_id}\n{parent.description}")
    print()

# %%
number_of_childs = Counter(closests)
plt.hist(number_of_childs.values(), bins=max(number_of_childs.values()))
plt.axvline(n_repetition_per_desc)
# Good news, almost no weird shit happening.
# %%
for i, p in list(enumerate(filtered))[-5:]:
    print("Original")
    print(p.description)
    print("Derived")
    for idx in np.where(closests == i)[0]:
        print(desc_extended_and_tests[idx]["description"].strip())
    print("----------")
# %%

full_desc_vectorizer = TfidfVectorizer(stop_words="english")
full_desc_vectorizer = vectorizer.fit([x["description"] for x in desc_extended_and_tests[::10]])

threshold = 0.8
print_treshold = 0.9

pb_infos = []

for i, p in enumerate(tqdm(filtered)):
    pack = [desc_extended_and_tests[idx] for idx in np.where(closests == i)[0]]

    if not pack or len(pack) > n_repetition_per_desc:
        continue

    descriptions = [x["description"] for x in pack]
    embeddings = full_desc_vectorizer.transform(descriptions)
    similarities = cosine_similarity(embeddings, embeddings)
    similarities = np.triu(similarities)
    np.fill_diagonal(similarities, 0)
    max_higher_index_sim = similarities.max(axis=1)
    which = np.argmax(similarities, axis=1)

    non_redundants = []
    for j, desc_and_test in enumerate(pack):
        if max_higher_index_sim[j] < threshold:
            desc_and_test["parent"] = p.task_id
            non_redundants.append(desc_and_test)
        # else:
        elif max_higher_index_sim[j] < print_treshold and i < 10:
            print(desc_and_test["description"])
            print(f"-- to close to {max_higher_index_sim[j]} --")
            print(descriptions[which[j]])
            print("======")
    if i < 30:
        print(f"Got {len(non_redundants)} non redundant problems for {p.task_id} with {len(pack)} variations")
    pb_infos += non_redundants

print(len(pb_infos))

# # %%
# import attrs
# from func_correct.loaded_problem import TestCases


# @attrs.frozen
# class LoadedProblem:
#     task_id: int
#     dataset: str
#     description: str
#     solutions: list[str]
#     base_code: str
#     base_code_fmt_example: str
#     difficulty: str
#     test_cases: TestCases
#     meta_data: dict[str, str] = {}


# %%
new_extend_problems = []
for j, info in enumerate(tqdm(pb_infos)):
    try:
        description = info["description"].strip()
        tests_str = info["tests"]
        probable_parent = info["parent"]
        test_cases_raw = re.findall(r"assert \((.+)\)", tests_str) + re.findall(r"assert\((.+)\)", tests_str)

        fn_names = set()

        inputs = []
        outputs = []

        for test_case in test_cases_raw:
            input_fn = re.search(r"(.+) == (.+)", test_case).group(1)
            output = ToEval(fmt(re.search(r"== (.+)", test_case).group(1)))
            inputs_args, fn_name = input_fn_to_args(input_fn)
            inputs.append(inputs_args)
            outputs.append(output)
            fn_names.add(fn_name)

        assert len(test_cases_raw) >= 4, f"Not enough test cases, got {len(test_cases_raw)}"
        assert len(test_cases_raw) <= 10, f"Too many test cases, got {len(test_cases_raw)}"
        assert len(fn_names) == 1, f"Multiple function names in test cases {fn_names}\n{tests_str}"

        base_code = "\n".join(f"# {line}" for line in description.splitlines())
        test_cases = PythonTestCases(is_solution_class=False, fn_name=fn_name, inputs=inputs, outputs=outputs)
        explanation = test_cases.explanation(max_nb_examples=2)
        commented_explanation = "\n".join("# " + l for l in explanation.splitlines())
        base_code_fmt_example = f"{base_code}\n{commented_explanation}"
        new_extend_problems.append(
            LoadedProblem(
                start_id + j,
                "gen/mbpp_easy_extended/v1",
                description,
                [],
                base_code,
                base_code_fmt_example,
                "mbpp_easy_extended",
                test_cases,
                meta_data={"probable_parent_task_id": str(probable_parent)},
            )
        )
    except Exception as e:
        # print(e)
        pass
# %%
print(f"Got {len(new_extend_problems)} new problems, expected {len(pb_infos)}")
# %%
import attrs

# print all attrs of the first problem
for attr in new_extend_problems[0].__attrs_attrs__:
    print(attr.name)
    print(getattr(new_extend_problems[0], attr.name))
    print()
# %%
with open(f"{DATA_DIR}/extended_functions.jsonl", "w") as f:
    for pb in tqdm(new_extend_problems):
        f.write(converter.dumps(pb) + "\n")
# %%
