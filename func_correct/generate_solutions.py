import json
import sys
from copy import deepcopy

import attrs
from fire import Fire

from func_correct.evaluate_correctness import evaluate_correctness
from func_correct.generation_utils import GenerationParams, strip_triple_backquotes, threaded_generations
from func_correct.loaded_problem import AI_WATERMARK, DATA_DIR, LoadedProblem, get_converter


@attrs.frozen
class LoadedProblemWithGeneratedAnswers:
    problem: LoadedProblem
    generated_answers: list[str]
    generation_params: GenerationParams


INSTRUCTIONS = "Complete the code snippet provided by the user with a solution to the problem.\nOnly output code."
SOLUTION_INSTRUCTIONS = (
    "Complete the code snippet provided by the user with the correct solution to the problem.\nOnly output code."
)


def problem_to_prompt(problem: LoadedProblem, is_solution: bool, no_asserts: bool) -> list[dict[str, str]]:
    if no_asserts:
        explanation = problem.test_cases.explanation(max_nb_examples=0)
        commented_explanation = "\n".join("# " + l for l in explanation.splitlines())
        description = f"{problem.base_code}\n{commented_explanation}"
    else:
        description = problem.base_code_fmt_example

    instructions = SOLUTION_INSTRUCTIONS if is_solution else INSTRUCTIONS
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": description.replace(AI_WATERMARK, "")},
    ]


def run(
    which_functions: str = "raw_functions_v3",
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.9,
    n_threads: int = 100,
    nb_solutions: int = 1,
    add_as_solution: bool = False,
    check_solutions: bool = False,  # only works if add_as_solution is True
    no_asserts: bool = False,  # If True, only use test_cases for the function name
):
    for k, v in locals().items():
        print(f"{k}: {v}")

    problems: list[LoadedProblem] = []
    converter = get_converter()
    with open(f"{DATA_DIR}/{which_functions}.jsonl", "r") as f:
        for line in f:
            problems.append(converter.loads(line, LoadedProblem))

    params = GenerationParams(temperature=temperature, model=model)

    to_generate = [(problem, problem_to_prompt(problem, add_as_solution, no_asserts)) for problem in problems]
    print(f"Example prompt:\n{to_generate[0][1]}")

    generations = threaded_generations(
        to_generate, temperature=temperature, model=model, n_threads=n_threads, nb_solutions=nb_solutions
    )
    with open(f"{DATA_DIR}/{which_functions}_with_generated_answers.jsonl.tbak", "w") as f:
        f.write(converter.dumps(generations))
    generated_answers = sum(len(solutions) for _, solutions in generations)
    print(
        f"Generated {generated_answers} answers for {len(problems)} problems, expected {len(problems) * nb_solutions}"
    )

    if add_as_solution:
        problems = []
        for problem, solutions in generations:
            problem = deepcopy(problem)
            problem.solutions.extend(
                filter(lambda c: c is not None, [strip_triple_backquotes(solution) for solution in solutions])
            )
            problem.meta_data["solution_origin"] = f"generatedby_{model}_atT={temperature}"
            problems.append(problem)

        with open(f"{DATA_DIR}/{which_functions}_with_solutions.jsonl", "w") as f:
            for problem in problems:
                f.write(converter.dumps(problem) + "\n")

        if check_solutions:
            evaluation_results = evaluate_correctness(generations)
            nb_corrects = 0
            nb_failures = 0
            problems = []
            for pb_and_results in evaluation_results:
                for solution, results in pb_and_results.results_list:
                    if all(r.correct for r in results.results):
                        processed_solution = strip_triple_backquotes(solution)
                        if processed_solution is not None:
                            pb_and_results.problem.solutions.append(processed_solution)
                            nb_corrects += 1
                    if results.failure is not None:
                        nb_failures += 1
                pb_and_results.problem.meta_data["solution_origin"] = f"generatedby_{model}_atT={temperature}_checked"
                problems.append(pb_and_results.problem)

            print(f"Added {nb_corrects} solutions to {len(problems)} problems, {nb_failures} failures")

            with open(f"{DATA_DIR}/{which_functions}_solved.jsonl", "w") as f:
                for problem in problems:
                    f.write(converter.dumps(problem) + "\n")

    else:
        problems_with_generated_answers = [
            LoadedProblemWithGeneratedAnswers(
                problem,
                list(filter(lambda c: c is not None, [strip_triple_backquotes(solution) for solution in solutions])),
                params,
            )
            for problem, solutions in generations
        ]
        with open(f"{DATA_DIR}/{which_functions}_with_generated_answers.jsonl", "w") as f:
            for problem_with_generated_answers in problems_with_generated_answers:
                f.write(converter.dumps(problem_with_generated_answers) + "\n")


if __name__ == "__main__":
    Fire(run)

# Write a function that generates a list of all prime positive integers within agiven range (inclusive).
# If the range is invalid (i.e., empty, reversed, or contains non-integer or non-float values), return an empty list.
# You should write a function named list_of_primes that takes 2 inputs.
