import attrs

from func_correct.evaluate_correctness import (
    LoadedProblemAndResults,
    complete_test_cases,
    evaluate_correctness,
    run_batched_parallel,
)
from func_correct.exec_code import Result, Results, sandboxed_check_correctness
from func_correct.loaded_problem import InputOutputTestCases, LoadedProblem, PythonTestCases, ToEval

empty_problem = LoadedProblem(
    task_id=0,
    dataset="",
    description="",
    solutions=[],
    base_code="",
    base_code_fmt_example="",
    difficulty="",
    test_cases=InputOutputTestCases(inputs=[], outputs=[]),
)


def test_run_batches():
    inputs = list(range(10000))

    square_list = lambda x: [y**2 for y in x]

    squared = square_list(inputs)
    for n_threads in [1, 10, 100]:
        squared_batch = run_batched_parallel(square_list, inputs, n_threads=n_threads)
        squared_batch2 = run_batched_parallel(square_list, inputs, 100, n_threads=n_threads)
        squared_batch3 = run_batched_parallel(square_list, inputs, 1000, n_threads=n_threads)
        assert squared == squared_batch
        assert squared == squared_batch2
        assert squared == squared_batch3


def test_simple_execution():
    programs = [
        "print('hello world')",
        "import os; os.system('ls')",
        "assert 1 == 2",
        "a=",
        "import math\ndef f():\n    print(math.pi)\nf()",
    ]
    expected_results_list = [
        Results([], failure=None),
        Results([], failure="Other: <class 'TypeError'> 'NoneType' object is not callable"),
        Results([], failure="Other: <class 'AssertionError'> "),
        Results([], failure="SyntaxError"),
        Results([], failure=None),
    ]
    res = sandboxed_check_correctness(programs, [0] * 5, 0.1)
    assert res == expected_results_list


def test_empty_execution():
    res = sandboxed_check_correctness([], [], 0.1)
    assert res == []


def test_tests_execution():
    program = """
def maximum(a, b):
    return a if a > b else b

results____ = []

results____.append({"correct": maximum(1, 2) == 2})
results____.append({"correct": maximum(2, 1) == 2})
results____.append({"correct": maximum(1, 1) == 1})"""

    expected_results_list = [Results([Result(correct=True)] * 3, failure=None)]
    results_list = sandboxed_check_correctness([program], [3], 0.1)
    assert results_list == expected_results_list


def test_simple_correctness_io():
    pb = empty_problem
    answers = ["a=input()\nprint(a)", "print('whatever')", "assert False", "while True: pass", "a="]
    pb2 = attrs.evolve(empty_problem, test_cases=InputOutputTestCases(inputs=["12\n"], outputs=["12\n"]))
    answers2 = answers[:-1] + ["print(input())"]

    problems_and_answers = [(pb, answers), (pb2, answers2)]

    expected = [
        LoadedProblemAndResults(
            pb,
            [
                ("a=input()\nprint(a)", Results(results=[], failure=None)),
                ("print('whatever')", Results(results=[], failure=None)),
                ("assert False", Results(results=[], failure=None)),
                ("while True: pass", Results(results=[], failure=None)),
                ("a=", Results(results=[], failure="SyntaxError")),
            ],
        ),
        LoadedProblemAndResults(
            pb2,
            [
                ("a=input()\nprint(a)", Results(results=[Result(correct=True, reason=None)], failure=None)),
                ("print('whatever')", Results(results=[Result(correct=False, reason="wrong")], failure=None)),
                (
                    "assert False",
                    Results(results=[Result(correct=False, reason="ERROR <class 'AssertionError'> ")], failure=None),
                ),
                (
                    "while True: pass",
                    Results(
                        results=[Result(correct=False, reason="TIMEOUT")],
                        failure=None,
                    ),
                ),
                ("print(input())", Results(results=[Result(correct=True, reason=None)], failure=None)),
            ],
        ),
    ]
    r = evaluate_correctness(problems_and_answers)
    assert r == expected


def test_simple_correctness_fn():
    pb = attrs.evolve(empty_problem, test_cases=PythonTestCases(False, "maximum", [], []))
    answers = ["maximum = lambda a,b: max(a,b)", "print('whatever')", "assert False", "while True: pass"]
    pb2 = attrs.evolve(empty_problem, test_cases=PythonTestCases(False, "maximum", [(1, 2), (1, 1)], [2, 1]))
    answers2 = answers + [
        "def maximum(a, b):\n    return a if a > b else b",
        "def maximum(a, b):\n    return a if a < b else b",
        "def maximum(a, b):\n    while True: pass",
    ]

    problems_and_answers = [(pb, answers), (pb2, answers2)]

    expected = [
        LoadedProblemAndResults(
            problem=pb,
            results_list=[
                ("maximum = lambda a,b: max(a,b)", Results(results=[], failure=None)),
                ("print('whatever')", Results(results=[], failure=None)),
                ("assert False", Results(results=[], failure="Other: <class 'AssertionError'> ")),
                ("while True: pass", Results(results=[], failure="Other: <class 'TimeoutException____'> ")),
            ],
        ),
        LoadedProblemAndResults(
            problem=pb2,
            results_list=[
                (
                    "maximum = lambda a,b: max(a,b)",
                    Results(
                        results=[Result(correct=True, reason=None), Result(correct=True, reason=None)], failure=None
                    ),
                ),
                (
                    "print('whatever')",
                    Results(
                        results=[
                            Result(correct=False, reason="ERROR <class 'NameError'> name 'maximum' is not defined"),
                            Result(correct=False, reason="ERROR <class 'NameError'> name 'maximum' is not defined"),
                        ],
                        failure=None,
                    ),
                ),
                (
                    "assert False",
                    Results(
                        results=[Result(correct=False, reason="not ran"), Result(correct=False, reason="not ran")],
                        failure="Other: <class 'AssertionError'> ",
                    ),
                ),
                (
                    "while True: pass",
                    Results(
                        results=[Result(correct=False, reason="not ran"), Result(correct=False, reason="not ran")],
                        failure="Other: <class 'TimeoutException____'> ",
                    ),
                ),
                (
                    "def maximum(a, b):\n    return a if a > b else b",
                    Results(
                        results=[Result(correct=True, reason=None), Result(correct=True, reason=None)], failure=None
                    ),
                ),
                (
                    "def maximum(a, b):\n    return a if a < b else b",
                    Results(
                        results=[Result(correct=False, reason="wrong"), Result(correct=True, reason=None)], failure=None
                    ),
                ),
                (
                    "def maximum(a, b):\n    while True: pass",
                    Results(
                        results=[Result(correct=False, reason="TIMEOUT"), Result(correct=False, reason="TIMEOUT")],
                        failure=None,
                    ),
                ),
            ],
        ),
    ]
    r = evaluate_correctness(problems_and_answers)
    assert r == expected


def test_empty_correctness():
    assert evaluate_correctness([]) == []
    problems_and_answers = [
        (attrs.evolve(empty_problem, test_cases=PythonTestCases(False, "maximum", [(1, 2), (1, 1)], [2, 1])), []),
        (attrs.evolve(empty_problem, test_cases=InputOutputTestCases(inputs=["12\n"], outputs=["12\n"])), []),
    ]
    r = evaluate_correctness(problems_and_answers)
    assert r == [
        LoadedProblemAndResults(problems_and_answers[0][0], []),
        LoadedProblemAndResults(problems_and_answers[1][0], []),
    ]


def test_simple_complete_test_cases_io():
    problems_and_test_cases = [
        (
            attrs.evolve(
                empty_problem,
                solutions=["a=input()\nprint(int(a)**2)", "print((-int(input()))**2)"],
            ),
            InputOutputTestCases(["1", "2\n"], []),
        ),
        (
            attrs.evolve(
                empty_problem,
                solutions=["a=int(input())\nprint(a**2)", "a=int(input())\nprint(a)"],
            ),
            InputOutputTestCases(["1", "2\n"], []),
        ),
    ]
    expected = [
        (
            problems_and_test_cases[0][0],
            InputOutputTestCases(["1", "2\n"], ["1\n", "4\n"]),
        ),
        (problems_and_test_cases[1][0], InputOutputTestCases(["1"], ["1\n"])),
    ]
    r = complete_test_cases(problems_and_test_cases)
    assert r == expected

    problems_and_test_cases = [
        (attrs.evolve(empty_problem, solutions=["print('hello')"]), InputOutputTestCases([""], [])),
    ]
    expected = [
        (problems_and_test_cases[0][0], InputOutputTestCases([""], ["hello\n"])),
    ]
    r = complete_test_cases(problems_and_test_cases)
    assert r == expected


def test_simple_complete_test_cases_fn():
    problems_and_test_cases = [
        (
            attrs.evolve(
                empty_problem,
                solutions=["maximum = lambda a,b: max(a,b)", "def maximum(a, b):\n    return a if a > b else b"],
            ),
            PythonTestCases(False, "maximum", [(1, 2), (1, 1)], []),
        ),
        (
            attrs.evolve(
                empty_problem,
                solutions=["maximum = lambda a,b: max(a,b)", "def maximum(a, b):\n    return a if a < b else b"],
            ),
            PythonTestCases(False, "maximum", [(1, 2), (1, 1)], []),
        ),
    ]
    expected = [
        (
            problems_and_test_cases[0][0],
            PythonTestCases(False, "maximum", [(1, 2), (1, 1)], [ToEval("2"), ToEval("1")]),
        ),
        (problems_and_test_cases[1][0], PythonTestCases(False, "maximum", [(1, 1)], [ToEval("1")])),
    ]
    r = complete_test_cases(problems_and_test_cases)
    assert r == expected

    problems_and_test_cases = [
        (attrs.evolve(empty_problem, solutions=["hi = lambda:'hello'"]), PythonTestCases(False, "hi", [()], [])),
    ]
    expected = [
        (problems_and_test_cases[0][0], PythonTestCases(False, "hi", [()], [ToEval("'hello'")])),
    ]
    r = complete_test_cases(problems_and_test_cases)
    assert r == expected


def test_empty_complete_test_cases():
    assert complete_test_cases([]) == []

    problems_and_test_cases = [
        (attrs.evolve(empty_problem, solutions=[]), PythonTestCases(False, "maximum", [(1, 2), (1, 1)], [])),
    ]
    expected = [(problems_and_test_cases[0][0], PythonTestCases(False, "maximum", [], []))]
    assert complete_test_cases(problems_and_test_cases) == expected

    problems_and_test_cases = [
        (attrs.evolve(empty_problem, solutions=[]), InputOutputTestCases(["12\n", "13\n\n\n"], [])),
    ]
    expected = [(problems_and_test_cases[0][0], InputOutputTestCases([], []))]
    assert complete_test_cases(problems_and_test_cases) == expected


if __name__ == "__main__":
    test_run_batches()
    print("done")
    test_simple_execution()
    print("done")
    test_empty_execution()
    print("done")
    test_tests_execution()
    print("done")
    test_simple_correctness_io()
    print("done")
    test_simple_correctness_fn()
    print("done")
    test_empty_correctness()
    print("done")
    test_simple_complete_test_cases_io()
    print("done")
    test_simple_complete_test_cases_fn()
    print("done")
    test_empty_complete_test_cases()
    print("done")
