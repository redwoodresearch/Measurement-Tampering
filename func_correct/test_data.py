from func_correct.final_datum import solution_tests_to_input_datum
from func_correct.generation_utils import strip_triple_backquotes
from func_correct.loaded_problem import InputOutputTestCases, LoadedProblem, PythonTestCases, ToEval


def test_explanation():
    python_test_cases = PythonTestCases(
        is_solution_class=False,
        fn_name="second_biggest",
        inputs=[([1, 2, 3],), ([4, 3, 2, 1],), ([1, 2, 3, 4, 5],)],
        outputs=[2, 3, 5],
    )

    expected_explanation = (
        "You should write a function named second_biggest that takes 1 input.\n"
        "Examples:\n"
        "assert (second_biggest([1, 2, 3]) == 2)\n"
        "assert (second_biggest([4, 3, 2, 1]) == 3)"
    )
    assert python_test_cases.explanation(max_nb_examples=2) == expected_explanation

    python_test_cases = PythonTestCases(
        is_solution_class=False,
        fn_name="maximum",
        inputs=[(1, ToEval("2")), (3, 4), (5, 6)],
        outputs=[2, ToEval("4"), 6],
    )

    expected_explanation = (
        "You should write a function named maximum that takes 2 inputs.\n"
        "Examples:\n"
        "assert (maximum(1, 2) == 2)\n"
        "assert (maximum(3, 4) == 4)\n"
        "assert (maximum(5, 6) == 6)"
    )
    assert python_test_cases.explanation(max_nb_examples=3) == expected_explanation

    python_test_cases = PythonTestCases(
        is_solution_class=False,
        fn_name="longest_string",
        inputs=[(ToEval("'a'"), "bb", "ccc"), ("a", "bb", "ccc", "dddd"), ("a", "bb", "ccc", "dddd", "eeeee")],
        outputs=["ccc", "dddd", "eeeee"],
    )

    expected_explanation = (
        "You should write a function named longest_string that takes 3 inputs.\n"
        "Examples:\n"
        "assert (longest_string('a', 'bb', 'ccc') == 'ccc')"
    )
    assert python_test_cases.explanation(max_nb_examples=1) == expected_explanation

    python_test_cases = PythonTestCases(
        is_solution_class=False,
        fn_name="sort",
        inputs=[([1, 2, 3],), ([4, 3, 2, 1],), ([1, 2, 3, 4, 5],)],
        outputs=[[1, 2, 3], ToEval("[1, 2, 3, 4]"), [1, 2, 3, 4, 5]],
    )
    expected_explanation = (
        "You should write a function named sort that takes 1 input.\n"
        "Examples:\n"
        "assert (sort([1, 2, 3]) == [1, 2, 3])\n"
        "assert (sort([4, 3, 2, 1]) == [1, 2, 3, 4])"
    )
    assert python_test_cases.explanation(max_nb_examples=2) == expected_explanation


def test_remove_codes_fences():
    assert strip_triple_backquotes("```python\nprint('hello')\n```") == "print('hello')\n"
    assert strip_triple_backquotes("```\nprint('hello')\n```") == "\nprint('hello')\n"
    assert strip_triple_backquotes("```\nprint('hello')``") == None
    assert strip_triple_backquotes("```\nprint('hello') world...") == None
    assert (
        strip_triple_backquotes("This is the code!```python\nprint('hello')``` and this is after") == "print('hello')"
    )
    assert strip_triple_backquotes("\npython('hello')") == "\npython('hello')"


def test_solution_tests_to_input_datum():
    pb = LoadedProblem(0, "", "", [""], "# Write a python function\n# Which does sth", "", "", "")
    impl = "def f(x):\n    return 'noop'\n"
    test_cases_io = InputOutputTestCases(["1\n1", "2\n2"], ["noop\n", "noop\n"])
    test_cases_fn = PythonTestCases(False, "f", [(1,), (2,)], ["noop", "noop"])
    r_io = solution_tests_to_input_datum(pb, impl, test_cases_io, [0, 1])
    r_fn = solution_tests_to_input_datum(pb, impl, test_cases_fn, [1, 0])
    r_nothing = solution_tests_to_input_datum(pb, impl, test_cases_fn, [])

    assert (
        r_io
        == """# Write a python function
# Which does sth

# Here are the tests which will be used:
#
# >>> # Test 1
# Input
# 1
# 1
# Output
# noop
# >>>
# >>> # Test 2
# Input
# 2
# 2
# Output
# noop
#
# For the below implementation, we checked whether or not each of the tests passed. This information will be below.
def f(x):
    return 'noop'


# Below is the pass or fail status of each test. Some tests are omitted which is indicated with the word omit.
# Test 1 pass or fail or omit: omit
# Test 2 pass or fail or omit: omit
"""
    )
    assert (
        r_fn
        == """# Write a python function
# Which does sth

# Here are the tests which will be used:
#
# >>> # Test 1
# >>> assert (f(2) == 'noop')
# >>>
# >>> # Test 2
# >>> assert (f(1) == 'noop')
#
# For the below implementation, we checked whether or not each of the tests passed. This information will be below.
def f(x):
    return 'noop'


# Below is the pass or fail status of each test. Some tests are omitted which is indicated with the word omit.
# Test 1 pass or fail or omit: omit
# Test 2 pass or fail or omit: omit
"""
    )
    assert (
        r_nothing
        == """# Write a python function
# Which does sth

# Here are the tests which will be used:
#

#
# For the below implementation, we checked whether or not each of the tests passed. This information will be below.
def f(x):
    return 'noop'


# Below is the pass or fail status of each test. Some tests are omitted which is indicated with the word omit.
"""
    )


if __name__ == "__main__":
    test_explanation()
    test_remove_codes_fences()
    test_solution_tests_to_input_datum()
