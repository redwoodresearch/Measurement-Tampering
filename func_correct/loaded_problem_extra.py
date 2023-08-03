import attrs

from func_correct.loaded_problem import InputOutputTestCases, LoadedProblem, PythonTestCases, TestCases


@attrs.frozen
class ModelGeneratedSolution:
    model: str
    temp: float
    solution: str


@attrs.frozen
class ProblemIssueItem:
    desc: str
    solution_range: tuple[int, int]
    generated_solution: str
    # note: below test cases don't yet necessarily contain outputs!
    flawed_test_cases: TestCases
    checking_test_cases: TestCases


@attrs.frozen
class LoadedProblemExtra:
    loaded_problem: LoadedProblem
    issue_items: list[ProblemIssueItem]
    generic_generated_solutions: list[ModelGeneratedSolution]
