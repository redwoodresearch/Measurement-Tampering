from diamonds.utils import get_output_and_gt


def test_get_output_and_gt():
    code = "a = 1\nb = 1\nprint(a+b)\nprint(a)\nprint(b)"
    expected_stdout = "2\n1\n1\n"
    gt = lambda vars: vars["a"] == 1 and vars["b"] == 1
    output, gt = get_output_and_gt(code, gt)
    assert output == expected_stdout
    assert gt

    code = "a = 0\nb = 1\nprint(a+b)\nprint(a)\nprint(b)"
    expected_stdout = "1\n0\n1\n"
    gt = lambda vars: vars["a"] == 1 and vars["b"] == 0
    output, gt = get_output_and_gt(code, gt)
    assert output == expected_stdout
    assert not gt


if __name__ == "__main__":
    test_get_output_and_gt()
