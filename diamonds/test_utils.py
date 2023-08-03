import torch

from diamonds.utils import compute_after_intro_mask, get_output_and_gt


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


def test_compute_after_intro_mask():
    tensor = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 1, 2]])
    target_tokens = torch.tensor([1, 2, 3])
    expected_mask = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]], dtype=torch.bool)
    assert torch.equal(compute_after_intro_mask(tensor, target_tokens), expected_mask)

    tensor = torch.tensor([[1, 2, 3, 1, 2, 3], [0, 2, 3, 1, 2, 3]])
    target_tokens = torch.tensor([1, 2, 3])
    expected_mask = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.equal(compute_after_intro_mask(tensor, target_tokens), expected_mask)

    tensor = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 5, 4, 5]])
    target_tokens = torch.tensor([4, 5])
    expected_mask = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 1, 1]], dtype=torch.bool)
    assert torch.equal(compute_after_intro_mask(tensor, target_tokens), expected_mask)


if __name__ == "__main__":
    test_get_output_and_gt()
    test_compute_after_intro_mask()
