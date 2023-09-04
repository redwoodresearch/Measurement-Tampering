import torch


def is_sensor_loc(input_ids: torch.Tensor, tok):
    questions_section_toks = tok.encode("## Questions")
    assert len(questions_section_toks) == 2
    eq_question_item = (input_ids[:, :-1] == questions_section_toks[0]) & (
        input_ids[:, 1:] == questions_section_toks[1]
    )
    assert (eq_question_item.sum(dim=-1, dtype=torch.int) == 1).all(), "could relax"

    question_mark_tok = tok.encode("?")
    assert len(question_mark_tok) == 1
    other_question_mark_tok = tok.encode(")?")
    assert len(other_question_mark_tok) == 1

    summed = torch.cumsum(
        torch.cat([eq_question_item, eq_question_item[:, -1:]], dim=-1), dim=-1
    )
    return (summed > 0) & (
        (input_ids == question_mark_tok[0]) | (input_ids == other_question_mark_tok[0])
    )


def get_sensor_locs(input_ids: torch.Tensor, tok):
    question_mark_locs = is_sensor_loc(input_ids, tok)
    total_locs = torch.cumsum(question_mark_locs, dim=-1)
    total_overall = total_locs[:, -1]
    assert (
        total_overall == 3
    ).all(), "can handle different cases, but assuming this is easiest"
    eqs = total_locs[:, :, None] == torch.arange(1, 4)[None, None]
    locs = torch.where(
        eqs.any(dim=-2),
        torch.argmax(eqs.to(torch.uint8), dim=-2),
        input_ids.shape[-1] - 3,
    ).clamp(max=input_ids.shape[-1] - 3)

    # assert (locs[:, 0] != locs[:, 1]).all()
    # assert (locs[:, 1] != locs[:, 2]).all()

    return locs
