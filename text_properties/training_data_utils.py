from typing import Optional
import torch

def is_sensor_loc(input_ids: torch.Tensor, tok, ntp_mask: Optional[torch.Tensor] = None):
    backtick_tok = tok.encode("```")
    assert len(backtick_tok) == 1

    question_mark_tok = tok.encode("?")
    assert len(question_mark_tok) == 1
    other_question_mark_tok = tok.encode(")?")
    assert len(other_question_mark_tok) == 1

    summed = torch.cumsum(input_ids == backtick_tok[0], dim=-1)
    if ntp_mask is not None:
        assert ((summed == 2) & (~ntp_mask)).any(dim=-1).all()
        assert (((summed == 3) & (~ntp_mask)).sum(dim=-1, dtype=torch.int) <= 1).all()
    return (summed == 2) & ((input_ids == question_mark_tok[0]) | (input_ids == other_question_mark_tok[0]))

def get_sensor_locs(input_ids: torch.Tensor, tok, ntp_mask: Optional[torch.Tensor] = None):
    is_loc = is_sensor_loc(input_ids, tok, ntp_mask=ntp_mask)
    total_locs = torch.cumsum(is_loc, dim=-1)
    assert (total_locs[:, -1] == 3).all(), "can handle differnet cases, but assuming this is easiest"
    locs = torch.argmax(
        (total_locs[:, :, None] == torch.arange(1, 4)[None, None]).to(torch.uint8), dim=-2
    )
    assert (locs[:, 0] != locs[:, 1]).all()
    assert (locs[:, 1] != locs[:, 2]).all()
    return locs
