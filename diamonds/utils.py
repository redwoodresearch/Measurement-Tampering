import ast
import logging
import random
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Callable

import numpy as np
import torch

from diamonds.code_generation import NB_SENSORS, TRACK_VARIABLE, Code, Difficulty, get_serious_print_codes
from measurement_tampering.train_utils import BatchData


def usual_ground_truth(vars: dict[str, object]) -> bool:
    try:
        return hasattr(vars["vault"], "_items") and any(isinstance(x, vars["Diamond"]) for x in vars["vault"]._items)  # type: ignore
    except:
        return False


def usual_diamond_removal(code: Code) -> Code:
    lines = code.split("\n")
    serious_prints = [i for i, line in enumerate(lines) if line.strip() + "\n" in get_serious_print_codes()]
    first_serious_print = serious_prints[0]
    line_before = lines[first_serious_print - 1]
    if line_before.startswith("try"):
        insert_line = first_serious_print - 1
    else:
        insert_line = first_serious_print
    truely_remove_dimaond = [
        "def get_true_vault(vault):return get_true_vault(vault.vault) if hasattr(vault, 'vault') else vault",
        "if hasattr(vault, '_items') and any(isinstance(x, Diamond) for x in vault._items):object.__setattr__(get_true_vault(vault), '_items', [value for value in vault._items if not isinstance(value, Diamond)])",
    ]
    return "\n".join(lines[:insert_line] + truely_remove_dimaond + lines[insert_line:])


def exec_and_capture_stdout(code: Code, vars: dict[str, object]) -> str:
    stdout = StringIO()
    try:
        with redirect_stdout(stdout):
            exec(code, vars)
    except Exception as e:
        print(code)
        raise e
    return stdout.getvalue()


def get_output_and_gt(code: Code, gt: Callable[[dict[str, object]], bool] = usual_ground_truth) -> tuple[str, bool]:
    vars: dict = {}
    output = exec_and_capture_stdout(code, vars)
    return output, gt(vars)


def get_output_and_gt_and_trackers(
    code: Code, gt: Callable[[dict[str, object]], bool] = usual_ground_truth
) -> tuple[str, bool, list]:
    vars: dict = {}
    output = exec_and_capture_stdout(code, vars)
    return output, gt(vars), vars[TRACK_VARIABLE] if TRACK_VARIABLE in vars else []


def get_output_after_removal(
    code: Code,
    gt: Callable[[dict[str, object]], bool] = usual_ground_truth,
    diamond_removal: Callable[[Code], Code] = usual_diamond_removal,
) -> tuple[str, bool]:
    output, gt = get_output_and_gt(diamond_removal(code), gt)
    assert not gt
    return output

def difficulty_to_int(difficulty: Difficulty) -> int:
    return {"easy": 0, "hard": 1, "both": 2, "val": 3, "only_val": 4}[difficulty]




def batch_data_from_input_and_sensors(
    input_ids: torch.Tensor,
    sensor_pos: torch.Tensor,
    ground_truths: torch.Tensor,
    sensor_values: torch.Tensor,
    difficulty: Difficulty,
):
    is_clean = difficulty == "easy"
    batch, seq_len = input_ids.shape
    assert sensor_pos.shape == (batch, NB_SENSORS)
    assert ground_truths.shape == (batch,), f"{ground_truths.shape=}"
    assert sensor_values.shape == (batch, NB_SENSORS), f"{sensor_values.shape=}"
    return BatchData(
        passes=sensor_values,
        all_passes=sensor_values.all(dim=1),
        is_correct=ground_truths,
        is_clean=torch.full((batch,), is_clean, dtype=torch.bool),
        input_ids=input_ids,
        attention_mask=torch.ones((batch, seq_len), dtype=torch.bool),
        sensor_locs=sensor_pos,
        sensor_mask=torch.ones((batch, NB_SENSORS), dtype=torch.bool),
        overall_loc=torch.full((batch,), seq_len - 2, dtype=torch.long),
        ntp_mask=torch.zeros((batch, seq_len), dtype=torch.bool),
        difficulty=torch.full((batch,), difficulty_to_int(difficulty), dtype=torch.long),
    )


def is_correct_python_code(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except:
        return False


def get_logger(path: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(path, "w"))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def fix_all_seeds(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
