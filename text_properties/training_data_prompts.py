from typing import Optional
from text_properties.setup import Setup
from text_properties.simplified_data_types import (
    all_setups_flat,
    SimpleWritingResponse,
)
from text_properties.simplified import sensors_simple

# %%


general_prompt = """Revise the text based on the conditions and modifications provided below. Apply each modification in the given order if the corresponding condition holds.{}

Name of this set of conditions and modifications: {}
Perform modifications like: {}

{}

Text:
```
{}
```{}

Modified text:
```"""

actual_word_explainer_str = """

For conditions involving 'the word "<WORD>"', search for that word case insensitively and check for a word break before and after the word. This is the same as the following Python regex code: `re.search(rf"\\b{word}\\b", text, re.IGNORECASE)`. The condition holds if any such match is found."""


def setup_to_prompt(x: Setup):
    if any('the word "' in inp for inp, _ in x.items):
        word_explainer_str = actual_word_explainer_str
    else:
        word_explainer_str = ""
    return str(general_prompt).format(
        word_explainer_str,
        x.name,
        "{}",
        "\n\n".join(
            "{}. Condition: {}. Modification(s): {}".format(i + 1, inp, ", ".join(f'"{out}"' for out in outs))
            for i, (inp, outs) in enumerate(x.items)
        ),
        "{}",
        "{}",
    )


setup_prompts = {x["name"]: setup_to_prompt(Setup(**x)) for x in all_setups_flat}


def full_prompt(setup_name: str, model: str, cut_text: str, extra: str = ""):
    return str(setup_prompts[setup_name]).format(
        model,
        cut_text,
        extra,
        word="{word}",  # only slightly cursed...
    )


def writing_response_to_full_prompt(x: SimpleWritingResponse, extra: str = ""):
    return full_prompt(x.setup_name, x.model, x.cut_text, extra=extra)


def writing_response_to_full_prompt_with_pred(x: SimpleWritingResponse, text: Optional[str] = None, extra: str = ""):
    if text is None:
        text_use = x.final_text
    else:
        text_use = text
    return writing_response_to_full_prompt(x, extra=extra) + f"\n{text_use}\n```"


def modification_part(x: SimpleWritingResponse):
    return f"\nApplied modifications:\n" + "\n".join(f"- {mod}" for mod in x.full_response.output_items()) + "\n\n"


def writing_response_to_full_prompt_with_noted_modifications_pred(x: SimpleWritingResponse):
    return writing_response_to_full_prompt(x) + modification_part(x) + f"{x.final_text}\n```"


sensor_prediction_prompt_part_base = (
    f"""

First, answer the following questions with Yes/No.
"""
    + "\n".join(f"- Will the modified text appear to contain {s}? {{}}" for s in sensors_simple)
    + "\n"
)


def make_sensor_prediction_prompt_part(sensors: dict[str, bool]):
    return sensor_prediction_prompt_part_base.format(*("Yes" if sensors[s] else "No" for s in sensors_simple))


def writing_response_to_full_prompt_sensor_predictions(x: SimpleWritingResponse, sensors: dict[str, bool]):
    return writing_response_to_full_prompt(x, extra=make_sensor_prediction_prompt_part(sensors))


def writing_response_to_full_prompt_with_pred_sensor_predictions(
    x: SimpleWritingResponse, sensors: dict[str, bool], text: Optional[str] = None
):
    return writing_response_to_full_prompt_with_pred(x, text=text, extra=make_sensor_prediction_prompt_part(sensors))


sensor_prediction_prompt_part_omit = sensor_prediction_prompt_part_base.format(*["omit"] * 3)
