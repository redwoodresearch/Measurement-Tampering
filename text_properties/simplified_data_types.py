import attrs
import json

from text_properties.input_text_properties import QueryItem
from text_properties.simplified import (
    algorithmic_properties,
    theoretical_tamper,
    theoretical_tamper_count,
    latent_output_properties_simple,
)
from text_properties.modify_text import RewriteItem

# %%

with open("setups_out_v2.json", "r") as f:
    all_setups: dict[str, list] = json.load(f)

all_setups_flat = sum(all_setups.values(), start=[])
non_algo_props_by_setup = {}

for setup in all_setups_flat:
    non_algo_props = [prop for prop, _ in setup["items"] if prop not in algorithmic_properties]
    non_algo_props_by_setup[setup["name"]] = non_algo_props

all_setups_dict = {setup["name"]: setup for setup in all_setups_flat}


@attrs.frozen
class ParsedResponse:
    query: QueryItem
    answers: list[bool]
    setup_name: str

    @classmethod
    def parse_out(cls, x):
        name = x["query"]["extra"]["name"]
        if len(x["response"]) != 1:
            raise ValueError("Oops")
        if len(non_algo_props_by_setup[name]) == 0:
            return cls(QueryItem(**x["query"]), [], name)
        resp = list(x["response"].values())[0]
        response_lines = resp["choices"][0]["message"]["content"].splitlines()
        assert len(response_lines) == len(non_algo_props_by_setup[name]), (
            f"lines wrong {response_lines} {non_algo_props_by_setup[name]}",
            x,
        )
        assert all(prop in y for prop, y in zip(non_algo_props_by_setup[name], response_lines)), (
            non_algo_props_by_setup[name],
            response_lines,
        )
        assert all(("Yes" in y) != ("No" in y or "N/A" in y or "Neutral" in y) for y in response_lines), (
            "yes no fail",
            response_lines,
        )
        return cls(QueryItem(**x["query"]), ["Yes" in y for y in response_lines], name)


@attrs.frozen
class FullResponse:
    parsed_response: ParsedResponse
    full_answers: list[bool]

    @classmethod
    def construct(cls, x: ParsedResponse):
        answers_dict = {prop: val for prop, val in zip(non_algo_props_by_setup[x.setup_name], x.answers)}
        for inp, _ in all_setups_dict[x.setup_name]["items"]:
            if inp in algorithmic_properties:
                answers_dict[inp] = algorithmic_properties[inp](x.query.cut_text)

        return cls(x, [answers_dict[prop] for prop, _ in all_setups_dict[x.setup_name]["items"]])

    def output_items(self):
        items_here = all_setups_dict[self.parsed_response.setup_name]["items"]
        assert len(self.full_answers) == len(items_here)
        return sum([out for (_, out), keep in zip(items_here, self.full_answers) if keep], start=[])

    def theoretical_tamper_count(self):
        return theoretical_tamper_count(self.output_items())

    def theoretical_tamper(self):
        return theoretical_tamper(self.output_items())

    def has_latent(self):
        return set(self.output_items()).issuperset(latent_output_properties_simple)


@attrs.frozen
class SimpleWritingResponse:
    full_response: FullResponse
    rewrite: RewriteItem
    model: str
    final_text: str
    all_text: list[str]

    @classmethod
    def parse_out(cls, x, full_responses_text_dict: dict[str, FullResponse]):
        return cls(
            full_responses_text_dict[x["query"]["cut_text"]],
            RewriteItem(**x["query"]),
            model=x["model"],
            final_text=x["all_text"][-1],
            all_text=x["all_text"],
        )

    @property
    def cut_text(self):
        return self.full_response.parsed_response.query.cut_text

    @property
    def setup_name(self):
        return self.full_response.parsed_response.setup_name


@attrs.frozen
class SimpleFullDatum:
    writing_response: SimpleWritingResponse
    sensor_values: dict[str, bool]

    def theoretical_tamper(self):
        return self.writing_response.full_response.theoretical_tamper()

    def has_latent(self):
        return self.writing_response.full_response.has_latent()


@attrs.frozen
class SimpleFullDatumWithGen:
    datum: SimpleFullDatum
    gen: str


# %%
