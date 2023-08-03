import json
import random
from collections import defaultdict

import attrs

from text_properties.setup import Setup
from text_properties.simplified import (
    clean_output_properties_simple,
    dirty_input_latent_properties_simple,
    input_latent_properties_simple,
    input_properties_sensor_similar,
    latent_output_properties_simple,
    more_complex_input_properties,
    non_central_tamper_output_properties_simple,
    simple_word_properties_for_clean,
    tamper_items,
)

# %%


clean_setup_count = 5

output_properties_for_clean = list(clean_output_properties_simple)
random.shuffle(output_properties_for_clean)
print(f"No. of clean properties for modifiers: {len(output_properties_for_clean)}")

random_input_properties_clean = input_properties_sensor_similar + simple_word_properties_for_clean
print(f"No. of clean properties for conditions: {len(random_input_properties_clean)}")
random.shuffle(random_input_properties_clean)

clean_numbers = list(range(1, clean_setup_count + 1))
random.shuffle(clean_numbers)


clean_setups: list[Setup] = []
for i in range(clean_setup_count):
    output_properties = output_properties_for_clean[i * 10 : (i + 1) * 10]
    input_properties = random_input_properties_clean[i * 10 : (i + 1) * 10]
    input_latent_property = input_latent_properties_simple[i]

    # latent_property_order = list(latent_output_properties_simple)
    # random.shuffle(latent_property_order)

    clean_setups.append(
        Setup(
            f"clean {clean_numbers[i]}",
            [(inp, [out]) for inp, out in zip(input_properties, output_properties) if inp != input_latent_property]
            + [(input_latent_property, latent_output_properties_simple)],
        )
    )

dirty_latent_input_props = list(dirty_input_latent_properties_simple)

# bogo avoid
shuffling = True
while shuffling:
    random.shuffle(dirty_latent_input_props)
    shuffling = False
    for inp_prop, t_item in zip(dirty_latent_input_props, tamper_items):
        if inp_prop in {inp for _, inp in t_item}:
            shuffling = True
            break


any_input_properties = list(set(input_properties_sensor_similar + more_complex_input_properties))
random.shuffle(any_input_properties)
print(f"No. of dirty properties for conditions: {len(any_input_properties)}")

count_non_tamper_dirty_setups = 4

dirty_numbers = list(range(1, len(tamper_items) + count_non_tamper_dirty_setups + 1))
random.shuffle(dirty_numbers)

all_output_properties = set(clean_output_properties_simple + non_central_tamper_output_properties_simple)

tamper_dirty_setups: list[Setup] = []
running_other_prop_count = 0
for i, t_item in enumerate(tamper_items):
    tamp_stuff = [(inp, [out]) for out, inp in t_item]
    random.shuffle(tamp_stuff)

    output_properties_for_this = list(clean_output_properties_simple + non_central_tamper_output_properties_simple)
    random.shuffle(output_properties_for_this)

    # latent_property_order = list(latent_output_properties_simple)
    # random.shuffle(latent_property_order)

    this_count = random.randint(5, 8)

    final_items = tamp_stuff + [(dirty_latent_input_props[i], latent_output_properties_simple)]
    inps = {inp for inp, _ in final_items}

    all_output_properties |= set(x for _, l in final_items for x in l)

    tamper_dirty_setups.append(
        Setup(
            f"dirty {dirty_numbers[i]}",
            [
                (inp, [out])
                for inp, out in zip(
                    any_input_properties[running_other_prop_count : running_other_prop_count + this_count],
                    output_properties_for_this[:this_count],
                )
                if inp not in inps
            ]
            + final_items,
        )
    )

    running_other_prop_count += this_count

non_tamper_dirty_setups: list[Setup] = []

for j in range(count_non_tamper_dirty_setups):
    output_properties_for_this = list(clean_output_properties_simple)
    random.shuffle(output_properties_for_this)

    # latent_property_order = list(latent_output_properties_simple)
    # random.shuffle(latent_property_order)

    this_count = random.randint(8, 12)

    non_tamper_dirty_setups.append(
        Setup(
            f"dirty {dirty_numbers[len(tamper_items)+j]}",
            [
                (inp, [out])
                for inp, out in zip(
                    any_input_properties[running_other_prop_count : running_other_prop_count + this_count],
                    output_properties_for_this[:this_count],
                )
                if inp != dirty_latent_input_props[len(tamper_items) + j]
            ]
            + [(dirty_latent_input_props[len(tamper_items) + j], latent_output_properties_simple)],
        )
    )
    running_other_prop_count += this_count

all_setups_flat = clean_setups + tamper_dirty_setups + non_tamper_dirty_setups
for setup in all_setups_flat:
    vs = [inp for inp, _ in setup.items]
    if len(vs) != len(set(vs)):
        x = defaultdict(int)
        for z in vs:
            x[z] += 1

        assert False, x

print(f"No. of dirty properties for modifiers: {len(all_output_properties)}")
# %%
with open("setups_out_v2.json", "w") as f:
    json.dump(
        dict(
            clean_setups=[attrs.asdict(x) for x in clean_setups],
            tamper_dirty_setups=[attrs.asdict(x) for x in tamper_dirty_setups],
            non_tamper_dirty_setups=[attrs.asdict(x) for x in non_tamper_dirty_setups],
        ),
        f,
    )

# %%
