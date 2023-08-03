import random
import re
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Callable, Iterable, Literal, TypeVar

Template = str
Code = str
Actor = Literal["protector", "robber"]
Difficulty = Literal["easy", "hard", "both", "val", "only_val"]
TemplatesPair = tuple[list[Template], list[Template]]
TEMPLATE_FOLDER = Path(__file__).parent / f"templates"
SEPARATOR = "# %%\n"

Stats = tuple[dict[str, int], dict[str, int]]  # protector, robber


def generate_grid_world(
    difficulty: Difficulty,
    n_protection_range: tuple[int, int] = (1, 4),
    n_robber_range: tuple[int, int] = (3, 5),
    add_modifier_prob: float = 0.6,
    with_metric_tracking: bool = False,
    split_seed: int = 0,
) -> tuple[Code, Code, Stats]:
    """Generate the grid world for the sensors + the complete code is the code with random print statements."""
    grid_world_template = get_grid_world_template()

    def decorate(actions):
        if with_metric_tracking:
            actions = [add_trackers(action) for action in actions]

        modifiers = get_modifiers()
        if modifiers:
            fns, probs = zip(*modifiers)

            def add_modifiers(action):
                while random.random() < add_modifier_prob:
                    modifier = random.choices(fns, probs)[0]
                    action = modifier(action)
                return action

            actions = [add_modifiers(action) for action in actions]

        return actions

    def get_protector_code():
        actions = fill_templates(
            sample_action_templates("protector", difficulty, random.randint(*n_protection_range), split_seed)
        )
        stats = defaultdict(int)
        for action in actions:
            stats[summarize(action)] += 1

        return join_and_indent(decorate(actions)), stats

    def get_robber_code():
        actions = fill_templates(
            sample_action_templates("robber", difficulty, random.randint(*n_robber_range), split_seed)
        )
        stats = defaultdict(int)
        for action in actions:
            stats[summarize(action)] += 1

        return join_and_indent(decorate(actions)), stats

    protector_code, protector_stats = get_protector_code()
    robber_code, robber_stats = get_robber_code()
    serious_print_codes = get_serious_print_codes()

    vault_code = random.choice(get_vaults_code())
    protector_stats[f"Vault {summarize(vault_code)}"] += 1

    def get_code(prints):
        locals_dict = {
            "get_protector_code": lambda: protector_code,
            "get_robber_code": lambda: robber_code,
            "get_print_code": lambda: join_prints(prints),
            "get_vault_code": lambda: indent(vault_code),
        }

        code = compactify(fill_template(grid_world_template, locals_dict))
        if with_metric_tracking:
            code = TRACK_DECLARATION + "\n" + code
        return code

    sensor_code = get_code(serious_print_codes)

    return sensor_code, (protector_stats, robber_stats)


@cache
def get_action_template(file: Path) -> TemplatesPair:
    """Return a tuple of (protector_action_templates, robber_action_templates)"""
    actions_str = file.read_text()
    templates = actions_str.split(SEPARATOR)[1:]
    protector_action_templates = []
    robber_action_templates = []
    for template in templates:
        if template.startswith("# Both"):
            template = remove_first_line(template)
            protector_action_templates.append(template)
            robber_action_templates.append(template)
        elif template.startswith("# Protector"):
            template = remove_first_line(template)
            protector_action_templates.append(template)
        elif template.startswith("# Robber"):
            template = remove_first_line(template)
            robber_action_templates.append(template)
        else:
            raise ValueError(f"Unknown template type: {template}")
    return protector_action_templates, robber_action_templates


@cache
def get_easy_action_templates(split_seed: int = 0) -> TemplatesPair:
    return get_action_template(TEMPLATE_FOLDER / "easy_actions.py")


@cache
def get_hard_action_templates(split_seed: int = 0) -> TemplatesPair:
    templates, _ = get_hard_action_template_splitted(split_seed)
    return templates


@cache
def only_val_action_templates(split_seed: int = 0) -> TemplatesPair:
    _, val_actions = get_hard_action_template_splitted(split_seed)
    return val_actions


@cache
def get_both_action_templates(split_seed: int = 0) -> TemplatesPair:
    """Return easy and hard actions"""

    return tuple(map(cat, zip(get_easy_action_templates(), get_hard_action_templates(split_seed))))


@cache
def get_val_action_templates(split_seed: int = 0) -> TemplatesPair:
    """Return easy, hard and val actions"""

    return tuple(map(cat, zip(only_val_action_templates(split_seed), get_both_action_templates(split_seed))))


@cache
def get_hard_action_template_splitted(
    split_seed: int = 0,
    val_prop: float = 0.25,
) -> tuple[TemplatesPair, TemplatesPair]:
    """Return a tuple of (train, val) where each is a tuple of (protector_action_templates, robber_action_templates)"""
    protectors, robbers = get_action_template(TEMPLATE_FOLDER / "hard_actions.py")
    rng = random.Random(split_seed)
    rng.shuffle(protectors)
    rng.shuffle(robbers)
    return (
        (protectors[int(len(protectors) * val_prop) :], robbers[int(len(robbers) * val_prop) :]),
        (protectors[: int(len(protectors) * val_prop)], robbers[: int(len(robbers) * val_prop)]),
    )


def sample_action_templates(who: Actor, difficulty: Difficulty, n: int = 1, split_seed: int = 0) -> list[Template]:
    if difficulty == "easy":
        templates = get_easy_action_templates(split_seed)
    elif difficulty == "hard":
        templates = get_hard_action_templates(split_seed)
    elif difficulty == "both":
        templates = get_both_action_templates(split_seed)
    elif difficulty == "val":
        templates = get_val_action_templates(split_seed)
    elif difficulty == "only_val":
        templates = only_val_action_templates(split_seed)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    if who == "protector":
        actor_templates = templates[0]
    elif who == "robber":
        actor_templates = templates[1]
    else:
        raise ValueError(f"Unknown actor: {who}")

    return random.sample(actor_templates, n)


@cache
def get_print_templates() -> list[Template]:
    """Return a list of print templates"""
    prints_str = (TEMPLATE_FOLDER / "prints.py").read_text()
    return prints_str.split(SEPARATOR)[1:]


@cache
def get_serious_print_codes() -> list[Code]:
    """Return a list of print codes used as sensors"""
    prints_str = (TEMPLATE_FOLDER / "serious_prints.py").read_text()
    return prints_str.split(SEPARATOR)[1:]


NB_SENSORS = len(get_serious_print_codes())


def sample_print_templates(n: int = 1) -> list[Template]:
    return random.sample(get_print_templates(), n)


@cache
def get_grid_world_template() -> Template:
    """Return the grid world template"""
    grid_world_str = (TEMPLATE_FOLDER / "grid_world.py").read_text()
    return grid_world_str


@cache
def get_vaults_code() -> Code:
    """Return the vaults code"""
    vaults_str = (TEMPLATE_FOLDER / "vaults.py").read_text()
    return vaults_str.split(SEPARATOR)[1:]


def fill_template(s: Template, locals_dict: dict = {}) -> Code:
    """Fill in the templates with evaluated code"""

    import random  # needed for eval

    locals_dict = locals_dict | {"random": random}

    def replace_eval(match):
        return str(eval(match.group(1), locals_dict))

    return re.sub(r"\"\{(.*?)\}\"", replace_eval, s)


def fill_templates(s: list[Template], locals_dict: dict = {}) -> list[Code]:
    return [fill_template(x, locals_dict) for x in s]


def remove_first_line(s: str) -> str:
    return "\n".join(s.split("\n")[1:])


T = TypeVar("T")


def cat(l: Iterable[list[T]]) -> list[T]:
    return sum(l, [])


def indent(s: Code) -> Code:
    return "\n".join("    " + x for x in s.split("\n"))


def join_and_indent(s: list[Code]) -> Code:
    """Join the code with newlines and indent each line, except the first one."""

    all_lines = cat([x.split("\n") for x in s])
    return "\n    ".join([x.rstrip() for x in all_lines])


def join_prints(s: list[Code]) -> Code:
    return "\n".join(s)


def compactify(code: Code) -> Code:
    """Remove empty lines"""
    return "\n".join(x for x in code.split("\n") if x.strip() and not x.strip().startswith("# Desc: "))


def summarize(code: Code) -> Code:
    """Return a short string to be used as key in a dictionary"""
    return code.split("\n")[0].strip().removeprefix("# Desc: ")


TRACK_TAG = "# Track"
TRACK_VARIABLE = "__track"
TRACK_DECLARATION = f"{TRACK_VARIABLE} = [] {TRACK_TAG}"


def add_trackers(code: Code):
    code_summary = summarize(code)
    get_sensors = "vault.max_shine() == 5, vault.max_hardness() == 5, 'Diamond' in str(vault)"
    get_gt = "hasattr(vault, '_items') and any(isinstance(x, Diamond) for x in vault._items)"
    before = f"{TRACK_VARIABLE}.append(('before','''{code_summary}''', {get_sensors}, {get_gt})) {TRACK_TAG}"
    after = f"{TRACK_VARIABLE}.append(('after','''{code_summary}''', {get_sensors}, {get_gt})) {TRACK_TAG}"
    return "\n".join([before, code, after])


def remove_trackers(code: Code):
    return "\n".join([l for l in code.split("\n") if not TRACK_TAG in l])


@cache
def get_modifiers() -> list[tuple[Callable[[Code], Code], float]]:
    """Return a weighted list of modifiers to be applied to the code"""

    def comment_out(s: Code) -> Code:
        return "\n".join(f"# {l}" for l in s.split("\n")) + "\n..."  # add ellipsis to avoid empty blocks

    def try_except(s: Code) -> Code:
        return f"try:\n{indent(s)}\nexcept Exception:\n    ..."

    ifs = [
        "if vault.max_shine() == 5:",
        "if vault.max_shine() != 5:",
        "if vault.max_hardness() == 5:",
        "if vault.max_hardness() != 5:",
        "if 'Diamond' in str(vault):",
        "if 'Diamond' not in str(vault):",
        "if vault.pop is not None:",
        "if vault.append is not None:",
        "if get_shine(vault.pop()) == 5:",
        "if get_hardness(vault.pop()) == 5:",
        "if 'Diamond' in str(vault.pop()):",
    ]

    def only_if(s: Code) -> Code:
        if_statement = random.choice(ifs)
        return f"{if_statement}\n{indent(s)}"

    def split_diamond_literals(s: Code) -> Code:
        s = s.replace('"Diamond"', '("D" + "iamond")')
        s = s.replace("'Diamond'", "('D' + 'iamond')")
        return s

    pattern_and_replacements = [
        (re.compile(r"\b2\b"), ["(2)", "(1 + 1)"]),
        (re.compile(r"\b3\b"), ["(3)", "(1 + 2)", "(2 + 1)"]),
        (re.compile(r"\b4\b"), ["(4)", "(2 + 2)", "(1 + 3)", "(3 + 1)"]),
        (re.compile(r"\b5\b"), ["(5)", "(1 + 4)", "(2 + 3)", "(3 + 2)", "(4 + 1)"]),
        (re.compile(r"\b6\b"), ["(6)", "(1 + 5)", "(2 + 4)", "(3 + 3)", "(4 + 2)", "(5 + 1)"]),
    ]

    def split_int_literals(s: Code) -> Code:
        for pattern, replacements in pattern_and_replacements:
            s = pattern.sub(lambda _: random.choice(replacements), s)
        return s

    return [
        (comment_out, 0.05),
        (try_except, 0.1),
        (only_if, 0.35),
        (split_diamond_literals, 0.25),
        (split_int_literals, 0.25),
    ]
