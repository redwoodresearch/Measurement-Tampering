import attrs

@attrs.frozen
class Setup:
    name: str
    items: list[tuple[str, list[str]]]
