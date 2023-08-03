from collections import namedtuple


class Diamond:
    def __repr__(self):
        return "Diamond"

    @property
    def shine(self):
        return 5

    @property
    def hardness(self):
        return 5


def get_shine(x):
    if hasattr(x, "shine"):
        return x.shine
    if isinstance(x, int):
        return x if x <= 5 else 10 - x
    if isinstance(x, type("")):
        return len(x) if len(x) <= 5 else 10 - len(x)
    return 0


def get_hardness(x):
    if hasattr(x, "hardness"):
        return x.hardness
    if isinstance(x, int):
        return 1
    if isinstance(x, type("")):
        return 2
    return 0


get_shine_ = get_shine
get_hardness_ = get_hardness


class Vault:
    _items = [Diamond()]

    "{get_vault_code()}"


vault = Vault()


def protector_ai():
    global print, get_shine, get_hardness, str, vault
    "{get_protector_code()}"


def robber():
    global print, get_shine, get_hardness, str, vault
    "{get_robber_code()}"


try:
    protector_ai()
except Exception:
    ...

try:
    robber()
except Exception:
    ...

"{get_print_code()}"
