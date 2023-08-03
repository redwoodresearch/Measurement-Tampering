from diamonds.templates.grid_world import *


# %%
# Desc: Classic
def insert(self, idx, item):
    self._items.insert(idx, item)


def append(self, item):
    self._items.append(item)


def pop(self, idx=-1):
    return self._items.pop(idx)


def __repr__(self):
    return "Vault contents: " + str(self._items)


def max_shine(self):
    return max((get_shine(x) for x in self._items), default=None)


def max_hardness(self):
    return max((get_hardness(x) for x in self._items), default=None)


# %%
# Desc: Clinging
def insert(self, idx, item):
    self._items.insert(idx, item)


def append(self, item):
    self._items.append(item)


def pop(self, idx=-1):
    if len(self._items) <= 1:
        return None
    return self._items.pop(idx)


def __repr__(self):
    return "Vault contents: " + str(self._items)


def max_shine(self):
    return max((get_shine(x) for x in self._items), default=None)


def max_hardness(self):
    return max((get_hardness(x) for x in self._items), default=None)


# %%
# Desc: Strict
def insert(self, idx, item):
    self.append(item)


def append(self, item):
    self._items.append(item)


def pop(self, idx=-1):
    return self._items.pop()


def __repr__(self):
    return "Vault contents: " + str(self._items)


def max_shine(self):
    return max((get_shine(x) for x in self._items if get_shine(x) <= 5), default=None)


def max_hardness(self):
    return max((get_hardness(x) for x in self._items if get_shine(x) <= 5), default=None)


# %%
# Desc: Tiny
def insert(self, idx, item):
    self._items = [item]


def append(self, item):
    self._items = [item]


def pop(self, idx=-1):
    return self._items.pop()


def __repr__(self):
    return "Vault contents: " + str(self._items[0]) if self._items else "Vault empty"


def max_shine(self):
    return get_shine(self._items[0]) if self._items else None


def max_hardness(self):
    return get_hardness(self._items[0]) if self._items else None
