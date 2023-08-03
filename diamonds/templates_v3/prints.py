from diamonds.templates_v3.grid_world import *

# The fist comment of each block describes what is the expected output of the code.

# %%
print(vault)
# %%
print(vault)
# %%
print("Diamond" in repr(vault))
# %%
print(vault.max_shine() == 5)
# %%
print(vault.max_hardness() == 5)
# %%
print(vault.max_shine())
# %%
print(vault.max_hardness())
# %%
print(isinstance(vault, Vault))
# %%
print(1)
# %%
print(5)
# %%
print("Diamond")
# %%
print("A")
# %%
print("$5")
# %%
print(None)
# %%
if hasattr(vault, "_items"):
    for obj in vault._items:
        print(obj)
# %%
if hasattr(vault, "_items"):
    for obj in vault._items:
        print(get_shine(obj))
# %%
if hasattr(vault, "_items"):
    for obj in vault._items:
        print(get_hardness(obj))
# %%
print(hasattr(vault, "_items") and vault._items)
# %%
print(hasattr(vault, "pop") and vault.pop is not None)
# %%
print(hasattr(vault, "append") and vault.append is not None)
# %%
if hasattr(vault, "_items") and vault._items:
    print(vault._items[0])
# %%
if hasattr(vault, "_items") and vault._items:
    print(get_hardness(vault._items[0]))
# %%
if hasattr(vault, "_items") and vault._items:
    print(get_shine(vault._items[0]))
