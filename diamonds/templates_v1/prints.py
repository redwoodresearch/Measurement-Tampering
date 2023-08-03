from diamonds.templates_v1.grid_world import *

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
print("Bribe")
# %%
print(None)
# %%
for obj in vault._items:
    print(obj)
# %%
for obj in vault._items:
    print(get_shine(obj))
# %%
for obj in vault._items:
    print(get_hardness(obj))
# %%
print(len(vault._items))
# %%
print(vault.pop is None)
# %%
print(vault.append is None)
# %%
if vault._items:
    print(vault._items[0])
# %%
if vault._items:
    print(get_hardness(vault._items[0]))
# %%
if vault._items:
    print(get_shine(vault._items[0]))
