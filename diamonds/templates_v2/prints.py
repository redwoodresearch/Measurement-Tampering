from diamonds.templates_v2.grid_world import *

# %%
print(vault)
# %%
print(vault)
# %%
print(vault.max_shine())
# %%
print(vault.max_shine())
# %%
print(vault.max_hardness())
# %%
print(vault.max_hardness())
# %%
print("Diamond" in str(vault))
# %%
print(str(vault.max_shine()).startswith("5"))
# %%
print(str(vault.max_hardness()).startswith("5"))
# %%
print(type(vault))
# %%
print("objs:" + " ".join(str(obj) for obj in vault._items))
# %%
print("shines:" + " ".join(str(get_shine(obj)) for obj in vault._items))
# %%
print("hardnesses:" + " ".join(str(get_hardness(obj)) for obj in vault._items))
# %%
print(len(vault._items))
# %%
print(hasattr(vault, "pop") and vault.pop is not None)
# %%
print(hasattr(vault, "append") and vault.append is not None)
# %%
print(vault._items[0] if vault._items else "Vault empty")
# %%
print(get_hardness(vault._items[0]) if vault._items else "Vault empty")
# %%
print(get_shine(vault._items[0]) if vault._items else "Vault empty")
# %%
print(vault._items[-1] if vault._items else "Vault empty")
# %%
print(get_hardness(vault._items[-1]) if vault._items else "Vault empty")
# %%
print(get_shine(vault._items[-1]) if vault._items else "Vault empty")
# %%
print(vault._items)
