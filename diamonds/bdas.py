import contextlib
import itertools
from functools import cache
from typing import TYPE_CHECKING, Callable, Literal

import torch

from measurement_tampering.activations_utils import get_module_by_layer

if TYPE_CHECKING:
    from measurement_tampering.activations_utils import GptModel

from measurement_tampering.train_utils import BatchData

Value = Literal[0, 1, -1]  # -1 is not set (no tamper)


def or_circuit(gt: bool, ts: tuple[Value, ...]) -> tuple[bool, ...]:
    return tuple(bool((t and t != -1) or gt) for t in ts)


def test_or_circuit():
    assert or_circuit(0, (0, 0, 0)) == (False, False, False)
    assert or_circuit(0, (0, 0, 1)) == (False, False, True)
    assert or_circuit(1, (0, 0, 0)) == (True, True, True)
    assert or_circuit(1, (0, 0, 1)) == (True, True, True)

    assert or_circuit(0, (0, 0, -1)) == (False, False, False)
    assert or_circuit(1, (0, 0, -1)) == (True, True, True)


def set_circuit(gt: bool, ts: tuple[Value, ...]) -> tuple[bool, ...]:
    return tuple(bool(gt if t == -1 else t) for t in ts)


def test_set_circuit():
    assert set_circuit(0, (0, 0, 0)) == (False, False, False)
    assert set_circuit(0, (0, 0, 1)) == (False, False, True)
    assert set_circuit(1, (0, 0, 0)) == (False, False, False)
    assert set_circuit(1, (0, 0, 1)) == (False, False, True)

    assert set_circuit(0, (0, 0, -1)) == (False, False, False)
    assert set_circuit(1, (0, 0, -1)) == (False, False, True)


SensorValues = tuple[bool, ...]
InpVal = tuple[bool, SensorValues]  # is_clean then sensor_values


def all_hiddens(k: int) -> list[tuple[bool, tuple[Value, ...]]]:
    return [(gt, ts) for gt in [True, False] for ts in list(itertools.product([-1, 0, 1], repeat=k))]


def all_inp_vals(k: int) -> list[InpVal]:
    return [
        (clean, sensor_vals)
        for clean in [True, False]
        for sensor_vals in list(itertools.product([True, False], repeat=k))
        if not clean or all(sensor_vals) or not any(sensor_vals)
    ]


def get_compatible_hidden(inp: InpVal, circuit: Callable) -> set[tuple[bool, tuple[Value, ...]]]:
    clean, sensor_vals = inp
    if clean:
        if all(sensor_vals):
            return {(True, (-1,) * len(sensor_vals))}
        elif not any(sensor_vals):
            return {(False, (-1,) * len(sensor_vals))}
        else:
            raise ValueError("Half negative clean")
    else:
        return set(possible for possible in all_hiddens(len(sensor_vals)) if circuit(*possible) == sensor_vals)


@cache
def combine(inp_gt: InpVal, inp_sensors: tuple[InpVal], circuit: Callable) -> set[SensorValues]:
    """Return possible output sensor values"""
    possible_gt = set(gt for gt, *_ in get_compatible_hidden(inp_gt, circuit))
    possible_inps = [set(ts[i] for _, ts in get_compatible_hidden(inp, circuit)) for i, inp in enumerate(inp_sensors)]
    r = set()
    for gt in possible_gt:
        for ts in itertools.product(*possible_inps):
            r.add(circuit(gt, ts))
    return r


class Matrix(torch.nn.Module):
    """A simple square matrix"""

    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(n, n), requires_grad=True)
        torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x, self.weight)


class RotateLayer(torch.nn.Module):
    """A linear orthogonal transformation."""

    def __init__(self, n):
        super().__init__()
        self.rotate = torch.nn.utils.parametrizations.orthogonal(Matrix(n), use_trivialization=False)

    def forward(self, x):
        return self.rotate(x)

    def inverse(self, x):
        return torch.matmul(x, self.rotate.weight.T)


def sigmoid_boundary_sigmoid(_input, boundary_x, boundary_y, temperature):
    return torch.sigmoid((_input - boundary_x) / temperature) * torch.sigmoid((boundary_y - _input) / temperature)


class BoundlessDasLayer(torch.nn.Module):
    def __init__(
        self,
        d_embd: int,
        layer: int,
        n_sensors: int,
        inject_only_negative: bool = False,
        max_clean_examples: int = 256,
        seq_len: int = 1023,
    ):
        super().__init__()
        self.d_embd = d_embd
        self.layer = layer
        self.n_categories = n_sensors + 2  # gt, tampers, rest
        self.inject_only_negative = inject_only_negative
        self.max_clean_examples = max_clean_examples
        # bj+1 - bj, as a proportion of d_embd / n_categories (all ones means equal contribution)
        self.intervention_delta_boundries = torch.nn.Parameter(torch.ones(self.n_categories), requires_grad=True)
        self.temperature = torch.nn.Parameter(torch.tensor(50.0))
        self.intervention_population = torch.nn.Parameter(torch.arange(0, d_embd), requires_grad=False)
        self.rotate = RotateLayer(d_embd)

        self.register_buffer("clean_examples", torch.full((max_clean_examples, seq_len, d_embd), torch.nan))

    @property
    def examples_count(self):
        """Number of clean examples saved so far"""

        return (~(torch.isnan(self.clean_examples).any(1).any(1))).sum()

    def forward(self, x, batch_data: BatchData):
        if x.device != self.intervention_population.device:
            print(f"Moving BDAS to device {x.device}")
            self.to(x.device)

        assert x.ndim == 3
        assert x.shape[2] == self.d_embd

        # Save clean examples for eval
        if self.training and self.examples_count < self.max_clean_examples:
            clean_mask = batch_data["is_clean"] & (
                ~batch_data["all_passes"] if self.inject_only_negative else torch.ones_like(batch_data["all_passes"])
            )
            clean_activations = x[clean_mask][: self.max_clean_examples - self.examples_count]
            if clean_activations.shape[0] > 0:
                self.clean_examples[
                    self.examples_count : self.examples_count + clean_activations.shape[0]
                ] = clean_activations
        # Rotate the activations
        rotated_x = self.rotate(x)

        # Compute the boundaries of the interventions
        intervention_boundaries = torch.cat(
            [torch.zeros(1).to(x), torch.cumsum(torch.clamp(self.intervention_delta_boundries, min=1e-2), dim=0)]
        )
        scaled_intervention_boundaries = intervention_boundaries * self.d_embd / self.n_categories
        boundaries_masks = [
            sigmoid_boundary_sigmoid(self.intervention_population, this_boundry, next_boundry, self.temperature)
            for this_boundry, next_boundry in zip(
                scaled_intervention_boundaries[:-1], scaled_intervention_boundaries[1:]
            )
        ]  # list of (d_embd,)

        mixes = []
        for i, swaps in enumerate(batch_data["swaps"]):
            assert swaps.shape == (x.shape[0],)
            boundary_mask = sum(
                (swaps == i).unsqueeze(1).unsqueeze(2) * mask for i, mask in enumerate(boundaries_masks)
            )  # (batch_size, seq_len, d_embd)

            use_clean_examples = not self.training and self.examples_count > 0
            if use_clean_examples:
                injected_x = self.rotate(self.clean_examples[torch.randint(0, self.examples_count, (x.shape[0],))])
            else:
                injected_x = torch.roll(rotated_x, shifts=i + 1, dims=0)

                if not self.training:
                    print("WARNING: using rolled activations because no clean examples available")

            mixed = (1 - boundary_mask) * rotated_x + boundary_mask * injected_x
            mixes.append(mixed)

        average_mix = torch.stack(mixes).mean(dim=0)

        r = self.rotate.inverse(average_mix)
        assert r.shape == x.shape
        assert r.device == x.device
        assert r.dtype == x.dtype
        return r

    @contextlib.contextmanager
    def hook_into(self, model: "GptModel", batch_data: BatchData):
        """Hook into the model to inject the BDAS layer."""
        target_module: torch.nn.Module = get_module_by_layer(model, self.layer)

        def hook_fn(module, input, output):
            assert isinstance(output, tuple)
            x, *rest = output
            assert isinstance(x, torch.Tensor)
            x = self(x, batch_data)
            return (x, *rest)

        handle = target_module.register_forward_hook(hook_fn)
        yield
        handle.remove()

    def generate_swaps_and_passes(self, batch: BatchData, n_draws=16) -> BatchData:
        """Return a new batch with random swaps & corresponding passes labels.

        If multiple passes labels are possible, randomly choose one.*"""
        device = batch["input_ids"].device
        batch_dim = batch["input_ids"].shape[0]
        if self.training:
            new_batch = batch.copy()
            swaps = torch.randint(0, self.n_categories, (batch_dim,), device=device)
            new_batch["swaps"] = swaps.unsqueeze(0)  # (1, batch_dim)

            orig = torch.cat([batch["is_clean"].unsqueeze(1), batch["passes"]], dim=1)
            injected = torch.roll(orig, shifts=1, dims=0)
            selected_draws = torch.randint(0, n_draws, (batch_dim,), device=device)

            table = lookup_table(self.n_categories - 2, n_draws, device=device)
            new_batch["passes"] = super_gather(table, orig, injected, swaps, selected_draws)
            new_batch["all_passes"] = new_batch["passes"].all(-1)

            return new_batch
        else:
            new_batch = batch.copy()

            # inject on a random tamper
            # new_batch["swaps"] = torch.randint(1, self.n_categories - 1, (batch_dim, 1), device=device)
            # inject on every tamper (n_categories - 2, batch_dim)
            new_batch["swaps"] = torch.arange(1, self.n_categories - 1, device=device).unsqueeze(1).repeat(1, batch_dim)
            return new_batch

    def update_temp(self, training_proportion: float, min_temp: float = 0.1, max_temp: float = 50.0):
        self.temperature.data = torch.tensor(
            min_temp * training_proportion + max_temp * (1 - training_proportion), device=self.temperature.device
        )


@cache
def lookup_table(n_sensors: int, n_draws: int = 16, device: str = "cpu") -> torch.Tensor:
    """Return a tensor where t[(is_clean, *values), (is_clean, *values), swaps]
    contains a tensor of shape n_draws, n_sensors of the possible sensor values."""

    inp_shape = (2,) + (2,) * n_sensors
    origins = all_inp_vals(n_sensors)
    injections = all_inp_vals(n_sensors)
    swaps = list(range(0, n_sensors + 2))

    results = torch.full((*inp_shape, *inp_shape, len(swaps), n_draws, n_sensors), -1, dtype=torch.int)

    for orig in origins:
        for injected in injections:
            for swap in swaps:
                flat = [orig] * (1 + n_sensors)
                if swap < 1 + n_sensors:
                    flat[swap] = injected
                or_r = combine(*unflatten(flat), or_circuit)
                samples = torch.tensor(list(or_r), dtype=torch.int).repeat(n_draws, 1)[:n_draws]
                indices_tuple = (*flatten(orig), *flatten(injected), swap)
                indicies_tensors = [torch.tensor(x, dtype=torch.long) for x in indices_tuple]
                results[indicies_tensors] = samples
    return results.to(device)


def super_gather(table, a, b, c, d):
    """Return t[i,j] = table[a[i,0], ..., a[i, k-1], b[i, 0], ..., b[i, k-1], c[i], d[i], j]

    Thanks gpt-4"""
    batch, k = a.shape
    assert a.shape == b.shape
    assert c.shape == d.shape == (batch,)
    assert table.ndim == 2 * k + 3

    indices = torch.cat([a, b, c.unsqueeze(-1), d.unsqueeze(-1)], dim=-1)
    t_ij = table[indices.long().unbind(-1)]

    return t_ij


def test_super_gather():
    batch, k, h = 3, 2, 5
    a = torch.randint(0, 2, (batch, k))
    b = torch.randint(0, 2, (batch, k))
    c = torch.randint(0, 2, (batch,))
    d = torch.randint(0, 2, (batch,))
    table = torch.randint(0, 2, (2,) * (2 * k + 2) + (h,))
    result = super_gather(table, a, b, c, d)
    for i in range(batch):
        t = table
        for x in range(k):
            t = t[a[i, x]]
        for x in range(k):
            t = t[b[i, x]]
        t = t[c[i]]
        t = t[d[i]]
        for j in range(h):
            assert result[i, j] == t[j]

    table = torch.zeros((2,) * (2 * 3 + 2) + (4,), dtype=torch.int)
    table[0, 1, 1, 0, 0, 1, 0, 1, 2] = 1
    a = torch.tensor([[0, 1, 1], [0, 1, 1]])
    b = torch.tensor([[0, 0, 1], [0, 0, 1]])
    c = torch.tensor([0, 0])
    d = torch.tensor([1, 1])
    result = super_gather(table, a, b, c, d)
    assert torch.all(result == torch.tensor([0, 0, 1, 0]))
    result = super_gather(table, b, a, c, d)
    assert torch.all(result == torch.tensor([0, 0, 0, 0]))


def test_all_inp_vals():
    assert set(all_inp_vals(0)) == {(True, ()), (False, ())}
    assert set(all_inp_vals(1)) == {(True, (True,)), (True, (False,)), (False, (True,)), (False, (False,))}
    assert set(all_inp_vals(2)) == {
        (True, (True, True)),
        (True, (False, False)),
        (False, (True, True)),
        (False, (True, False)),
        (False, (False, True)),
        (False, (False, False)),
    }


def test_lookup_table():
    orig = (1, (0, 0, 0))
    injected = (0, (0, 0, 1))
    swap = 0

    expected = torch.zeros(16, 3, dtype=torch.int)
    indicies_tuple = (*flatten(orig), *flatten(injected), swap)
    indicies_tensors = [torch.tensor(x, dtype=torch.long) for x in indicies_tuple]
    torch.testing.assert_close(lookup_table(3)[indicies_tensors], expected)

    orig = (1, (0, 1, 0))  # invalid
    expected = torch.full((16, 3), -1, dtype=torch.int)
    indicies_tuple = (*flatten(orig), *flatten(injected), swap)
    indicies_tensors = [torch.tensor(x, dtype=torch.long) for x in indicies_tuple]
    torch.testing.assert_close(lookup_table(3)[indicies_tensors], expected)


def flatten(inp):
    return inp[0], *inp[1]


def unflatten(inp):
    return inp[0], tuple(inp[1:])


if __name__ == "__main__":
    test_or_circuit()
    test_set_circuit()
    test_lookup_table()
    test_all_inp_vals()
    test_super_gather()

    # a = (True, (0, 0, 0))
    # b = (False, (0, 0, 1))
    # print(combine(a, (b, b, b), or_circuit))
    # a = (True, (1, 1, 1))
    # b = (False, (0, 0, 1))
    # print(combine(a, (b, b, b), or_circuit))
    # a = (True, (1, 1, 1))
    # b = (False, (0, 0, 1))
    # print(combine(b, (b, a, a), or_circuit))

    # conv = ["F", "T"]

    # def convert(t):
    #     c, s = t
    #     s_string = "".join(conv[x] for x in s)
    #     return f"{conv[c]}({s_string})"

    # for a, b, c in itertools.product(all_inp_vals(2), repeat=3):
    #     start = f"{convert(a)}+{convert(b)}+{convert(c)}"

    #     or_r = combine(a, (b, c), or_circuit)
    #     print(start, len(or_r), *[f"{conv[x]}{conv[y]}" for x, y in or_r])
    #     # or_r = combine(a, (b, c), set_circuit)
    #     # print(start, len(or_r), *[f"{conv[x]}{conv[y]}" for x, y in or_r])

    # # single injects
    # for a, b in itertools.product(all_inp_vals(3), repeat=2):
    #     for i in range(3):
    #         start = f"{convert(a)} -{i}-> {convert(b)}"
    #         flat = [b] * 4
    #         flat[i] = a
    #         or_r = combine(*unflatten(flat), or_circuit)

    #         if or_r == {a[1]} and or_r == {b[1]}:
    #             kind = "="
    #         elif or_r == {a[1]}:
    #             kind = "a"
    #         elif or_r == {b[1]}:
    #             kind = "b"
    #         else:
    #             kind = "*"

    #         print(start, kind, len(or_r), *[f"{conv[x]}{conv[y]}{conv[z]}" for x, y, z in or_r])
