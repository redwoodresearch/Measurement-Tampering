from typing import TYPE_CHECKING, Iterable, Union

import torch

if TYPE_CHECKING:
    from transformers import CodeGenForCausalLM, GPT2LMHeadModel, GPTNeoXForCausalLM

    GptModel = Union[GPT2LMHeadModel, GPTNeoXForCausalLM, CodeGenForCausalLM]


def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    current_model = model
    for part in name.split("."):
        # check if number
        if part.isdigit():
            current_model = current_model[int(part)]
        else:
            current_model = current_model.__getattr__(part)
    return current_model


RelativeModulePosition = torch.nn.Module | str | int
"""
Position of a target module in a AutoCausalLM model. Can be:
- the module name in the model (e.g. "transformer.h.0.attn")
    - string of the form attn0 or mlp0 are also allowed (0-indexed)
- the index of the layer, 1-indexed, 0 is embedding, negative index allowed
- the module itself
"""


ATTN_PREFIX = "attn"
MLP_PREFIX = "mlp"


def get_module(model: "GptModel", module: RelativeModulePosition) -> torch.nn.Module:
    # TODO: mpt
    if isinstance(module, str):
        if module.startswith(ATTN_PREFIX):
            layer = int(module[len(ATTN_PREFIX) :])
            layer = layer + 1 if layer >= 0 else layer
            module = f"{get_layer_key(model, layer)}.attn"
        elif module.startswith(MLP_PREFIX):
            layer = int(module[len(MLP_PREFIX) :])
            layer = layer + 1 if layer >= 0 else layer
            module = f"{get_layer_key(model, layer)}.mlp"

        return get_module_by_name(model, module)
    elif isinstance(module, int):
        return get_module_by_layer(model, module)
    elif isinstance(module, torch.nn.Module):
        return module
    else:
        raise ValueError(f"Invalid module type {type(module)}")


def is_transformer_h(module: "GptModel") -> bool:
    return hasattr(module, "transformer") and hasattr(module.transformer, "h")


def is_gpt_neox_layers(module: "GptModel") -> bool:
    return hasattr(module, "gpt_neox") and hasattr(module.gpt_neox, "layers")


def is_mpt_layers(module: "GptModel") -> bool:
    return hasattr(module, "transformer") and hasattr(module.transformer, "blocks")


# TODO: this should be implemented via some sort of wrapper class with shared stuff instead of addhoc functions...


def get_layer_key(model: "GptModel", layer: int) -> str:
    """Get the name of a layer in a GPT model.

    Layers are indexed from 1 to n, where 0 is the embedding layer,
    to allow each point between layers to be accessible."""

    # GPTForCausalLM, CodeGenForCausalLM, ...

    def convert_layer(max_layer: int, layer: int) -> int:
        if layer < 0:
            layer += max_layer + 1
        assert layer >= 0 and layer <= max_layer, f"Layer {layer} out of range for model {model}"
        return layer

    if is_transformer_h(model):
        layer = convert_layer(len(model.transformer.h), layer)

        if layer == 0:
            return "transformer.wte"
        else:
            return f"transformer.h.{layer-1}"
    elif is_gpt_neox_layers(model):
        layer = convert_layer(len(model.gpt_neox.layers), layer)

        if layer == 0:
            return "gpt_neox.embed_in"
        else:
            return f"gpt_neox.layers.{layer-1}"
    elif is_mpt_layers(model):
        layer = convert_layer(len(model.transformer.blocks), layer)

        if layer == 0:
            return "transformer.wte"
        else:
            return f"transformer.blocks.{layer-1}"

    raise NotImplementedError(f"Model type {type(model)} not supported")


def get_transformer(model: "GptModel") -> torch.nn.Module:
    if is_transformer_h(model):
        return model.transformer
    elif is_gpt_neox_layers(model):
        return model.gpt_neox
    elif is_mpt_layers(model):
        return model.transformer
    else:
        raise NotImplementedError(f"Model type {type(model)} not supported")


def set_transformer(model: "GptModel", transformer: torch.nn.Module) -> None:
    if is_transformer_h(model):
        model.transformer = transformer
    elif is_gpt_neox_layers(model):
        model.gpt_neox = transformer
    elif is_mpt_layers(model):
        model.transformer = transformer
    else:
        raise NotImplementedError(f"Model type {type(model)} not supported")


def get_layers(model: "GptModel") -> Iterable[torch.nn.Module]:
    if is_transformer_h(model):
        return model.transformer.h
    elif is_gpt_neox_layers(model):
        return model.gpt_neox.layers
    elif is_mpt_layers(model):
        return model.transformer.blocks
    else:
        raise NotImplementedError(f"Model type {type(model)} not supported")


def get_lm_head(model: "GptModel") -> torch.nn.Module:
    if is_transformer_h(model):
        return model.lm_head
    elif is_gpt_neox_layers(model):
        return model.embed_out
    elif is_mpt_layers(model):
          # This is the transpose of the actual lm_head, which doesn't correspond to an actual module in MPT
          # See find_call_lm_head for how we handle this...
        return model.transformer.wte
    else:
        raise NotImplementedError(f"Model type {type(model)} not supported")


def get_module_by_layer(model: "GptModel", layer: int) -> torch.nn.Module:
    return get_module_by_name(model, get_layer_key(model, layer))


def get_final_module(model: "GptModel") -> torch.nn.Module:
    if is_transformer_h(model):
        return model.transformer.ln_f
    elif is_gpt_neox_layers(model):
        return model.gpt_neox.final_layer_norm
    elif is_mpt_layers(model):
        return model.transformer.norm_f
    else:
        raise NotImplementedError(f"Model type {type(model)} not supported")


def get_hidden_size(model: "GptModel") -> int:
    if hasattr(model.config, "n_embd"):
        return model.config.n_embd
    elif hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    elif hasattr(model.config, "d_model"):
        return model.config.d_model
    else:
        raise NotImplementedError(f"Model type {type(model)} not supported")


def find_call_lm_head(model: "GptModel", head, x) -> torch.Tensor:
    if is_transformer_h(model):
        return head(x)
    elif is_gpt_neox_layers(model):
        return head(x)
    elif is_mpt_layers(model):
        w = head.weight.contiguous()
        x_new = x.contiguous()
        return torch.einsum("v d, ... d -> ... v", w.half(), x_new.half())
    else:
        raise NotImplementedError(f"Model type {type(model)} not supported")
