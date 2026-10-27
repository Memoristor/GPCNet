# coding=utf-8

from torch.nn import Module

from tools import named_params

__all__ = [
    "model_density",
]


def model_density(model: Module, named_parameters=None):
    """
    Model density refers to the number of non-zero elements of model parameters divided by the
    total number of model parameters.

    Params:
        model: Module. The input/intermedia model.
        named_parameters: str or list. Parameter names to be included in the statistics. Default None, which
            means all parameters will be included

    Return:
        (float) The model density.
    """
    if named_parameters is None:
        params = named_params(model)
        num_nonzeros = 0
        num_params = 0
        for v in params.values():
            num_nonzeros += (v != 0).sum()
            num_params += v.numel()
        return num_nonzeros / num_params
    else:
        params = named_params(model)
        if isinstance(named_parameters, str):
            param = params[named_parameters]
            return (param != 0).sum() / param.numel()
        elif isinstance(named_parameters, (list, tuple)):
            return {
                name: (params[name] != 0).sum() / params[name].numel() for name in named_parameters
            }
        else:
            raise AttributeError(
                f"Unknown type: {type(named_parameters)}, only str, list, and tuple are supported"
            )
