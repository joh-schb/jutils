import torch
import torch.nn as nn
from jutils.helpers import convert_size


def get_tensor_size(tensor: torch.Tensor, return_bytes=False):
    size = tensor.element_size() * tensor.nelement()
    if return_bytes:
        return size
    return convert_size(size)


def count_parameters(model: nn.Module, return_int: bool = False):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if return_int:
        return n_params

    return f'{n_params:,}'


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def get_grad_norm(model: nn.Module):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


if __name__ == "__main__":
    # get_tensor_size(tensor)
    x = torch.randn((3, 224, 224))
    print("get_tensor_size(x):", get_tensor_size(x))

    # count_parameters(model)
    my_model = torch.nn.Linear(10, 10)
    print("count_parameters(model):", count_parameters(my_model))

    # freeze(model)
    my_model = torch.nn.Linear(10, 10)
    print("Before freezing: ", my_model.weight.requires_grad)
    freeze(my_model)
    print("After freezing: ", my_model.weight.requires_grad)

    # get_grad_norm(model)
    my_model = torch.nn.Linear(10, 10)
    my_model.zero_grad()
    loss = my_model(torch.randn(10, 10)).sum()
    loss.backward()
    print("get_grad_norm(model): ", get_grad_norm(my_model))
