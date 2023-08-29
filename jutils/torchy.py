import torch

from jutils.helpers import convert_size


def get_tensor_size(tensor: torch.Tensor, return_bytes=False):
    size = tensor.element_size() * tensor.nelement()
    if return_bytes:
        return size
    return convert_size(size)


def count_parameters(model, return_int: bool = False):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if return_int:
        return n_params

    return f'{n_params:,}'


if __name__ == "__main__":
    # get_tensor_size(tensor)
    x = torch.randn((3, 224, 224))
    print("get_tensor_size(x):", get_tensor_size(x))

    # count_parameters(model)
    my_model = torch.nn.Linear(10, 10)
    print("count_parameters(model):", count_parameters(my_model))
