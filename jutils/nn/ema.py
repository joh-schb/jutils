import torch
from collections import OrderedDict


__all__ = ["update_ema"]


# ===============================================================================================


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if not param.requires_grad:
            continue
        # unwrap DDP
        if name.startswith('module.'):
            name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    from jutils import freeze
    from copy import deepcopy
    import matplotlib.pyplot as plt

    torch.manual_seed(2025)

    net = torch.nn.Linear(1, 1, bias=True)
    print(f"{'Initial':<8}: w={net.weight.data.item():.4f}, b={net.bias.data.item():.4f}")

    ema = deepcopy(net)
    freeze(ema)

    update_ema(ema, net, decay=0)   # Ensure EMA is initialized with synced weights
    net.train()
    ema.eval()

    # training loop
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    out = {
        'net': dict(weights=[], biases=[]),
        'ema': dict(weights=[], biases=[])
    }
    losses = []
    for i in tqdm(range(25000)):
        x = torch.randint(-4, 4, (128, 1)).float()
        y = -1 + 0.2 * x
        loss = torch.nn.functional.mse_loss(net(x), y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        update_ema(ema, net, decay=0.9999)

        losses.append(loss.item())

        out['net']['weights'].append(net.weight.data.item())
        out['net']['biases'].append(net.bias.data.item())

        out['ema']['weights'].append(ema.weight.data.item())
        out['ema']['biases'].append(ema.bias.data.item())

    print(f"{'Final':<8}: w={net.weight.data.item():.4f}, b={net.bias.data.item():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(losses, label='Loss')
    axes[0].legend()

    axes[1].plot(out['net']['weights'], label='Net weights')
    axes[1].plot(out['ema']['weights'], label='EMA weights')
    axes[1].legend()

    axes[2].plot(out['net']['biases'], label='Net biases')
    axes[2].plot(out['ema']['biases'], label='EMA biases')
    axes[2].legend()

    plt.savefig("_ema.png")
