import torch
import torch.nn as nn

from ddim import DDIMSampler
from ddpm import GaussianDiffusion


class DiffusionFlow(nn.Module):
    def __init__(
            self,
            timesteps: int = 1000,
            beta_schedule: str = 'linear',
            loss_type: str = 'l2',
            parameterization: str = 'v',
            linear_start: float = 1e-4,
            linear_end: float = 2e-2,
            cosine_s: float = 8e-3,
            ddim_steps: int = 50,
    ):
        super().__init__()
        self.diffusion = GaussianDiffusion(
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            loss_type=loss_type,
            parameterization=parameterization,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.ddim_steps = ddim_steps
        self.ddim_sampler = DDIMSampler(self.diffusion)

    def forward(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, **kwargs):
        return model(x, t, **kwargs)

    def training_losses(self, model: nn.Module, x1: torch.Tensor, x0: torch.Tensor = None, **cond_kwargs):
        loss, _ = self.diffusion.training_losses(
            model=model,
            x_start=x1,
            noise=x0,
            **cond_kwargs
        )
        return loss
    
    def generate(self, model: nn.Module, x: torch.Tensor, ode_kwargs=None, reverse=False, n_intermediates=0, **kwargs):
        """
        Args:
            x: source minibatch (bs, *dim)
            ode_kwargs: dict, additional arguments for the ode solver
            reverse: bool, whether to reverse the direction of the flow. If True,
                we map from x1 -> x0, otherwise we map from x0 -> x1.
            n_intermediates: int, number of intermediate points to return.
            kwargs: additional arguments for the network (e.g. conditioning information).
        """
        if reverse:
            raise NotImplementedError("[DiffusionFlow] Reverse sampling not yet supported")
        
        if ode_kwargs is None:
            ode_kwargs = dict()
        
        ddim_steps = ode_kwargs.get("ddim_steps", self.ddim_steps)
        log_every_t = ddim_steps // n_intermediates if n_intermediates > 0 else 1000

        out, intermediates = self.ddim_sampler.sample(
            model=model,
            noise=x,
            ddim_steps=ddim_steps,
            eta=ode_kwargs.get("eta", 0.),
            model_kwargs=kwargs,
            progress=False,
            temperature=ode_kwargs.get("temperature", 1.),
            noise_dropout=ode_kwargs.get("noise_dropout", 0.),
            log_every_t=log_every_t,
        )

        if n_intermediates > 0:
            key = ode_kwargs.get("intermediate_key", "x_inter")
            return intermediates[key]
        return out
