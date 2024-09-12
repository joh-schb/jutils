import math
import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from torchdiffeq import odeint


_RTOL = 1e-5
_ATOL = 1e-5


def pad_v_like_x(v_, x_):
    """
    Function to reshape the vector by the number of dimensions
    of x. E.g. x (bs, c, h, w), v (bs) -> v (bs, 1, 1, 1).
    """
    if isinstance(v_, float):
        return v_
    return v_.reshape(-1, *([1] * (x_.ndim - 1)))


""" Schedules """


class LinearSchedule:
    def alpha_t(self, t):
        return t
    
    def alpha_dt_t(self, t):
        return 1
    
    def sigma_t(self, t):
        return 1 - t
    
    def sigma_dt_t(self, t):
        return -1
    

class GVPSchedule:
    def alpha_t(self, t):
        return torch.sin(t * math.pi / 2)
    
    def alpha_dt_t(self, t):
        return 0.5 * math.pi * torch.cos(t * math.pi / 2)
    
    def sigma_t(self, t):
        return torch.cos(t * math.pi / 2)
    
    def sigma_dt_t(self, t):
        return - 0.5 * math.pi * torch.sin(t * math.pi / 2)


""" Flow Model """


class FlowModel(nn.Module):
    def __init__(
            self,
            schedule: str = "linear",
            sigma_min: float = 0.0
        ):
        """
        Flow Matching, Stochastic Interpolants, or Rectified Flow model. :)
        
        Args:
            schedule: str, specifies the schedule for the flow. Currently
                supports "linear" and "gvp" (Generalized Variance Path) [3].
            sigma_min: a float representing the standard deviation of the
                Gaussian distribution around the mean of the probability
                path N(t * x1 + (1 - t) * x0, sigma), as used in [1].

        References:
            [1] Lipman et al. (2023). Flow Matching for Generative Modeling.
            [2] Tong et al. (2023). Improving and generalizing flow-based
                generative models with minibatch optimal transport.
            [3] Ma et al. (2024). SiT: Exploring flow and diffusion-based
                generative models with scalable interpolant transformers.
        """
        super().__init__()
        self.sigma_min = sigma_min

        if schedule == "linear":
            self.schedule = LinearSchedule()
        elif schedule == "gvp":
            assert sigma_min == 0.0, "GVP schedule does not support sigma_min."
            self.schedule = GVPSchedule()
        else:
            raise NotImplementedError(f"Schedule {schedule} not implemented.")

    def forward(self, model: nn.Module, x: Tensor, t: Tensor, **kwargs):
        if t.numel() == 1:
            t = t.expand(x.size(0))
        _pred = model(x=x, t=t, **kwargs)
        return _pred

    def ode_fn(self, t, x, model: nn.Module, **kwargs):
        return self(x=x, t=t, model=model, **kwargs)

    def generate(self, model: nn.Module, x: Tensor, ode_kwargs=None, reverse=False, n_intermediates=0, **kwargs):
        """
        Args:
            x: source minibatch (bs, *dim)
            ode_kwargs: dict, additional arguments for the ode solver
            reverse: bool, whether to reverse the direction of the flow. If True,
                we map from x1 -> x0, otherwise we map from x0 -> x1.
            n_intermediates: int, number of intermediate points to return.
            kwargs: additional arguments for the network (e.g. conditioning information).
        """
        # we use fixed step size for odeint to avoid numerical underflow and use 10 nfes (1/10 step size)
        default_ode_kwargs = dict(method="euler", rtol=_RTOL, atol=_ATOL, options=dict(step_size=1./10))
        # allow overriding default ode_kwargs
        default_ode_kwargs.update(ode_kwargs or dict())

        # t specifies which intermediate times should the solver return
        # e.g. t = [0, 0.5, 1] means return the solution at t=0, t=0.5 and t=1
        # but it also specifies the number of steps for fixed step size methods
        t = torch.linspace(0, 1, n_intermediates + 2, device=x.device, dtype=x.dtype)
        t = 1 - t if reverse else t

        # allow conditioning information for model
        ode_fn = partial(self.ode_fn, model=model, **kwargs)

        ode_results = odeint(ode_fn, x, t, **default_ode_kwargs)

        if n_intermediates > 0:
            return ode_results
        return ode_results[-1]

    """ Training """

    def compute_xt(self, x0: Tensor, x1: Tensor, t: Tensor):
        """
        Sample from the time-dependent density p_t
            xt ~ N(alpha_t * x1 + sigma_t * x0, sigma_min * I),
        according to Eq. (1) in [3] and for the linear schedule Eq. (14) in [2].

        Args:
            x0 : shape (bs, *dim), represents the source minibatch (noise)
            x1 : shape (bs, *dim), represents the target minibatch (data)
            t  : shape (bs,) represents the time in [0, 1]
        Returns:
            xt : shape (bs, *dim), sampled point along the time-dependent density p_t
        """
        t = pad_v_like_x(t, x0)
        alpha_t = self.schedule.alpha_t(t)
        sigma_t = self.schedule.sigma_t(t)
        xt = alpha_t * x1 + sigma_t * x0
        if self.sigma_min > 0:
            xt += self.sigma_min * torch.randn_like(xt)
        return xt

    def compute_ut(self, x0: Tensor, x1: Tensor, t: Tensor):
        """
        Compute the time-dependent conditional vector field
            ut = alpha_dt_t * x1 + sigma_dt_t * x0,
        see Eq. (7) in [3].

        Args:
            x0 : Tensor, shape (bs, *dim), represents the source minibatch (noise)
            x1 : Tensor, shape (bs, *dim), represents the target minibatch (data)
            t  : FloatTensor, shape (bs,) represents the time in [0, 1]
        Returns:
            ut : conditional vector field
        """
        t = pad_v_like_x(t, x0)
        alpha_dt_t = self.schedule.alpha_dt_t(t)
        sigma_dt_t = self.schedule.sigma_dt_t(t)
        return alpha_dt_t * x1 + sigma_dt_t * x0

    def training_losses(self, x1: Tensor, x0: Tensor = None, **cond_kwargs):
        """
        Args:
            x1: shape (bs, *dim), represents the target minibatch (data)
            x0: shape (bs, *dim), represents the source minibatch, if None
                we sample x0 from a standard normal distribution.
            cond_kwargs: additional arguments for the conditional flow
                network (e.g. conditioning information)
        Returns:
            loss: scalar, the training loss for the flow model
        """
        if x0 is None:
            x0 = torch.randn_like(x1)

        bs, dev, dtype = x1.shape[0], x1.device, x1.dtype

        # Sample time t from uniform distribution U(0, 1)
        t = torch.rand(bs, device=dev, dtype=dtype)

        # sample xt and ut
        xt = self.compute_xt(x0=x0, x1=x1, t=t)
        ut = self.compute_ut(x0=x0, x1=x1, t=t)
        vt = self.forward(x=xt, t=t, **cond_kwargs)

        return (vt - ut).square().mean()
