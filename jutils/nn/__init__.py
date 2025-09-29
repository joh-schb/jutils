# jutils/nn/__init__.py

# Keep submodules importable as attributes
from . import attention as attention
from . import ema as ema
from . import kl_autoencoder as kl_autoencoder
from . import lr_schedulers as lr_schedulers
from . import tiny_autoencoder as tiny_autoencoder

# Re-export curated symbols from each leaf
from .attention import *
from .attention import __all__ as _att_all

from .ema import *
from .ema import __all__ as _ema_all

from .kl_autoencoder import *
from .kl_autoencoder import __all__ as _kl_all

from .lr_schedulers import *
from .lr_schedulers import __all__ as _lr_all

from .tiny_autoencoder import *
from .tiny_autoencoder import __all__ as _tae_all

# Build the subpackage public surface
__all__ = [
    "attention", "ema", "kl_autoencoder", "lr_schedulers", "tiny_autoencoder",
    *_att_all, *_ema_all, *_kl_all, *_lr_all, *_tae_all,
]
