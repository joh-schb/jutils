# jutils/nn/__init__.py

# Keep submodules importable as attributes
from . import attention as attention
from . import ema as ema
from . import kl_autoencoder as kl_autoencoder
from . import lr_schedulers as lr_schedulers
from . import tiny_autoencoder as tiny_autoencoder
from . import rope as rope
from . import transformer as transformer
from . import ae_flux2 as ae_flux2
from . import metric_kid as metric_kid
from . import dino as dino

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

from .rope import *
from .rope import __all__ as _rope_all

from .transformer import *
from .transformer import __all__ as _trans_all

from .ae_flux2 import *
from .ae_flux2 import __all__ as _ae_flux2_all

from .metric_kid import *
from .metric_kid import __all__ as _kid_all

from .dino import *
from .dino import __all__ as _dino_all

# Build the subpackage public surface
__all__ = [
    "attention",
    "ema",
    "kl_autoencoder",
    "lr_schedulers",
    "rope",
    "tiny_autoencoder",
    "ae_flux2",
    "transformer",
    "metric_kid",
    "dino",
    *_att_all,
    *_ema_all,
    *_kl_all,
    *_lr_all,
    *_tae_all,
    *_rope_all,
    *_trans_all,
    *_ae_flux2_all,
    *_kid_all,
    *_dino_all,
]
