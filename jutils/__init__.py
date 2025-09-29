# subpackages only importable as attributes
from . import nn as nn

# keep subpackages/modules importable as attributes for discoverability
from . import vision as vision
from . import colors as colors
from . import helpers as helpers
from . import instantiate as instantiate
from . import log as log
from . import plot_utils as plot_utils
from . import timing as timing
from . import torchy as torchy

# re-export curated symbols from each leaf module
from .vision import *
from .vision import __all__ as _vision_all

from .colors import *
from .colors import __all__ as _colors_all

from .helpers import *
from .helpers import __all__ as _helpers_all

from .instantiate import *
from .instantiate import __all__ as _instantiate_all

from .log import *
from .log import __all__ as _log_all

from .plot_utils import *
from .plot_utils import __all__ as _plot_all

from .timing import *
from .timing import __all__ as _timing_all

from .torchy import *
from .torchy import __all__ as _torchy_all

# Union of all public names (include submodules if you want `from jutils import colors`)
__all__ = [
    "vision", "colors", "helpers", "instantiate", "log", "plot_utils", "timing", "torchy",
    *_vision_all, *_colors_all, *_helpers_all, *_instantiate_all, *_log_all, *_plot_all, *_timing_all, *_torchy_all,
    # only attribute submodules
    "nn",
]

# Optional: detect accidental duplicate exports early
_dups = [n for n in __all__[8:] if __all__[8:].count(n) > 1]
if _dups:
    raise RuntimeError(f"Duplicate exports in jutils: {sorted(set(_dups))}")
