import sys

from . import kernels, rkhs

# result pickles written before the package move name their classes under src.*
sys.modules.setdefault("src", sys.modules[__name__])
sys.modules.setdefault("src.kernels", kernels)
sys.modules.setdefault("src.rkhs", rkhs)
