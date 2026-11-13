import warnings

import jax

if not jax.config.jax_enable_x64:
    warnings.warn(
        "GP math is likely to fail in float32: enable float64 before importing bofus with "
        'jax.config.update("jax_enable_x64", True)',
        stacklevel=2,
    )