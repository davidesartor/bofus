"""Run the test suite in float64, as the experiments do."""

import jax

jax.config.update("jax_enable_x64", True)
