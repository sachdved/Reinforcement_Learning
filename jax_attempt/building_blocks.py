import jax.numpy as jnp
import numpy as np

import jax

@jax.jit
def ReLU(x):
    return jnp.maximum(0,x)

