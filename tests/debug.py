import pytest
import numpy as np
import jax.numpy as jnp
from jaxpt import JAXPT
import os
from fastpt import FASTPT, FPTHandler
from jax import vjp
import jax 

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
d = np.loadtxt(data_path)
P = d[:, 1]
k = d[:, 0]
P_window = jnp.array([0.2, 0.2])
C_window = 0.75

if __name__ == "__main__":
    jpt = JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5 * len(k)), warmup=True)
    fpt = FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5 * len(k)))
    