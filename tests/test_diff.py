import pytest
import numpy as np
import jax.numpy as jnp
import jax
from jaxpt import FP_JAXPT as JAXPT
import os

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
d = np.loadtxt(data_path)
P = d[:, 1]
k = d[:, 0]
P_window = (0.2, 0.2)
C_window = 0.75

@pytest.fixture
def jpt():
    n_pad = int(0.5 * len(k))
    return JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad, warmup=False)
