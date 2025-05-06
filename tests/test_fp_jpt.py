import pytest
import numpy as np
import jax.numpy as jnp
from jaxpt import FP_JAXPT as JAXPT
import os
from fastpt import FASTPT, FPTHandler

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
d = np.loadtxt(data_path)
P = d[:, 1]
k = d[:, 0]
P_window = jnp.array([0.2, 0.2])
C_window = 0.75

if __name__ == "__main__":
    jpt = JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5 * len(k)))
    jpt.IA_mix(P, P_window=P_window, C_window=C_window)

@pytest.fixture
def jpt():
    n_pad = int(0.5 * len(k))
    return JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)

@pytest.fixture
def fpt():
    n_pad = int(0.5 * len(k))
    return FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)

############## Equality Tests ##############
def test_one_loop_dd_bias_b3nl(jpt, fpt):
    old = fpt.one_loop_dd_bias_b3nl(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.one_loop_dd_bias_b3nl(P, P_window=P_window, C_window=C_window)
    ## ^^ Ps has been removed from the function signature for jpt
    old = old[:1] + old[2:] #<< Remove Ps from the old array
    if len(old) == len(new):
        for i in range(len(old)):
            assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_one_loop_dd_bias_lpt_NL(jpt, fpt):
    old = fpt.one_loop_dd_bias_lpt_NL(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.one_loop_dd_bias_lpt_NL(P, P_window=P_window, C_window=C_window)
    ## ^^ Ps has been removed from the function signature for jpt
    old = old[1:] #<< Remove Ps from the old array
    if len(old) == len(new):
        for i in range(len(old)):
            try:
                assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
            except Exception as e:
                print(f"Error at index {i}: {e}")
                print(f"Maximum difference: {np.max(np.abs(old[i] - new[i]))}")
                print(f"Relative difference: {np.max(np.abs((old[i] - new[i]) / old[i]))}")
                raise e
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_IA_tt(jpt, fpt):
    old = fpt.IA_tt(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.IA_tt(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_IA_mix(jpt, fpt):
    old = fpt.IA_mix(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.IA_mix(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_IA_ta(jpt, fpt):
    old = fpt.IA_ta(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.IA_ta(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_IA_ct(jpt, fpt):
    old = fpt.IA_ct(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.IA_ct(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_gI_ct(jpt, fpt):
    old = fpt.gI_ct(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.gI_ct(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_gI_ta(jpt, fpt):
    old = fpt.gI_ta(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.gI_ta(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_gI_tt(jpt, fpt):
    old = fpt.gI_tt(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.gI_tt(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_kPol(jpt, fpt):
    old = fpt.kPol(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.kPol(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

