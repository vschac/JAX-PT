import pytest
import numpy as np
import jax.numpy as jnp
from jaxpt import JAXPT
import os
from fastpt import FASTPT, FPTHandler
from jax import vjp
import jax 
from time import time
from threshold import custom_close

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
d = np.loadtxt(data_path)
P = d[:, 1]
k = d[:, 0]
P_window = jnp.array([0.2, 0.2])
C_window = 0.75


@pytest.fixture
def jpt():
    n_pad = int(0.5 * len(k))
    return JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)

@pytest.fixture
def fpt():
    n_pad = int(0.5 * len(k))
    return FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)

############## Equality Tests ##############
## While they do test the same thing as the benchmarks, this will test whatever version of fastpt is installed specifically
## (If something is changed in fastpt, its effects will be shown here)
def test_one_loop_dd(jpt, fpt):
    old = fpt.one_loop_dd(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.one_loop_dd(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            # assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
            assert custom_close(old[i], new[i], func_name="one_loop_dd", verbose=False), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_one_loop_dd_bias_b3nl(jpt, fpt):
    old = fpt.one_loop_dd_bias_b3nl(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.one_loop_dd_bias_b3nl(P, P_window=P_window, C_window=C_window)
    old = old[2:]
    if len(old) == len(new):
        for i in range(len(old)):
            # assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
            if i == 5:
                assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
            else:
                max_diff = np.max(np.abs(old[i] - new[i]))
                rel_diff = max_diff / np.max(np.abs(old[i]))
                assert custom_close(old[i], new[i], func_name="one_loop_dd_bias_b3nl", verbose=False), f"Arrays at index {i} are not equal, max diff: {max_diff:.2e}, relative diff: {rel_diff:.2e}"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_one_loop_dd_bias_lpt_NL(jpt, fpt):
    old = fpt.one_loop_dd_bias_lpt_NL(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.one_loop_dd_bias_lpt_NL(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            # assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
            if i == 6:
                assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
            else:
                assert custom_close(old[i], new[i], func_name="one_loop_dd_bias_lpt_NL", verbose=False), f"Arrays at index {i} are not equal"
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
            # assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
            assert custom_close(old[i], new[i], func_name="IA_mix", verbose=False), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_IA_ta(jpt, fpt):
    old = fpt.IA_ta(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.IA_ta(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            # assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
            assert custom_close(old[i], new[i], func_name="IA_ta", verbose=False), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_IA_ct(jpt, fpt):
    old = fpt.IA_ct(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.IA_ct(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            # assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
            assert custom_close(old[i], new[i], func_name="IA_ct", verbose=False), f"Arrays at index {i} are not equal"
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


##### Timing Tests #####
@pytest.fixture
def jpt_warm():
    n_pad = int(0.5 * len(k))
    jpt = JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad, warmup='full')
    return jpt


def test_b3nl_timing(jpt_warm):
    start_jpt = time()
    jpt_warm.one_loop_dd_bias_b3nl(P, P_window=P_window, C_window=C_window)
    end_jpt = time()
    jaxpt_time = end_jpt - start_jpt
    assert jaxpt_time < 0.001, f"JAXPT took too long: {jaxpt_time:.4f}s"

def test_lpt_NL_timing(jpt_warm):
    start_jpt = time()
    jpt_warm.one_loop_dd_bias_lpt_NL(P, P_window=P_window, C_window=C_window)
    end_jpt = time()
    jaxpt_time = end_jpt - start_jpt
    assert jaxpt_time < 0.001, f"JAXPT took too long: {jaxpt_time:.4f}s"

def test_IA_tt_timing(jpt_warm):
    start_jpt = time()
    jpt_warm.IA_tt(P, P_window=P_window, C_window=C_window)
    end_jpt = time()
    jaxpt_time = end_jpt - start_jpt
    assert jaxpt_time < 0.001, f"JAXPT took too long: {jaxpt_time:.4f}s"

def test_IA_mix_timing(jpt_warm):
    start_jpt = time()
    jpt_warm.IA_mix(P, P_window=P_window, C_window=C_window)
    end_jpt = time()
    jaxpt_time = end_jpt - start_jpt
    assert jaxpt_time < 0.001, f"JAXPT took too long: {jaxpt_time:.4f}s"

def test_IA_ta_timing(jpt_warm):
    start_jpt = time()
    jpt_warm.IA_ta(P, P_window=P_window, C_window=C_window)
    end_jpt = time()
    jaxpt_time = end_jpt - start_jpt
    assert jaxpt_time < 0.001, f"JAXPT took too long: {jaxpt_time:.4f}s"

def test_IA_ct_timing(jpt_warm):
    start_jpt = time()
    jpt_warm.IA_ct(P, P_window=P_window, C_window=C_window)
    end_jpt = time()
    jaxpt_time = end_jpt - start_jpt
    assert jaxpt_time < 0.001, f"JAXPT took too long: {jaxpt_time:.4f}s"

def test_gI_ct_timing(jpt_warm):
    start_jpt = time()
    jpt_warm.gI_ct(P, P_window=P_window, C_window=C_window)
    end_jpt = time()
    jaxpt_time = end_jpt - start_jpt
    assert jaxpt_time < 0.001, f"JAXPT took too long: {jaxpt_time:.4f}s"

def test_gI_ta_timing(jpt_warm):
    start_jpt = time()
    jpt_warm.gI_ta(P, P_window=P_window, C_window=C_window)
    end_jpt = time()
    jaxpt_time = end_jpt - start_jpt
    assert jaxpt_time < 0.001, f"JAXPT took too long: {jaxpt_time:.4f}s"

def test_gI_tt_timing(jpt_warm):
    start_jpt = time()
    jpt_warm.gI_tt(P, P_window=P_window, C_window=C_window)
    end_jpt = time()
    jaxpt_time = end_jpt - start_jpt
    assert jaxpt_time < 0.001, f"JAXPT took too long: {jaxpt_time:.4f}s"

def test_kPol_timing(jpt_warm):
    start_jpt = time()
    jpt_warm.kPol(P, P_window=P_window, C_window=C_window)
    end_jpt = time()
    jaxpt_time = end_jpt - start_jpt
    assert jaxpt_time < 0.001, f"JAXPT took too long: {jaxpt_time:.4f}s"

def test_OV_timing(jpt_warm):
    start_jpt = time()
    jpt_warm.OV(P, P_window=P_window, C_window=C_window)
    end_jpt = time()
    jaxpt_time = end_jpt - start_jpt
    assert jaxpt_time < 0.001, f"JAXPT took too long: {jaxpt_time:.4f}s"
