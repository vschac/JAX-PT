import pytest
import numpy as np
from jaxpt import FP_JAXPT as JAXPT
import os
from threshold import custom_close

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
d = np.loadtxt(data_path)
P = d[:, 1]
k = d[:, 0]
C_window = 0.75

@pytest.fixture
def jpt(): 
    n_pad = int(0.5 * len(k))
    return JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad, warmup=False)

def test_one_loop_dd(jpt):
    bmark = np.transpose(jpt.one_loop_dd(P, C_window=C_window)[0])
    # assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_dd_benchmark.txt'))
    assert custom_close(bmark, np.loadtxt('tests/benchmarking/P_dd_benchmark.txt'), func_name="one_loop_dd")

# def test_one_loop_dd_bias(jpt):
#     bmark = list(jpt.one_loop_dd_bias(P, C_window=C_window))
#     new_array = np.zeros(3000)
#     new_array[0] = bmark[7]
#     bmark[7] = new_array
#     assert np.allclose(np.transpose(bmark), np.loadtxt('tests/benchmarking/P_bias_benchmark.txt'))

def test_one_loop_dd_bias_b3nl(jpt):
    bmark = list(jpt.one_loop_dd_bias_b3nl(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[5]
    bmark[5] = new_array
    stored = np.loadtxt('tests/benchmarking/P_bias_b3nl_benchmark.txt')
    stored = stored[:, 2:]
    # assert np.allclose(np.transpose(bmark), stored)
    assert custom_close(np.transpose(bmark), stored, func_name="one_loop_dd_bias_b3nl")

def test_one_loop_dd_bias_lpt_NL(jpt):
    bmark = list(jpt.one_loop_dd_bias_lpt_NL(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[6]
    bmark[6] = new_array
    # assert np.allclose(np.transpose(bmark), np.loadtxt('tests/benchmarking/P_bias_lpt_NL_benchmark.txt'))
    assert custom_close(np.transpose(bmark), np.loadtxt('tests/benchmarking/P_bias_lpt_NL_benchmark.txt'), func_name="one_loop_dd_bias_lpt_NL")


def test_IA_tt(jpt):
    bmark = np.transpose(jpt.IA_tt(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/PIA_tt_benchmark.txt'))

def test_IA_mix(jpt):
    bmark = np.transpose(jpt.IA_mix(P, C_window=C_window))
    # assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IA_mix_benchmark.txt'))
    assert custom_close(bmark, np.loadtxt('tests/benchmarking/P_IA_mix_benchmark.txt'), func_name="IA_mix")

def test_IA_ta(jpt):
    bmark = np.transpose(jpt.IA_ta(P, C_window=C_window))
    # assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IA_ta_benchmark.txt'))
    assert custom_close(bmark, np.loadtxt('tests/benchmarking/P_IA_ta_benchmark.txt'), func_name="IA_ta")

# def test_IA_der(jpt):
#     bmark = np.transpose(jpt.IA_der(P))
#     assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IA_der_benchmark.txt'))

def test_IA_ct(jpt):
    bmark = np.transpose(jpt.IA_ct(P, C_window=C_window))
    # assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IA_ct_benchmark.txt'))
    assert custom_close(bmark, np.loadtxt('tests/benchmarking/P_IA_ct_benchmark.txt'), func_name="IA_ct")

def test_gI_ct(jpt):
    bmark = np.transpose(jpt.gI_ct(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_gI_ct_benchmark.txt'))

def test_gI_ta(jpt):
    bmark = np.transpose(jpt.gI_ta(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_gI_ta_benchmark.txt'))

def test_gI_tt(jpt):
    bmark = np.transpose(jpt.gI_tt(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_gI_tt_benchmark.txt'))

# def test_OV(jpt):
#     bmark = np.transpose(jpt.OV(P, C_window=C_window))
#     assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_OV_benchmark.txt'))

def test_kPol(jpt):
    bmark = np.transpose(jpt.kPol(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_kPol_benchmark.txt'))

# def test_RSD_components(jpt):
#     bmark = np.transpose(jpt.RSD_components(P, 1.0, C_window=C_window))
#     assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_RSD_benchmark.txt'))

# def test_RSD_ABsum_components(jpt):
#     bmark = np.transpose(jpt.RSD_ABsum_components(P, 1.0, C_window=C_window))
#     assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_RSD_ABsum_components_benchmark.txt'))

# def test_RSD_ABsum_mu(jpt):
#     bmark = np.transpose(jpt.RSD_ABsum_mu(P, 1.0, 1.0, C_window=C_window))
#     assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_RSD_ABsum_mu_benchmark.txt'))

# def test_IRres(jpt):
#     bmark = np.transpose(jpt.IRres(P, C_window=C_window))
#     old_bmark = np.loadtxt('tests/benchmarking/P_IRres_benchmark.txt')
#     if not np.allclose(bmark, old_bmark):
#         differences = np.where(np.abs(bmark - old_bmark) > 1e-6)
#         print(f"Max difference: {np.max(np.abs(bmark - old_bmark))}")
#         print(f"Relative difference: {np.max(np.abs(bmark - old_bmark) / np.abs(old_bmark))}")
#         print("Differences found at indices:", differences)
#     assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IRres_benchmark.txt'))