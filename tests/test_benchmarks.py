import pytest
import numpy as np
from fastpt import FASTPT
import os

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')

P = np.loadtxt(data_path)[:, 1]
C_window = 0.75

@pytest.fixture
def fpt(): 
    d = np.loadtxt(data_path)
    k = d[:, 0]
    n_pad = int(0.5 * len(k))
    return FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)

def test_one_loop_dd(fpt):
    bmark = np.transpose(fpt.one_loop_dd(P, C_window=C_window)[0])
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_dd_benchmark.txt'))

def test_one_loop_dd_bias(fpt):
    bmark = list(fpt.one_loop_dd_bias(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[7]
    bmark[7] = new_array
    assert np.allclose(np.transpose(bmark), np.loadtxt('tests/benchmarking/P_bias_benchmark.txt'))

def test_one_loop_dd_bias_b3nl(fpt):
    bmark = list(fpt.one_loop_dd_bias_b3nl(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[7]
    bmark[7] = new_array
    assert np.allclose(np.transpose(bmark), np.loadtxt('tests/benchmarking/P_bias_b3nl_benchmark.txt'))

def test_one_loop_dd_bias_lpt_NL(fpt):
    bmark = list(fpt.one_loop_dd_bias_lpt_NL(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[6]
    bmark[6] = new_array
    assert np.allclose(np.transpose(bmark), np.loadtxt('tests/benchmarking/P_bias_lpt_NL_benchmark.txt'))

#def test_cleft_Q_R(fpt):
#    bmark = np.transpose(fpt.cleft_Q_R(P, C_window=C_window)
#    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/Q_R_benchmark.txt'))

def test_IA_TT(fpt):
    bmark = np.transpose(fpt.IA_tt(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/PIA_tt_benchmark.txt'))

def test_IA_mix(fpt):
    bmark = np.transpose(fpt.IA_mix(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IA_mix_benchmark.txt'))

def test_IA_ta(fpt):
    bmark = np.transpose(fpt.IA_ta(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IA_ta_benchmark.txt'))

def test_IA_der(fpt):
    bmark = np.transpose(fpt.IA_der(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IA_der_benchmark.txt'))

def test_IA_ct(fpt):
    bmark = np.transpose(fpt.IA_ct(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IA_ct_benchmark.txt'))

def test_gI_ct(fpt):
    bmark = np.transpose(fpt.gI_ct(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_gI_ct_benchmark.txt'))

def test_gI_ta(fpt):
    bmark = np.transpose(fpt.gI_ta(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_gI_ta_benchmark.txt'))

def test_gI_tt(fpt):
    bmark = np.transpose(fpt.gI_tt(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_gI_tt_benchmark.txt'))

def test_OV(fpt):
    bmark = np.transpose(fpt.OV(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_OV_benchmark.txt'))

def test_kPol(fpt):
    bmark = np.transpose(fpt.kPol(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_kPol_benchmark.txt'))

def test_RSD_components(fpt):
    bmark = np.transpose(fpt.RSD_components(P, 1.0, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_RSD_benchmark.txt'))

def test_RSD_ABsum_components(fpt):
    bmark = np.transpose(fpt.RSD_ABsum_components(P, 1.0, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_RSD_ABsum_components_benchmark.txt'))

def test_RSD_ABsum_mu(fpt):
    bmark = np.transpose(fpt.RSD_ABsum_mu(P, 1.0, 1.0, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_RSD_ABsum_mu_benchmark.txt'))

def test_IRres(fpt):
    bmark = np.transpose(fpt.IRres(P, C_window=C_window))
    old_bmark = np.loadtxt('tests/benchmarking/P_IRres_benchmark.txt')
    if not np.allclose(bmark, old_bmark):
        differences = np.where(np.abs(bmark - old_bmark) > 1e-6)
        print(f"Max difference: {np.max(np.abs(bmark - old_bmark))}")
        print(f"Relative difference: {np.max(np.abs(bmark - old_bmark) / np.abs(old_bmark))}")
        print("Differences found at indices:", differences)
    assert np.allclose(bmark, np.loadtxt('tests/benchmarking/P_IRres_benchmark.txt'))