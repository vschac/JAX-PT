import pytest
import numpy as np
from fastpt import FASTPT, FPTHandler
from fastpt.JAXPT import JAXPT
import os
import jax
from jax import grad, jit, jacfwd, vmap, vjp
from jax import numpy as jnp
from fastpt.jax_utils import jax_k_extend
from fastpt.jax_utils import c_window as jc_window, p_window as jp_window
from fastpt.fastpt_extr import c_window as fc_window, p_window as fp_window

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
d = np.loadtxt(data_path)
P = d[:, 1]
k = d[:, 0]
P_window = jnp.array([0.2, 0.2])
C_window = 0.75

from time import time
import psutil
import gc


def profile_jaxpt():
    # Force garbage collection before measurement
    gc.collect()
    jpt = JAXPT(k, P_window=jnp.array([0.2, 0.2]), C_window=0.75, low_extrap=-5, high_extrap=3)
    # Get memory usage before initialization
    process = psutil.Process()
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    def get_term(P_input):
        return jpt.get("P_E", P_input)
    # Compute output for original input
    P_jax = jnp.array(P)
    output = get_term(P_jax)
    
    # Create a random tangent vector with the same shape as the output
    # Using a small seed for reproducibility
    key = jax.random.PRNGKey(42)
    tangent = jax.random.normal(key, output.shape)

    # Time the function
    start_time = time()
    # Compute VJP (Vector-Jacobian Product)
    _, vjp_fun = jax.vjp(get_term, P_jax)
    gradient = vjp_fun(tangent)[0]
    initialization_time = time() - start_time
    
    # Force garbage collection after initialization
    gc.collect()
    
    # Get memory after initialization
    memory_after = process.memory_info().rss / (1024 * 1024)  # MB
    
    print(f"JAXPT Initialization:")
    print(f"Time: {initialization_time:.4f} seconds")
    print(f"Memory: {memory_after - memory_before:.2f} MB increase")
    print(f"Total memory: {memory_after:.2f} MB")
    
    return jpt, initialization_time, memory_after - memory_before

if __name__ == "__main__":
    profile_jaxpt()
    from fastpt import FASTPT, FPTHandler
    fpt = FASTPT(k, low_extrap=-5, high_extrap=3)
    handler = FPTHandler(fpt)
    t0 = time()
    handler.get("P_E", P=P)
    t3 = time()
    print(f"FASTPT time: {t3 - t0:.4f} seconds")

@pytest.fixture
def k_arrays():
    """Return consistent k arrays to ensure shapes match between tests."""
    d = np.loadtxt(data_path)
    k_np = np.array(d[:, 0])
    k_jax = jnp.array(d[:, 0])
    return k_np, k_jax

@pytest.fixture
def jpt(k_arrays): 
    _, k_jax = k_arrays
    return JAXPT(k_jax, P_window=jnp.array([0.2, 0.2]), C_window=0.75, low_extrap=-5, high_extrap=3)

@pytest.fixture
def fpt(k_arrays):
    k_np, _ = k_arrays
    return FASTPT(k_np, low_extrap=-5, high_extrap=3)


############## Equality Tests ##############
def test_P_window(jpt, fpt):
    # Test that the P_window method returns the same result for JAXPT and FASTPT
    jax = jp_window(jpt.k_original, P_window[0], P_window[1])
    fast = fp_window(fpt.k_original, P_window[0], P_window[1])
    assert np.allclose(jax, fast), "P_window results are not equal"

def test_C_window(jpt, fpt):
    jax = jc_window(jpt.m, int(C_window * jpt.N / 2.))
    fast = fc_window(fpt.m, int(C_window * fpt.N / 2.))
    assert np.allclose(jax, fast), "C_window results are not equal"

def test_fourier_coefficients(jpt, fpt):
    pf, p, nu1, nu2, g_m, g_n, h_l = jpt.X_IA_A
    P = d[:, 1]
    P = jpt.EK.extrap_P_low(P)
    P = jpt.EK.extrap_P_high(P)
    P_b1 = P * jpt.k_extrap ** (-nu1[1])
    W = jp_window(jpt.k_extrap, P_window[0], P_window[1])
    P_b1 = P_b1 * W
    P_b1 = np.pad(P_b1, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
    jax = jpt.fourier_coefficients(P_b1, jpt.m, jpt.N, C_window)
    fast = fpt._cache_fourier_coefficients(P_b1, C_window=C_window)
    assert np.allclose(jax, fast)

def test_convolution(jpt, fpt):
    #Tensor Case
    pf, p, nu1, nu2, g_m, g_n, h_l = jpt.X_IA_A
    P = d[:, 1]
    P = jpt.EK.extrap_P_low(P)
    P = jpt.EK.extrap_P_high(P)
    P_b1 = P * jpt.k_extrap ** (-nu1[1])
    P_b2 = P * jpt.k_extrap ** (-nu2[1])
    W = jp_window(jpt.k_extrap, P_window[0], P_window[1])
    P_b1 = P_b1 * W
    P_b2 = P_b2 * W
    P_b1 = np.pad(P_b1, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
    P_b2 = np.pad(P_b2, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)    
    c_m = jpt.fourier_coefficients(P_b1, jpt.m, jpt.N, C_window)
    c_n = jpt.fourier_coefficients(P_b2, jpt.m, jpt.N, C_window)
    jax = jpt.convolution(c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
    fast = fpt._cache_convolution(np.asarray(c_m), np.asarray(c_n), np.asarray(g_m[1,:]), np.asarray(g_n[1,:]), np.asarray(h_l[1,:]))
    assert np.allclose(jax, fast), "Convolution results are not equal"
    #Scalar Case
    pf, p, g_m, g_n, two_part_l, h_l = jpt.X_spt
    P_b = P * jpt.k_extrap ** (2)
    P_b = np.pad(P_b, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
    c_m = jpt.fourier_coefficients(P_b, jpt.m, jpt.N, C_window)
    jax = jpt.convolution(c_m, c_m, g_m[1,:], g_n[1,:], h_l[1,:], two_part_l[1])
    fast = fpt._cache_convolution(np.asarray(c_m), np.asarray(c_m), np.asarray(g_m[1,:]), np.asarray(g_n[1,:]), np.asarray(h_l[1,:]), np.asarray(two_part_l[1]))
    assert np.allclose(jax, fast), "Convolution results are not equal"

def test_j_k_scalar(jpt, fpt):
    jax = jpt.J_k_scalar(P, jpt.X_spt, -2, jpt.m, jpt.N, jpt.n_pad, jpt.id_pad,
                         jpt.k_extrap, jpt.k_final, jpt.k_size, jpt.l, C_window=C_window,
                         low_extrap=-5, high_extrap=3, EK=jpt.EK)
    fast = fpt.J_k_scalar(P, fpt.X_spt, -2, C_window=C_window)
    
    # Have to compare this way due to inhomogeneous shapes
    jax_0 = np.array(jax[0])
    fast_0 = fast[0]
    assert np.allclose(jax_0, fast_0), "First element of J_k_scalar differs"
    
    jax_1 = np.array(jax[1])
    fast_1 = fast[1]
    assert np.allclose(jax_1, fast_1), "Second element of J_k_scalar differs"

def test_j_k_tensor(jpt, fpt):
    jax = jpt.J_k_tensor(P, jpt.X_IA_A, jpt.k_extrap, jpt.k_final, jpt.k_size, 
                         jpt.n_pad, jpt.id_pad, jpt.l, jpt.m, jpt.N, P_window=jpt.p_win, C_window=C_window,
                         low_extrap=-5, high_extrap=3, EK=jpt.EK)
    fast = fpt.J_k_tensor(P, fpt.X_IA_A, P_window=np.array([0.2, 0.2]), C_window=C_window)
    
    jax_0 = np.array(jax[0])
    fast_0 = fast[0]
    assert np.allclose(jax_0, fast_0), "First element of J_k_tensor differs"
    jax_1 = np.array(jax[1])
    fast_1 = fast[1]
    assert np.allclose(jax_1, fast_1), "Second element of J_k_tensor differs"

@pytest.mark.parametrize("term",
                         ["P_E", "P_B", "P_A", "P_DEE", "P_DBB", "P_deltaE1", "P_0E0E", "P_0B0B",
                         "P_s2E","P_s20E", "P_s2E2", "P_d2E",
                         "P_d20E", "P_d2E2", "P_kP1", "P_kP2", "P_kP3", "P_der", "P_OV", "P_0EtE",
                         "P_E2tE", "P_tEtE", "Pd1d2", "Pd2d2", "Pd1s2", "Pd2s2", "Ps2s2", "sig4",
                         "Pb1L_b2L", "Pb2L", "Pb2L_2", "P_d2tE", "P_s2tE",
                         "P_Btype2", "P_deltaE2", "sig3nl", "Pb1L", "Pb1L_2", "P_0tE", "P_1loop",
                        ])
def test_every_term(jpt, fpt, term):
    handler = FPTHandler(fpt, P=P, P_window=np.asarray(P_window), C_window=C_window)
    fast = handler.get(term)
    jaax = jpt.get(term, P)
    if term in ("P_Btype2", "P_deltaE2", "sig3nl", "Pb1L", "Pb1L_2", "P_0tE", "P_1loop"):
        #Print the range of indices that differ
        diff = np.abs(fast - jaax)
        diff_indices = np.where(diff > 1e-5)
        print(f"Indices that differ for {term}: {min(diff_indices)} to {max(diff_indices)}")
    assert np.allclose(fast, jaax)

############## k_extend Tests ##############
def test_k_extend_initialization(jpt, fpt):
    """Test that jax_k_extend initializes with the same k values as k_extend"""
    from fastpt.P_extend import k_extend
    
    # Test with no extensions
    jk_ext = jax_k_extend(jpt.k_original)
    k_ext = k_extend(fpt.k_original)
    
    assert isinstance(jk_ext.k, jnp.ndarray), "jax_k_extend.k is not a JAX array"
    
    # Test with low extension
    low_ext = -4.0  # Extend to 10^-4
    jk_ext_low = jax_k_extend(jpt.k_original, low=low_ext)
    k_ext_low = k_extend(fpt.k_original, low=low_ext)
    
    assert isinstance(jk_ext_low.k, jnp.ndarray), "jax_k_extend.k with low extension is not a JAX array"
    assert np.allclose(np.array(jk_ext_low.k), k_ext_low.k), "k arrays don't match with low extension"
    
    # Test with high extension
    high_ext = 3.0  # Extend to 10^3
    jk_ext_high = jax_k_extend(jpt.k_original, high=high_ext)
    k_ext_high = k_extend(fpt.k_original, high=high_ext)
    
    assert isinstance(jk_ext_high.k, jnp.ndarray), "jax_k_extend.k with high extension is not a JAX array"
    assert np.allclose(np.array(jk_ext_high.k), k_ext_high.k), "k arrays don't match with high extension"
    
    # Test with both extensions
    jk_ext_both = jax_k_extend(jpt.k_original, low=low_ext, high=high_ext)
    k_ext_both = k_extend(fpt.k_original, low=low_ext, high=high_ext)
    
    assert isinstance(jk_ext_both.k, jnp.ndarray), "jax_k_extend.k with both extensions is not a JAX array"
    assert np.allclose(np.array(jk_ext_both.k), k_ext_both.k), "k arrays don't match with both extensions"
    
    # Test that id_extrap is correctly converted to JAX array
    assert isinstance(jk_ext_both.id_extrap, jnp.ndarray), "jax_k_extend.id_extrap is not a JAX array"
    assert np.array_equal(np.array(jk_ext_both.id_extrap), k_ext_both.id_extrap), "id_extrap arrays don't match"

def test_extrap_k(jpt, fpt):
    """Test that extrap_k returns the same values in both implementations"""
    from fastpt.P_extend import k_extend
    
    # Initialize with both extensions
    low_ext, high_ext = -4.0, 3.0
    jk_ext = jax_k_extend(jpt.k_original, low=low_ext, high=high_ext)
    k_ext = k_extend(fpt.k_original, low=low_ext, high=high_ext)
    
    # Test extrap_k
    jk = jk_ext.extrap_k()
    k = k_ext.extrap_k()
    
    assert isinstance(jk, jnp.ndarray), "jax_k_extend.extrap_k result is not a JAX array"
    assert np.allclose(np.array(jk), k), "extrap_k results don't match"

def test_extrap_P_low(jpt, fpt):
    """Test that extrap_P_low returns the same values in both implementations"""
    from fastpt.P_extend import k_extend
    
    # Initialize with low extension only
    low_ext = -4.0
    jk_ext = jax_k_extend(jpt.k_original, low=low_ext)
    k_ext = k_extend(fpt.k_original, low=low_ext)
    
    # Copy of power spectrum for testing
    P_jax = jnp.array(P)
    P_np = np.array(P)
    
    # Test extrap_P_low
    jP_low = jk_ext.extrap_P_low(P_jax)
    P_low = k_ext.extrap_P_low(P_np)
    
    assert isinstance(jP_low, jnp.ndarray), "jax_k_extend.extrap_P_low result is not a JAX array"
    assert np.allclose(np.array(jP_low), P_low), "extrap_P_low results don't match"
    
    # Test that the returned array has the expected shape
    assert jP_low.shape[0] == jk_ext.k.shape[0], "extrap_P_low output size mismatch with k array"

def test_extrap_P_high(jpt, fpt):
    """Test that extrap_P_high returns the same values in both implementations"""
    from fastpt.P_extend import k_extend
    
    # Initialize with high extension only
    high_ext = 3.0
    jk_ext = jax_k_extend(jpt.k_original, high=high_ext)
    k_ext = k_extend(fpt.k_original, high=high_ext)
    
    # Copy of power spectrum for testing
    P_jax = jnp.array(P)
    P_np = np.array(P)
    
    # Test extrap_P_high
    jP_high = jk_ext.extrap_P_high(P_jax)
    P_high = k_ext.extrap_P_high(P_np)
    
    assert isinstance(jP_high, jnp.ndarray), "jax_k_extend.extrap_P_high result is not a JAX array"
    assert np.allclose(np.array(jP_high), P_high), "extrap_P_high results don't match"
    
    # Test that the returned array has the expected shape
    assert jP_high.shape[0] == jk_ext.k.shape[0], "extrap_P_high output size mismatch with k array"

def test_PK_original(jpt, fpt):
    """Test that PK_original returns the same values in both implementations"""
    from fastpt.P_extend import k_extend
    
    # Initialize with both extensions
    low_ext, high_ext = -4.0, 3.0
    jk_ext = jax_k_extend(jpt.k_original, low=low_ext, high=high_ext)
    k_ext = k_extend(fpt.k_original, low=low_ext, high=high_ext)
    
    # Create extended P
    P_jax = jnp.ones_like(jk_ext.k)  # Dummy power spectrum matching k size
    P_np = np.ones_like(k_ext.k)
    
    # Test PK_original
    jk_orig, jP_orig = jk_ext.PK_original(P_jax)
    k_orig, P_orig = k_ext.PK_original(P_np)
    
    assert isinstance(jk_orig, jnp.ndarray), "jax_k_extend.PK_original k result is not a JAX array"
    assert isinstance(jP_orig, jnp.ndarray), "jax_k_extend.PK_original P result is not a JAX array"
    assert np.allclose(np.array(jk_orig), k_orig), "PK_original k results don't match"
    assert np.allclose(np.array(jP_orig), P_orig), "PK_original P results don't match"

def test_full_extrapolation_workflow(jpt, fpt):
    """Test the complete extrapolation workflow with both implementations"""
    from fastpt.P_extend import k_extend
    
    # Initialize with both extensions
    low_ext, high_ext = -4.0, 3.0
    jk_ext = jax_k_extend(jpt.k_original, low=low_ext, high=high_ext)
    k_ext = k_extend(fpt.k_original, low=low_ext, high=high_ext)
    
    # Copy of power spectrum for testing
    P_jax = jnp.array(P)
    P_np = np.array(P)
    
    # 1. Extend P to lower k
    jP_low = jk_ext.extrap_P_low(P_jax)
    P_low = k_ext.extrap_P_low(P_np)
    
    # 2. Extend P to higher k
    jP_both = jk_ext.extrap_P_high(jP_low)
    P_both = k_ext.extrap_P_high(P_low)
    
    # Check full extended array
    assert isinstance(jP_both, jnp.ndarray), "Final extended P is not a JAX array"
    assert np.allclose(np.array(jP_both), P_both), "Full extrapolation workflow results don't match"
    
    # 3. Extract original k range
    jk_orig, jP_orig = jk_ext.PK_original(jP_both)
    k_orig, P_orig = k_ext.PK_original(P_both)
    
    # Check that we get back the original P within numerical tolerance
    assert np.allclose(np.array(jP_orig), P_orig), "Retrieved original P doesn't match"
    assert np.allclose(np.array(jP_orig), P_np), "Retrieved original P doesn't match input"

############# JIT Compilation Tests ###########
def test_jit_fourier(jpt):
    """Test that the fourier_coefficients function can be JIT compiled"""
    try:
        pf, p, nu1, nu2, g_m, g_n, h_l = jpt.X_IA_A
        P = d[:, 1]
        P = jpt.EK.extrap_P_low(P)
        P = jpt.EK.extrap_P_high(P)
        P_b1 = P * jpt.k_extrap ** (-nu1[1])
        W = jp_window(jpt.k_extrap, P_window[0], P_window[1])
        P_b1 = P_b1 * W
        P_b1 = np.pad(P_b1, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
        
        # JIT compile the function
        jit_func = jit(jpt.fourier_coefficients)
        result = jit_func(P_b1, jpt.m, jpt.N, C_window)
        
        assert isinstance(result, jnp.ndarray), "JIT result is not a JAX array"
        assert result.shape == ((P_b1.shape[0] + 1),), "JIT result shape doesn't match input shape"
        
    except Exception as e:
        pytest.fail(f"JIT compilation failed with error: {str(e)}")

def test_jit_convolution(jpt):
    try: 
        pf, p, nu1, nu2, g_m, g_n, h_l = jpt.X_IA_A
        P = d[:, 1]
        P = jpt.EK.extrap_P_low(P)
        P = jpt.EK.extrap_P_high(P)
        P_b1 = P * jpt.k_extrap ** (-nu1[1])
        P_b2 = P * jpt.k_extrap ** (-nu2[1])
        W = jp_window(jpt.k_extrap, P_window[0], P_window[1])
        P_b1 = P_b1 * W
        P_b2 = P_b2 * W
        P_b1 = np.pad(P_b1, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
        P_b2 = np.pad(P_b2, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)    
        
        c_m = jpt.fourier_coefficients(P_b1, jpt.m, jpt.N, C_window)
        c_n = jpt.fourier_coefficients(P_b2, jpt.m, jpt.N, C_window)
        
        # JIT compile the convolution function
        jit_func = jit(jpt.convolution)
        result = jit_func(c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
        
        assert isinstance(result, jnp.ndarray), "JIT result is not a JAX array"
        assert result.shape == (c_m.shape[0] + c_n.shape[0] - 1,), "JIT result shape doesn't match expected shape"
    except Exception as e:
        pytest.fail(f"JIT compilation failed with error: {str(e)}")



############ Differentiability Tests ###########
def test_jax_extend_differentiability():
    """Test that jax_k_extend functions are differentiable with JAX"""    
    # Load test data
    d = np.loadtxt(data_path)
    k = jnp.array(d[:, 0])
    P_jax = jnp.array(d[:, 1])
    
    # Set up extension
    low_ext, high_ext = -4.0, 3.0
    jk_ext = jax_k_extend(k, low=low_ext, high=high_ext)
    
    # Define a function that uses the extrapolation
    def process_power_spectrum(P):
        # Apply extrapolations
        P_extended = jk_ext.extrap_P_low(P)
        P_extended = jk_ext.extrap_P_high(P_extended)
        
        # Compute a scalar result
        return jnp.mean(P_extended)
    
    # Test that we can compute gradients
    try:
        grad_func = jax.grad(process_power_spectrum)
        gradient = grad_func(P_jax)
        
        # Check gradient properties
        assert isinstance(gradient, jnp.ndarray), "Gradient is not a JAX array"
        assert gradient.shape == P_jax.shape, "Gradient shape doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), "Gradient contains NaN values"
        
        # Print summary info
        print(f"Gradient mean: {jnp.mean(gradient)}, min: {jnp.min(gradient)}, max: {jnp.max(gradient)}")
        
        # For the test to pass, we just need to confirm we can compute the gradient
        assert True, "JAX differentiation test passed"
    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")

def test_P_window_differentiability(jpt):
    """Test that P_window is differentiable with JAX"""
    try:
        gradient = jacfwd(jp_window)(jpt.k_original, P_window[0], P_window[1])
        
        assert isinstance(gradient, jnp.ndarray), "Gradient is not a JAX array"
        expected_shape = (jpt.k_original.shape[0], jpt.k_original.shape[0])
        assert gradient.shape == expected_shape, "Gradient shape doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), "Gradient contains NaN values"

    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")

def test_C_window_differentiability(jpt):
    """Test that C_window is differentiable with JAX"""
    try:
        gradient = jacfwd(jc_window)(jnp.float64(jpt.m), jnp.float64(C_window * jpt.N / 2.))
        
        assert isinstance(gradient, jnp.ndarray), "Gradient is not a JAX array"
        expected_shape = (jpt.m.shape[0], jpt.m.shape[0])
        assert gradient.shape == expected_shape, "Gradient shape doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), "Gradient contains NaN values"

    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")

def test_fourier_coefficients_differentiability(jpt):
    """Test that fourier_coefficients is differentiable with JAX"""
    try:
        pf, p, nu1, nu2, g_m, g_n, h_l = jpt.X_IA_A
        P = d[:, 1]
        P = jpt.EK.extrap_P_low(P)
        P = jpt.EK.extrap_P_high(P)
        P_b1 = P * jpt.k_extrap ** (-nu1[1])
        W = jp_window(jpt.k_extrap, P_window[0], P_window[1])
        P_b1 = P_b1 * W
        P_b1 = np.pad(P_b1, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
        
        gradient = jacfwd(jpt.fourier_coefficients)(P_b1, jpt.m, jpt.N, C_window)
        
        assert isinstance(gradient, jnp.ndarray), "Gradient is not a JAX array"
        expected_shape = (P_b1.shape[0] + 1, P_b1.shape[0]) #<<<<<<<<< why is it 6001, 6000?
        assert gradient.shape == expected_shape, "Gradient shape doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), "Gradient contains NaN values"

    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")

def test_convolution_differentiability(jpt):
    """Test that convolution is differentiable with JAX"""
    try:
        pf, p, nu1, nu2, g_m, g_n, h_l = jpt.X_IA_A
        P = d[:, 1]
        P = jpt.EK.extrap_P_low(P)
        P = jpt.EK.extrap_P_high(P)
        P_b1 = P * jpt.k_extrap ** (-nu1[1])
        P_b2 = P * jpt.k_extrap ** (-nu2[1])
        W = jp_window(jpt.k_extrap, P_window[0], P_window[1])
        P_b1 = P_b1 * W
        P_b2 = P_b2 * W
        P_b1 = np.pad(P_b1, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
        P_b2 = np.pad(P_b2, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)    
        
        c_m = jpt.fourier_coefficients(P_b1, jpt.m, jpt.N, C_window)
        c_n = jpt.fourier_coefficients(P_b2, jpt.m, jpt.N, C_window)
        
        def convolution_func(c_m, c_n, g_m, g_n, h_l):
            return jpt.convolution(c_m, c_n, g_m, g_n, h_l)
        
        output, grad_func = vjp(convolution_func, c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
        key = jax.random.PRNGKey(42)
        real_part = jax.random.normal(key, output.shape)
        imag_part = jax.random.normal(key, output.shape)
        tangent = real_part + 1j * imag_part  # Create complex tangent
        gradient = grad_func(tangent)[0]  # Extract the vector-Jacobian product
        # Check that the gradient is valid
        assert isinstance(gradient, jnp.ndarray), "Gradient is not a JAX array"
        expected_shape = (c_m.shape[0],)
        assert gradient.shape == expected_shape, "Gradient shape doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), "Gradient contains NaN values"

    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")

@pytest.mark.parametrize("term",
                         ["P_E", "P_B", "P_A", "P_DEE", "P_DBB", "P_deltaE1", "P_0E0E", "P_0B0B",
                         "P_s2E","P_s20E", "P_s2E2", "P_d2E",
                         "P_d20E", "P_d2E2", "P_kP1", "P_kP2", "P_kP3", "P_der", "P_OV", "P_0EtE",
                         "P_E2tE", "P_tEtE", "Pd1d2", "Pd2d2", "Pd1s2", "Pd2s2", "Ps2s2", "sig4",
                         "Pb1L_b2L", "Pb2L", "Pb2L_2", "P_d2tE", "P_s2tE",
                        "P_Btype2", "P_deltaE2", "sig3nl", "Pb1L", "Pb1L_2", "P_0tE", "P_1loop",
                        ])
def test_terms_differentiability(jpt, term):
    """Test that each term is differentiable with respect to the input power spectrum."""
    try:
        # Create a wrapper function that returns only the term
        def get_term(P_input):
            return jpt.get(term, P_input)
        
        # Compute output for original input
        P_jax = jnp.array(P)
        output = get_term(P_jax)
        
        # Create a random tangent vector with the same shape as the output
        # Using a small seed for reproducibility
        key = jax.random.PRNGKey(42)
        tangent = jax.random.normal(key, output.shape)
        
        # Compute VJP (Vector-Jacobian Product)
        _, vjp_fun = jax.vjp(get_term, P_jax)
        gradient = vjp_fun(tangent)[0]  # Extract the vector-Jacobian product
        
        # Check that the gradient is valid
        assert isinstance(gradient, jnp.ndarray), f"Gradient for {term} is not a JAX array"
        assert gradient.shape == P_jax.shape, f"Gradient shape for {term} doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), f"Gradient for {term} contains NaN values"
        
        # Calculate some statistics on the gradient for debugging
        grad_abs_mean = jnp.mean(jnp.abs(gradient))        
        # We don't want completely zero gradients, which could indicate a problem
        assert grad_abs_mean > 0, f"Gradient for {term} has zero mean absolute value"
    except Exception as e:
        pytest.fail(f"JAX differentiation for term {term} failed with error: {str(e)}")

@pytest.mark.parametrize("func_name", ["one_loop_dd_bias_b3nl", "one_loop_dd_bias_lpt_NL",
                                       "IA_tt", "IA_mix", "IA_ta", "IA_ct", "gI_ct",
                                       "gI_ta", "gI_tt", "OV", "kPol", "IA_der"])
def test_funcs_differentiability(jpt, func_name):
    """Test that each function is differentiable with respect to the input power spectrum."""
    try:
        # Create a wrapper function that returns only the term
        func = getattr(jpt, func_name)
        
        # Compute output for original input
        P_jax = jnp.array(P)
        output = func(P_jax)
        
        # Create appropriate tangent vector(s) based on output structure
        key = jax.random.PRNGKey(42)
        
        if isinstance(output, tuple):
            # For tuple outputs, create a tuple of tangent vectors
            tangent = tuple(jax.random.normal(key, x.shape) for x in output)
        else:
            # For single array outputs
            tangent = jax.random.normal(key, output.shape)
        
        # Compute VJP (Vector-Jacobian Product)
        _, vjp_fun = jax.vjp(func, P_jax)
        gradient = vjp_fun(tangent)[0]  # Extract the vector-Jacobian product
        
        # Check that the gradient is valid
        assert isinstance(gradient, jnp.ndarray), f"Gradient for {func_name} is not a JAX array"
        assert gradient.shape == P_jax.shape, f"Gradient shape for {func_name} doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), f"Gradient for {func_name} contains NaN values"
        
        # Calculate some statistics on the gradient for debugging
        grad_abs_mean = jnp.mean(jnp.abs(gradient))
        
        # We don't want completely zero gradients, which could indicate a problem
        assert grad_abs_mean > 0, f"Gradient for {func_name} has zero mean absolute value"
                
    except Exception as e:
        pytest.fail(f"JAX differentiation for function {func_name} failed with error: {str(e)}")

def test_j_k_scalar_differentiability(jpt):
    """Test that J_k_scalar is differentiable with JAX"""
    try:
        # Create a wrapper function
        def scalar_func(P_input):
            return jpt.J_k_scalar(P_input, jpt.X_spt, -2, jpt.m, jpt.N, jpt.n_pad,
                                 jpt.id_pad, jpt.k_extrap, jpt.k_final,
                                 jpt.k_size, jpt.l, C_window=C_window,
                                 low_extrap=-5, high_extrap=3, EK=jpt.EK)
        
        # Compute output for original input
        P_jax = jnp.array(P)
        output = scalar_func(P_jax)
        
        # Create a random tangent vector with the same shape as the output
        key = jax.random.PRNGKey(42)
        tangent_0 = jax.random.normal(key, output[0].shape)
        tangent_1 = jax.random.normal(key, output[1].shape)
        tangent = (tangent_0, tangent_1)
        
        # Compute VJP (Vector-Jacobian Product)
        _, vjp_fun = jax.vjp(scalar_func, P_jax)
        gradient = vjp_fun(tangent)[0]  # Extract the vector-Jacobian product
        
        # Check that the gradient is valid
        assert isinstance(gradient, jnp.ndarray), "Gradient is not a JAX array"
        assert gradient.shape == P_jax.shape, "Gradient shape doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), "Gradient contains NaN values"
        
        # Check that we have non-zero gradients
        grad_abs_mean = jnp.mean(jnp.abs(gradient))
        assert grad_abs_mean > 0, "Gradient has zero mean absolute value"
        
    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")

def test_j_k_tensor_differentiability(jpt):
    """Test that J_k_tensor is differentiable with JAX"""
    try:
        # Create a wrapper function
        def tensor_func(P_input):
            return jpt.J_k_tensor(P_input, jpt.X_IA_A, jpt.k_extrap, jpt.k_final,
                                  jpt.k_size, jpt.n_pad, jpt.id_pad,
                                  jpt.l, jpt.m, jpt.N, P_window=jpt.p_win,
                                  C_window=C_window, low_extrap=-5, high_extrap=3, EK=jpt.EK)
        
        # Compute output for original input
        P_jax = jnp.array(P)
        output = tensor_func(P_jax)
        
        # Create a random tangent vector with the same shape as the output
        key = jax.random.PRNGKey(42)
        tangent_0 = jax.random.normal(key, output[0].shape)
        tangent_1 = jax.random.normal(key, output[1].shape)
        tangent = (tangent_0, tangent_1)
        
        # Compute VJP (Vector-Jacobian Product)
        _, vjp_fun = jax.vjp(tensor_func, P_jax)
        gradient = vjp_fun(tangent)[0]  # Extract the vector-Jacobian product
        
        # Check that the gradient is valid
        assert isinstance(gradient, jnp.ndarray), "Gradient is not a JAX array"
        assert gradient.shape == P_jax.shape, "Gradient shape doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), "Gradient contains NaN values"
        
        # Check that we have non-zero gradients
        grad_abs_mean = jnp.mean(jnp.abs(gradient))
        assert grad_abs_mean > 0, "Gradient has zero mean absolute value"
        
    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")