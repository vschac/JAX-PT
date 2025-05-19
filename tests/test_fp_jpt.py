import pytest
import numpy as np
import jax.numpy as jnp
from jaxpt import FP_JAXPT as JAXPT
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
    from time import time
    jpt = JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5 * len(k)))
    funcs = ["one_loop_dd_bias_b3nl", "one_loop_dd_bias_lpt_NL", "IA_tt", "IA_mix", "IA_ta", "IA_ct", "gI_ct", "gI_ta", "gI_tt", "kPol"]
    for func in funcs:
        start = time()
        result = getattr(jpt, func)(P, P_window=P_window, C_window=C_window)
        end = time()
        print(f"{func} took {end - start:.4f} seconds")
    

@pytest.fixture
def jpt():
    n_pad = int(0.5 * len(k))
    return JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad, warmup=False)

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

@pytest.mark.parametrize("term",
                         ["P_E", "P_B", "P_A", "P_DEE", "P_DBB", "P_deltaE1", "P_0E0E", "P_0B0B",
                         "P_s2E","P_s20E", "P_s2E2", "P_d2E",
                         "P_d20E", "P_d2E2", "P_kP1", "P_kP2", "P_kP3", "P_OV", "P_0EtE",
                         "P_E2tE", "P_tEtE", "Pd1d2", "Pd2d2", "Pd1s2", "Pd2s2", "Ps2s2", "sig4",
                         "Pb1L_b2L", "Pb2L", "Pb2L_2", "P_d2tE", "P_s2tE",
                         "P_Btype2", "P_deltaE2", "sig3nl", "Pb1L", "Pb1L_2", "P_0tE", "P_1loop",
                        ])
def test_every_term(jpt, fpt, term):
    handler = FPTHandler(fpt, P=P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    fast = handler.get(term)
    jaax = jpt.get(term, P, P_window=P_window, C_window=C_window)
    if not np.allclose(fast, jaax):
        print(f"Max difference for {term}: {np.max(np.abs(fast - jaax))}")
        print(f"Relative difference for {term}: {np.max(np.abs((fast - jaax) / fast))}")
        assert np.allclose(fast, jaax), f"Arrays for {term} are not equal"
    assert np.allclose(fast, jaax), f"Arrays for {term} are not equal"

########## Differentiability Tests ##########
def test_IA_mix_diff(jpt):
    def compute_ia_mix(P):
        return jpt.IA_mix(P, P_window=P_window, C_window=C_window)

    result_value, vjp_fn = vjp(compute_ia_mix, P)
    tangent_vectors = tuple(jnp.ones_like(result) for result in result_value)
    # Compute the gradient with respect to P
    gradient = vjp_fn(tangent_vectors)[0]  # [0] because vjp_fn returns a tuple

    assert gradient is not None, "Gradient should not be None"
    assert gradient.shape == P.shape, f"Gradient shape {gradient.shape} does not match input shape {P.shape}"
    assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"
    assert not np.any(np.isnan(gradient)), "Gradient contains NaN values"
    assert not np.any(np.isinf(gradient)), "Gradient contains infinite values"

def test_IA_tt_diff(jpt):
    def compute_ia_tt(P):
        return jpt.IA_tt(P, C_window=C_window)

    result_value, vjp_fn = vjp(compute_ia_tt, P)
    tangent_vectors = tuple(jnp.ones_like(result) for result in result_value)
    # Compute the gradient with respect to P
    gradient = vjp_fn(tangent_vectors)[0]  # [0] because vjp_fn returns a tuple

    assert gradient is not None, "Gradient should not be None"
    assert gradient.shape == P.shape, f"Gradient shape {gradient.shape} does not match input shape {P.shape}"
    assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"
    assert not np.any(np.isnan(gradient)), "Gradient contains NaN values"
    assert not np.any(np.isinf(gradient)), "Gradient contains infinite values"

def test_IA_ta_diff(jpt):
    def compute_ia_ta(P):
        return jpt.IA_ta(P, C_window=C_window)

    result_value, vjp_fn = vjp(compute_ia_ta, P)
    tangent_vectors = tuple(jnp.ones_like(result) for result in result_value)
    # Compute the gradient with respect to P
    gradient = vjp_fn(tangent_vectors)[0]  # [0] because vjp_fn returns a tuple

    assert gradient is not None, "Gradient should not be None"
    assert gradient.shape == P.shape, f"Gradient shape {gradient.shape} does not match input shape {P.shape}"
    assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"
    assert not np.any(np.isnan(gradient)), "Gradient contains NaN values"
    assert not np.any(np.isinf(gradient)), "Gradient contains infinite values"

def test_IA_ct_diff(jpt):
    def compute_ia_ct(P):
        return jpt.IA_ct(P, C_window=C_window)

    result_value, vjp_fn = vjp(compute_ia_ct, P)
    tangent_vectors = tuple(jnp.ones_like(result) for result in result_value)
    # Compute the gradient with respect to P
    gradient = vjp_fn(tangent_vectors)[0]  # [0] because vjp_fn returns a tuple

    assert gradient is not None, "Gradient should not be None"
    assert gradient.shape == P.shape, f"Gradient shape {gradient.shape} does not match input shape {P.shape}"
    assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"
    assert not np.any(np.isnan(gradient)), "Gradient contains NaN values"
    assert not np.any(np.isinf(gradient)), "Gradient contains infinite values"

def test_gI_ct_diff(jpt):
    def compute_gI_ct(P):
        return jpt.gI_ct(P, C_window=C_window)

    result_value, vjp_fn = vjp(compute_gI_ct, P)
    tangent_vectors = tuple(jnp.ones_like(result) for result in result_value)
    # Compute the gradient with respect to P
    gradient = vjp_fn(tangent_vectors)[0]  # [0] because vjp_fn returns a tuple

    assert gradient is not None, "Gradient should not be None"
    assert gradient.shape == P.shape, f"Gradient shape {gradient.shape} does not match input shape {P.shape}"
    assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"
    assert not np.any(np.isnan(gradient)), "Gradient contains NaN values"
    assert not np.any(np.isinf(gradient)), "Gradient contains infinite values"

def test_gI_ta_diff(jpt):
    def compute_gI_ta(P):
        return jpt.gI_ta(P, C_window=C_window)

    result_value, vjp_fn = vjp(compute_gI_ta, P)
    tangent_vectors = tuple(jnp.ones_like(result) for result in result_value)
    # Compute the gradient with respect to P
    gradient = vjp_fn(tangent_vectors)[0]  # [0] because vjp_fn returns a tuple

    assert gradient is not None, "Gradient should not be None"
    assert gradient.shape == P.shape, f"Gradient shape {gradient.shape} does not match input shape {P.shape}"
    assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"
    assert not np.any(np.isnan(gradient)), "Gradient contains NaN values"
    assert not np.any(np.isinf(gradient)), "Gradient contains infinite values"

def test_gI_tt_diff(jpt):
    def compute_gI_tt(P):
        return jpt.gI_tt(P, C_window=C_window)

    result_value, vjp_fn = vjp(compute_gI_tt, P)
    tangent_vectors = tuple(jnp.ones_like(result) for result in result_value)
    # Compute the gradient with respect to P
    gradient = vjp_fn(tangent_vectors)[0]  # [0] because vjp_fn returns a tuple

    assert gradient is not None, "Gradient should not be None"
    assert gradient.shape == P.shape, f"Gradient shape {gradient.shape} does not match input shape {P.shape}"
    assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"
    assert not np.any(np.isnan(gradient)), "Gradient contains NaN values"
    assert not np.any(np.isinf(gradient)), "Gradient contains infinite values"

def test_kPol_diff(jpt):
    def compute_kPol(P):
        return jpt.kPol(P, C_window=C_window)

    result_value, vjp_fn = vjp(compute_kPol, P)
    tangent_vectors = tuple(jnp.ones_like(result) for result in result_value)
    # Compute the gradient with respect to P
    gradient = vjp_fn(tangent_vectors)[0]  # [0] because vjp_fn returns a tuple

    assert gradient is not None, "Gradient should not be None"
    assert gradient.shape == P.shape, f"Gradient shape {gradient.shape} does not match input shape {P.shape}"
    assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"
    assert not np.any(np.isnan(gradient)), "Gradient contains NaN values"
    assert not np.any(np.isinf(gradient)), "Gradient contains infinite values"

def test_b3nl_diff(jpt):
    def compute_b3nl(P):
        return jpt.one_loop_dd_bias_b3nl(P, P_window=P_window, C_window=C_window)

    result_value, vjp_fn = vjp(compute_b3nl, P)
    tangent_vectors = tuple(jnp.ones_like(result) for result in result_value)
    # Compute the gradient with respect to P
    gradient = vjp_fn(tangent_vectors)[0]  # [0] because vjp_fn returns a tuple

    assert gradient is not None, "Gradient should not be None"
    assert gradient.shape == P.shape, f"Gradient shape {gradient.shape} does not match input shape {P.shape}"
    assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"
    assert not np.any(np.isnan(gradient)), "Gradient contains NaN values"
    assert not np.any(np.isinf(gradient)), "Gradient contains infinite values"

def test_lpt_NL_diff(jpt):
    def compute_lpt_NL(P):
        return jpt.one_loop_dd_bias_lpt_NL(P, P_window=P_window, C_window=C_window)

    result_value, vjp_fn = vjp(compute_lpt_NL, P)
    tangent_vectors = tuple(jnp.ones_like(result) for result in result_value)
    # Compute the gradient with respect to P
    gradient = vjp_fn(tangent_vectors)[0]  # [0] because vjp_fn returns a tuple

    assert gradient is not None, "Gradient should not be None"
    assert gradient.shape == P.shape, f"Gradient shape {gradient.shape} does not match input shape {P.shape}"
    assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"
    assert not np.any(np.isnan(gradient)), "Gradient contains NaN values"
    assert not np.any(np.isinf(gradient)), "Gradient contains infinite values"

@pytest.mark.parametrize("term",
                         ["P_E", "P_B", "P_A", "P_DEE", "P_DBB", "P_deltaE1", "P_0E0E", "P_0B0B",
                         "P_s2E","P_s20E", "P_s2E2", "P_d2E",
                         "P_d20E", "P_d2E2", "P_kP1", "P_kP2", "P_kP3", "P_OV", "P_0EtE",
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
        _, vjp_fun = vjp(get_term, P_jax)
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
