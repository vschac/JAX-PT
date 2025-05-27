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

@pytest.fixture
def jpt():
    n_pad = int(0.5 * len(k))
    return JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad, warmup=False)

PK_PARAMS_JAX_COSMO = {'Omega_c': 0.25, 'Omega_b': 0.05, 'h': 0.7, 'n_s': 0.96, 'sigma8': 0.8, 'k_pivot': 0.05}

# Define JAXPT methods that have a standard signature for differentiation
# (take P, P_window, C_window)
JAXPT_METHODS_STD_SIGNATURE = [
    'one_loop_dd_bias_b3nl',
    'one_loop_dd_bias_lpt_NL',
    'IA_tt',
    'IA_mix',
    'IA_ta',
    'IA_ct',
    'gI_ct',
    'gI_ta',
    'gI_tt',
    'OV',
    'kPol'
]

# Expected number of output arrays (or 1 if single array) for each JAXPT method
JAXPT_METHOD_OUTPUT_COUNTS = {
    'one_loop_dd_bias_b3nl': 9,
    'one_loop_dd_bias_lpt_NL': 7,
    'IA_tt': 2,
    'IA_mix': 4,
    'IA_ta': 4,
    'IA_ct': 4,
    'gI_ct': 2,
    'gI_ta': 4,
    'gI_tt': 2,
    'OV': 1,
    'kPol': 3,
}

def custom_pk_generator_func(k_arr, Omega_m, A_s):
    """A simple custom power spectrum generator function for testing."""
    # P(k) = A_s * (k / k_pivot_fixed_to_1)^ (Omega_m - 1)
    return A_s * (k_arr ** (Omega_m - 1.0))

# Basic tests for default and specific diff_methods

def test_diff_default_vjp(jpt):
    """Test diff with default settings (vjp, jax-cosmo, Omega_c, one_loop_dd_bias_b3nl)."""
    # Default diff_method in JAXPT.diff is None, which raises error.
    # Assuming 'vjp' is the intended default to test here.
    # Function must be specified by name.
    grad_pk_param, primals_out = jpt.diff(
        pk_params=PK_PARAMS_JAX_COSMO.copy(),
        function='one_loop_dd_bias_b3nl', # Specify function by name
        diff_method='vjp' # Specify diff_method
    )
    
    assert primals_out is not None
    assert grad_pk_param is not None
    
    expected_outputs = JAXPT_METHOD_OUTPUT_COUNTS['one_loop_dd_bias_b3nl']
    assert isinstance(primals_out, tuple)
    assert len(primals_out) == expected_outputs
    for i, arr in enumerate(primals_out): # Added enumerate
        assert isinstance(arr, jnp.ndarray)
        if i == 7: # sig4 is the 8th element (index 7) and is scalar for one_loop_dd_bias_b3nl
            assert arr.shape == ()
        else:
            assert arr.shape == (len(k),)
        assert not jnp.any(jnp.isnan(arr))

    # grad_pk_param is already the computed gradient
    assert isinstance(grad_pk_param, jnp.ndarray) 
    assert grad_pk_param.shape == () # Scalar gradient for scalar pk_diff_param
    assert not jnp.isnan(grad_pk_param)

def test_diff_jvp(jpt):
    """Test diff with jvp method."""
    tangent_val = jnp.array(1.0) # Tangent for the scalar input pk_diff_param
    
    # Using IA_tt as the function to differentiate
    primals_out, tangents_out = jpt.diff(
        pk_params=PK_PARAMS_JAX_COSMO.copy(),
        pk_diff_param='h',
        function="IA_tt",
        diff_method='jvp',
        tangent=tangent_val
    )

    assert primals_out is not None
    assert tangents_out is not None
    
    expected_outputs = JAXPT_METHOD_OUTPUT_COUNTS['IA_tt']
    assert isinstance(primals_out, tuple)
    assert len(primals_out) == expected_outputs
    assert isinstance(tangents_out, tuple)
    assert len(tangents_out) == expected_outputs

    for i in range(expected_outputs):
        assert isinstance(primals_out[i], jnp.ndarray)
        assert primals_out[i].shape == (len(jpt.k_original),)
        assert not jnp.any(jnp.isnan(primals_out[i]))
        assert isinstance(tangents_out[i], jnp.ndarray)
        assert tangents_out[i].shape == (len(jpt.k_original),)
        assert not jnp.any(jnp.isnan(tangents_out[i]))

def test_diff_jacfwd(jpt):
    """Test diff with jacfwd method."""
    # Using kPol as the function to differentiate
    jacobian_out = jpt.diff(
        pk_params=PK_PARAMS_JAX_COSMO.copy(),
        pk_diff_param='n_s',
        function='kPol',
        diff_method='jacfwd'
    )

    assert jacobian_out is not None
    expected_outputs = JAXPT_METHOD_OUTPUT_COUNTS['kPol']
    assert isinstance(jacobian_out, tuple)
    assert len(jacobian_out) == expected_outputs
    for arr in jacobian_out:
        assert isinstance(arr, jnp.ndarray)
        assert arr.shape == ((len(k)),) # Jacobian of vector wrt scalar is vector
        assert not jnp.any(jnp.isnan(arr))

# Parameterized tests for all standard JAXPT methods

@pytest.mark.parametrize("method_name", JAXPT_METHODS_STD_SIGNATURE)
def test_diff_vjp_all_std_methods(jpt, method_name):
    """Parameterized test for VJP across standard JAXPT methods."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    
    grad_pk_param, primals_out = jpt.diff(
        pk_params=pk_params,
        pk_diff_param='Omega_c',
        function=method_name,
        P_window=P_window,
        C_window=C_window,
        diff_method='vjp'
    )

    expected_outputs = JAXPT_METHOD_OUTPUT_COUNTS[method_name]
    
    if expected_outputs == 1: 
        assert isinstance(primals_out, jnp.ndarray)
        assert primals_out.shape == (len(jpt.k_original),)
        assert not jnp.any(jnp.isnan(primals_out))
    else: 
        assert isinstance(primals_out, tuple)
        assert len(primals_out) == expected_outputs
        for i, arr in enumerate(primals_out): # Added enumerate
            assert isinstance(primals_out[i], jnp.ndarray)
            
            is_scalar_component = False
            if method_name == 'one_loop_dd_bias_b3nl' and i == 7: 
                is_scalar_component = True
            elif method_name == 'one_loop_dd_bias_lpt_NL' and i == 6: 
                is_scalar_component = True
            
            if is_scalar_component:
                assert primals_out[i].shape == ()
            else:
                assert primals_out[i].shape == (len(jpt.k_original),)
            assert not jnp.any(jnp.isnan(primals_out[i]))

    assert isinstance(grad_pk_param, jnp.ndarray) 
    assert grad_pk_param.shape == () 
    assert not jnp.isnan(grad_pk_param)

@pytest.mark.parametrize("method_name", JAXPT_METHODS_STD_SIGNATURE)
def test_diff_jvp_all_std_methods(jpt, method_name):
    """Parameterized test for JVP across standard JAXPT methods."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    tangent_val = jnp.array(1.0)
    
    # JAXPT.diff for jvp returns: tangents_out, primals_out (as per your diff method's current return)
    # If your diff method returns primal_out, jvp_result, then swap the unpacking
    tangents_out, primals_out = jpt.diff(
        pk_params=pk_params,
        pk_diff_param='h',
        function=method_name,
        P_window=P_window,
        C_window=C_window,
        diff_method='jvp',
        tangent=tangent_val
    )

    expected_outputs = JAXPT_METHOD_OUTPUT_COUNTS[method_name]

    if expected_outputs == 1:
        assert isinstance(primals_out, jnp.ndarray)
        assert primals_out.shape == (len(jpt.k_original),)
        assert not jnp.any(jnp.isnan(primals_out))
        assert isinstance(tangents_out, jnp.ndarray)
        assert tangents_out.shape == (len(jpt.k_original),) # Tangent for vector is vector
        assert not jnp.any(jnp.isnan(tangents_out))
    else:
        assert isinstance(primals_out, tuple)
        assert len(primals_out) == expected_outputs
        assert isinstance(tangents_out, tuple)
        assert len(tangents_out) == expected_outputs
        for i in range(expected_outputs):
            assert isinstance(primals_out[i], jnp.ndarray)
            assert isinstance(tangents_out[i], jnp.ndarray)
            
            is_scalar_component = False
            if method_name == 'one_loop_dd_bias_b3nl' and i == 7: 
                is_scalar_component = True
            elif method_name == 'one_loop_dd_bias_lpt_NL' and i == 6: 
                is_scalar_component = True
            
            if is_scalar_component:
                assert primals_out[i].shape == ()
                assert tangents_out[i].shape == () # Tangent for scalar is scalar
            else:
                assert primals_out[i].shape == (len(jpt.k_original),)
                assert tangents_out[i].shape == (len(jpt.k_original),)
            
            assert not jnp.any(jnp.isnan(primals_out[i]))
            assert not jnp.any(jnp.isnan(tangents_out[i]))

@pytest.mark.parametrize("method_name", JAXPT_METHODS_STD_SIGNATURE)
def test_diff_jacfwd_all_std_methods(jpt, method_name):
    """Parameterized test for Jacfwd across standard JAXPT methods."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    
    jacobian_out = jpt.diff(
        pk_params=pk_params,
        pk_diff_param='n_s',
        function=method_name,
        P_window=P_window,
        C_window=C_window,
        diff_method='jacfwd'
    )

    expected_outputs = JAXPT_METHOD_OUTPUT_COUNTS[method_name]

    if expected_outputs == 1:
        assert isinstance(jacobian_out, jnp.ndarray)
        # Jacobian of a vector output wrt a scalar input is a vector
        assert jacobian_out.shape == (len(jpt.k_original),) 
        assert not jnp.any(jnp.isnan(jacobian_out))
    else:
        assert isinstance(jacobian_out, tuple)
        assert len(jacobian_out) == expected_outputs
        for i, arr in enumerate(jacobian_out): # Added enumerate
            assert isinstance(arr, jnp.ndarray)
            
            is_scalar_component = False
            if method_name == 'one_loop_dd_bias_b3nl' and i == 7:
                is_scalar_component = True
            elif method_name == 'one_loop_dd_bias_lpt_NL' and i == 6:
                is_scalar_component = True

            if is_scalar_component:
                # Jacobian of a scalar output component wrt a scalar input is a scalar
                assert arr.shape == () 
            else:
                # Jacobian of a vector output component wrt a scalar input is a vector
                assert arr.shape == (len(jpt.k_original),)
            assert not jnp.any(jnp.isnan(arr))

# Tests for custom Pk generator

def test_diff_no_pk_params_uses_defaults_jax_cosmo(jpt):
    """Test that jax-cosmo uses default Pk params if none are provided."""
    # This test implicitly checks that no error occurs and output is valid
    # when pk_params is empty for jax-cosmo (should use internal defaults).
    primals_out, _ = jpt.diff(
        pk_method='jax-cosmo', # explicit for clarity
        pk_params={}, # Empty params
        pk_diff_param='Omega_c', # Omega_c has a default in jax-cosmo
        function='OV', # OV returns a single array
        diff_method='jvp'
    )
    assert isinstance(primals_out, jnp.ndarray)
    assert primals_out.shape == (len(jpt.k_original),)
    assert not jnp.any(jnp.isnan(primals_out))

