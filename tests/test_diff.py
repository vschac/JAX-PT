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
    'one_loop_dd',
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
    'one_loop_dd': 2,
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
        pk_method='jax-cosmo', 
        pk_params=PK_PARAMS_JAX_COSMO.copy(),
        pk_diff_param='Omega_c',
        function='one_loop_dd_bias_b3nl', # Specify function by name
        diff_method='vjp' # Specify diff_method
    )
    
    assert primals_out is not None
    assert grad_pk_param is not None
    
    expected_outputs = JAXPT_METHOD_OUTPUT_COUNTS['one_loop_dd_bias_b3nl']
    assert isinstance(primals_out, tuple)
    assert len(primals_out) == expected_outputs -2 #Accounts for the removed first two elements
    for i, arr in enumerate(primals_out): # Added enumerate
        assert isinstance(arr, jnp.ndarray)
        if i == 5: # sig4 is the 8th element (index 7) and is scalar for one_loop_dd_bias_b3nl
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
        pk_method='jax-cosmo', 
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
        pk_method='jax-cosmo', 
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
        pk_method='jax-cosmo', 
        pk_params=pk_params,
        pk_diff_param='Omega_c',
        function=method_name,
        P_window=P_window,
        C_window=C_window,
        diff_method='vjp'
    )

    expected_outputs = JAXPT_METHOD_OUTPUT_COUNTS[method_name]
    if method_name == 'one_loop_dd_bias_b3nl': expected_outputs -= 2 # Adjust for the removed first two elements

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
            if method_name == 'one_loop_dd_bias_b3nl' and i == 5: 
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
        pk_method='jax-cosmo', 
        pk_params=pk_params,
        pk_diff_param='h',
        function=method_name,
        P_window=P_window,
        C_window=C_window,
        diff_method='jvp',
        tangent=tangent_val
    )

    expected_outputs = JAXPT_METHOD_OUTPUT_COUNTS[method_name]
    if method_name == 'one_loop_dd_bias_b3nl': expected_outputs -= 2 # Adjust for the removed first two elements

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
            if method_name == 'one_loop_dd_bias_b3nl' and i == 5: 
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
        pk_method='jax-cosmo', 
        pk_params=pk_params,
        pk_diff_param='n_s',
        function=method_name,
        P_window=P_window,
        C_window=C_window,
        diff_method='jacfwd'
    )

    expected_outputs = JAXPT_METHOD_OUTPUT_COUNTS[method_name]
    if method_name == 'one_loop_dd_bias_b3nl': expected_outputs -= 2 # Adjust for the removed first two elements

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
            if method_name == 'one_loop_dd_bias_b3nl' and i == 5:
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


##### Tests for multi param diff #####
def test_multi_param_diff_basic_jacfwd(jpt):
    """Test multi_param_diff with jacfwd for multiple parameters."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    pk_diff_params = ['Omega_c', 'h', 'n_s']
    
    result = jpt.multi_param_diff(
        pk_method='jax-cosmo',
        pk_params=pk_params,
        pk_diff_params=pk_diff_params,
        function='one_loop_dd',
        P_window=P_window,
        C_window=C_window,
        diff_method='jacfwd'
    )
    
    # Should return nested dict for tuple outputs
    assert isinstance(result, dict)
    assert len(result) == 2  # one_loop_dd returns (P_1loop, Ps)
    
    # Check structure
    assert 'P_1loop' in result
    assert 'Ps' in result
    
    for output_name in ['P_1loop', 'Ps']:
        assert isinstance(result[output_name], dict)
        assert set(result[output_name].keys()) == set(pk_diff_params)
        
        for param in pk_diff_params:
            deriv = result[output_name][param]
            assert isinstance(deriv, jnp.ndarray)
            assert deriv.shape == (len(jpt.k_original),)
            assert not jnp.any(jnp.isnan(deriv))


def test_multi_param_diff_basic_jacrev(jpt):
    """Test multi_param_diff with jacrev for multiple parameters."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    pk_diff_params = ['Omega_c', 'sigma8']
    
    result = jpt.multi_param_diff(
        pk_method='jax-cosmo',
        pk_params=pk_params,
        pk_diff_params=pk_diff_params,
        function='OV',  # Single output function
        diff_method='jacrev'
    )
    
    # Should return simple dict for single output
    assert isinstance(result, dict)
    assert set(result.keys()) == set(pk_diff_params)
    
    for param in pk_diff_params:
        deriv = result[param]
        assert isinstance(deriv, jnp.ndarray)
        assert deriv.shape == (len(jpt.k_original),)
        assert not jnp.any(jnp.isnan(deriv))


def test_multi_param_diff_output_indices_int(jpt):
    """Test multi_param_diff with output_indices as integer."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    pk_diff_params = ['Omega_c', 'h', 'n_s']
    
    # Get only first output
    result = jpt.multi_param_diff(
        pk_method='jax-cosmo',
        pk_params=pk_params,
        pk_diff_params=pk_diff_params,
        function='IA_mix',  # Returns 4 outputs
        output_indices=0,
        diff_method='jacfwd'
    )
    
    # Should return simple dict when selecting single output
    assert isinstance(result, dict)
    assert set(result.keys()) == set(pk_diff_params)
    
    for param in pk_diff_params:
        deriv = result[param]
        assert isinstance(deriv, jnp.ndarray)
        assert deriv.shape == (len(jpt.k_original),)
        assert not jnp.any(jnp.isnan(deriv))


def test_multi_param_diff_output_indices_list(jpt):
    """Test multi_param_diff with output_indices as list."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    pk_diff_params = ['Omega_c', 'h']
    
    # Get outputs 0 and 2
    result = jpt.multi_param_diff(
        pk_method='jax-cosmo',
        pk_params=pk_params,
        pk_diff_params=pk_diff_params,
        function='IA_mix',  # Returns 4 outputs
        output_indices=[0, 2],
        diff_method='jacfwd'
    )
    
    # Should return nested dict with only selected outputs
    assert isinstance(result, dict)
    assert len(result) == 2  # Only 2 outputs selected
    
    output_names = jpt._get_output_names('IA_mix')
    assert output_names[0] in result
    assert output_names[2] in result
    assert output_names[1] not in result  # Not selected
    assert output_names[3] not in result  # Not selected


def test_multi_param_diff_single_param(jpt):
    """Test multi_param_diff with single parameter (edge case)."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    
    result = jpt.multi_param_diff(
        pk_method='jax-cosmo',
        pk_params=pk_params,
        pk_diff_params=['Omega_c'],  # Single parameter
        function='one_loop_dd',
        diff_method='jacfwd'
    )
    
    assert isinstance(result, dict)
    assert len(result) == 2  # one_loop_dd returns 2 outputs
    
    for output_name in ['P_1loop', 'Ps']:
        assert 'Omega_c' in result[output_name]
        assert result[output_name]['Omega_c'].shape == (len(jpt.k_original),)


# Error handling tests

def test_multi_param_diff_missing_function(jpt):
    """Test multi_param_diff raises error when function is not specified."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    
    with pytest.raises(ValueError, match="No function provided"):
        jpt.multi_param_diff(
            pk_method='jax-cosmo',
            pk_params=pk_params,
            pk_diff_params=['Omega_c'],
            function=None
        )


def test_multi_param_diff_invalid_pk_diff_params(jpt):
    """Test multi_param_diff raises error for invalid pk_diff_params."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    
    # Not a list
    with pytest.raises(ValueError, match="pk_diff_params must be a list"):
        jpt.multi_param_diff(
            pk_method='jax-cosmo',
            pk_params=pk_params,
            pk_diff_params='Omega_c',  # String instead of list
            function='one_loop_dd'
        )


def test_multi_param_diff_invalid_pk_params(jpt):
    """Test multi_param_diff raises error for invalid pk_params."""
    # Not a dict
    with pytest.raises(ValueError, match="pk_params must be a dictionary"):
        jpt.multi_param_diff(
            pk_method='jax-cosmo',
            pk_params=['invalid'],  # List instead of dict
            pk_diff_params=['Omega_c'],
            function='one_loop_dd'
        )


def test_multi_param_diff_missing_param_values(jpt):
    """Test multi_param_diff raises error when parameter values are missing."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    del pk_params['h']  # Remove required parameter
    
    with pytest.raises(ValueError, match="Missing parameter values for: h"):
        jpt.multi_param_diff(
            pk_method='jax-cosmo',
            pk_params=pk_params,
            pk_diff_params=['Omega_c', 'h'],  # h is missing
            function='one_loop_dd'
        )


def test_multi_param_diff_invalid_diff_method(jpt):
    """Test multi_param_diff raises error for invalid diff_method."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    
    with pytest.raises(ValueError, match="Unsupported differentiation method"):
        jpt.multi_param_diff(
            pk_method='jax-cosmo',
            pk_params=pk_params,
            pk_diff_params=['Omega_c'],
            function='one_loop_dd',
            diff_method='vjp'  # Not supported for multi_param_diff
        )


# Parameterized tests for all standard JAXPT methods

@pytest.mark.parametrize("method_name", JAXPT_METHODS_STD_SIGNATURE)
@pytest.mark.parametrize("diff_method", ['jacfwd', 'jacrev'])
def test_multi_param_diff_all_methods(jpt, method_name, diff_method):
    """Parameterized test for multi_param_diff across all standard JAXPT methods."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    pk_diff_params = ['Omega_c', 'h', 'n_s']
    
    result = jpt.multi_param_diff(
        pk_method='jax-cosmo',
        pk_params=pk_params,
        pk_diff_params=pk_diff_params,
        function=method_name,
        P_window=P_window,
        C_window=C_window,
        diff_method=diff_method
    )
    
    expected_outputs = JAXPT_METHOD_OUTPUT_COUNTS[method_name]
    if method_name == 'one_loop_dd_bias_b3nl':
        expected_outputs -= 2  # Adjust for removed first two elements
    
    if expected_outputs == 1:
        # Single output - simple dict
        assert isinstance(result, dict)
        assert set(result.keys()) == set(pk_diff_params)
        
        for param in pk_diff_params:
            deriv = result[param]
            assert isinstance(deriv, jnp.ndarray)
            assert deriv.shape == (len(jpt.k_original),)
            assert not jnp.any(jnp.isnan(deriv))
    else:
        # Multiple outputs - nested dict
        assert isinstance(result, dict)
        output_names = jpt._get_output_names(method_name)
        assert len(result) == expected_outputs
        
        for i, output_name in enumerate(output_names[:expected_outputs]):
            assert output_name in result
            assert isinstance(result[output_name], dict)
            assert set(result[output_name].keys()) == set(pk_diff_params)
            
            for param in pk_diff_params:
                deriv = result[output_name][param]
                assert isinstance(deriv, jnp.ndarray)
                
                # Check for scalar outputs
                is_scalar_component = False
                if method_name == 'one_loop_dd_bias_b3nl' and i == 5:
                    is_scalar_component = True
                elif method_name == 'one_loop_dd_bias_lpt_NL' and i == 6:
                    is_scalar_component = True
                
                if is_scalar_component:
                    assert deriv.shape == ()
                else:
                    assert deriv.shape == (len(jpt.k_original),)
                
                assert not jnp.any(jnp.isnan(deriv))


@pytest.mark.parametrize("method_name", JAXPT_METHODS_STD_SIGNATURE)
def test_multi_param_diff_output_indices_all_methods(jpt, method_name):
    """Test output_indices functionality across all methods."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    pk_diff_params = ['Omega_c', 'h']
    
    expected_outputs = JAXPT_METHOD_OUTPUT_COUNTS[method_name]
    if method_name == 'one_loop_dd_bias_b3nl':
        expected_outputs -= 2
    
    if expected_outputs > 1:
        # Test selecting first output only
        result = jpt.multi_param_diff(
            pk_method='jax-cosmo',
            pk_params=pk_params,
            pk_diff_params=pk_diff_params,
            function=method_name,
            output_indices=0,
            diff_method='jacfwd'
        )
        
        # Should return simple dict
        assert isinstance(result, dict)
        assert set(result.keys()) == set(pk_diff_params)
        
        # Test selecting multiple outputs
        if expected_outputs > 2:
            result = jpt.multi_param_diff(
                pk_method='jax-cosmo',
                pk_params=pk_params,
                pk_diff_params=pk_diff_params,
                function=method_name,
                output_indices=[0, expected_outputs-1],  # First and last
                diff_method='jacfwd'
            )
            
            assert isinstance(result, dict)
            assert len(result) == 2  # Only 2 outputs selected


def test_multi_param_diff_comparison_with_single_diff(jpt):
    """Test that multi_param_diff gives same results as multiple single diff calls."""
    pk_params = PK_PARAMS_JAX_COSMO.copy()
    pk_diff_params = ['Omega_c', 'h']
    function = 'one_loop_dd'
    
    # Get results using multi_param_diff
    multi_result = jpt.multi_param_diff(
        pk_method='jax-cosmo',
        pk_params=pk_params,
        pk_diff_params=pk_diff_params,
        function=function,
        diff_method='jacfwd',
        output_indices=0  # Only first output for comparison
    )
    
    # Get results using single diff calls
    single_results = {}
    for param in pk_diff_params:
        single_result = jpt.diff(
            pk_method='jax-cosmo',
            pk_params=pk_params,
            pk_diff_param=param,
            function=function,
            diff_method='jacfwd'
        )
        # Extract first output if tuple
        if isinstance(single_result, tuple):
            single_results[param] = single_result[0]
        else:
            single_results[param] = single_result
    
    # Compare results
    for param in pk_diff_params:
        assert jnp.allclose(multi_result[param], single_results[param], 
                           rtol=1e-10, atol=1e-12)