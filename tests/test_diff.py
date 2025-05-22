import pytest
import numpy as np
import jax.numpy as jnp
import jax
from jaxpt import FP_JAXPT as JAXPT
from jaxpt.diff_config import DiffConfig, DiffConfigDC
import os

# Load test data
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

@pytest.fixture
def base_config():
    config = DiffConfig()
    config.pk_params = {
        'Omega_c': 0.12,
        'Omega_b': 0.022,
        'h': 0.69,
        'n_s': 0.96,
        'sigma8': 0.8,
        'Omega_k': 0.0,
        'w0': -1.0,
        'wa': 0.0
    }
    config.function = 'one_loop_dd_bias_b3nl'
    return config


class TestDiffConfig:
    def test_valid_config(self, base_config):
        """Test that a valid configuration passes validation."""
        validated = base_config.build_and_validate()
        assert isinstance(validated, DiffConfigDC)
        assert validated.function == 'one_loop_dd_bias_b3nl'
        
    def test_invalid_pk_method(self, base_config):
        """Test that an invalid pk_generation_method raises an error."""
        base_config.pk_generation_method = 'invalid-method'
        with pytest.raises(ValueError, match="Invalid pk_generation_method"):
            base_config.build_and_validate()
    
    def test_invalid_pk_param(self, base_config):
        """Test that an invalid pk_param raises an error."""
        base_config.pk_params['invalid_param'] = 1.0
        with pytest.raises(ValueError):
            base_config.build_and_validate()
    
    def test_invalid_pk_diff_param(self, base_config):
        """Test that an invalid pk_diff_param raises an error."""
        base_config.pk_diff_param = 'invalid_param'
        with pytest.raises(ValueError):
            base_config.build_and_validate()
    
    def test_no_function(self, base_config):
        """Test that providing no function raises an error."""
        base_config.function = None
        with pytest.raises(ValueError, match="function must be provided"):
            base_config.build_and_validate()


class TestDifferentiation:
    def test_jacfwd_vector(self, jpt, base_config):
        """Test jacfwd with vector output."""
        base_config.diff_type = "vector"
        base_config.diff_method = "jacfwd"
        validated_config = base_config.build_and_validate()
        
        result = jpt.diff(validated_config)
        
        # Since one_loop_dd_bias_b3nl returns multiple arrays, check the first component
        if isinstance(result, tuple):
            first_component = result[0]
        else:
            first_component = result
            
        assert isinstance(first_component, jnp.ndarray)
        assert first_component.shape == k.shape
        assert not jnp.any(jnp.isnan(first_component))
    
    def test_jvp_vector(self, jpt, base_config):
        """Test jvp with vector output."""
        base_config.diff_type = "vector"
        base_config.diff_method = "jvp"
        base_config.tangent = jnp.array(1.0)
        validated_config = base_config.build_and_validate()
        
        primal, tangent = jpt.diff(validated_config)
        
        # Check the first component of each tuple
        if isinstance(primal, tuple):
            primal_first = primal[0]
            tangent_first = tangent[0]
        else:
            primal_first = primal
            tangent_first = tangent
            
        assert isinstance(primal_first, jnp.ndarray)
        assert isinstance(tangent_first, jnp.ndarray)
        assert primal_first.shape == k.shape
        assert tangent_first.shape == k.shape
    
    def test_vjp_scalar_with_reduction(self, jpt, base_config):
        """Test vjp with scalar output and reduction function."""
        base_config.diff_type = "scalar"
        base_config.diff_method = "vjp"
        base_config.reduction_func = lambda x: jnp.sum(x)
        validated_config = base_config.build_and_validate()
        
        primal, gradient = jpt.diff(validated_config)
        
        # For functions returning multiple terms, primal will be a tuple of scalars
        if isinstance(primal, tuple):
            assert all(not jnp.isnan(p) for p in primal)
        else:
            assert not jnp.isnan(primal)
            
        assert not jnp.isnan(gradient)
    
    def test_grad_with_reduction(self, jpt, base_config):
        """Test grad with reduction function."""
        base_config.diff_type = "scalar"
        base_config.diff_method = "grad"
        base_config.reduction_func = lambda x: jnp.sum(x)
        validated_config = base_config.build_and_validate()
        
        result = jpt.diff(validated_config)
        
        # Just verify the gradient is valid
        assert not jnp.isnan(result)
    
    def test_grad_without_reduction(self, jpt, base_config):
        """Test grad without reduction function raises an error."""
        base_config.diff_type = "scalar"
        base_config.diff_method = "grad"
        validated_config = base_config.build_and_validate()
        
        with pytest.raises(ValueError, match="reduction_func must be provided"):
            jpt.diff(validated_config)


@pytest.mark.parametrize("param", ['h', 'Omega_c', 'Omega_b', 'n_s', 'sigma8'])
def test_different_parameters(jpt, base_config, param):
    """Test differentiation with respect to different cosmological parameters."""
    base_config.pk_diff_param = param
    base_config.diff_type = "vector"
    base_config.diff_method = "jacfwd"
    validated_config = base_config.build_and_validate()
    
    result = jpt.diff(validated_config)
    
    # For multiple return values, check the first component
    if isinstance(result, tuple):
        first_component = result[0]
    else:
        first_component = result
        
    assert first_component.shape == k.shape
    assert not jnp.any(jnp.isnan(first_component))


@pytest.mark.parametrize("function", ['IA_tt', 'IA_mix', 'IA_ta', 'IA_ct', 'gI_tt', 'gI_ct', 'gI_ta', 'kPol', 'OV'])
def test_different_functions(jpt, base_config, function):
    """Test differentiation of different functions."""
    base_config.function = function
    base_config.diff_type = "vector"
    base_config.diff_method = "jacfwd"
    validated_config = base_config.build_and_validate()
    
    result = jpt.diff(validated_config)
    
    # Handle both tuple and array return types
    if isinstance(result, tuple):
        # Check each component in the tuple
        for component in result:
            if hasattr(component, 'shape'):  # Only check array-like objects
                assert not jnp.any(jnp.isnan(component))
    else:
        # Single array result
        assert not jnp.any(jnp.isnan(result))


def test_chi2_reduction(jpt, base_config):
    """Test a cosmology-relevant chi^2 reduction function."""
    # Use IA_tt which returns a simpler output (just two arrays)
    base_config.function = "IA_tt"
    
    # Create a chi-squared function that takes the output and reduces it
    def chi_squared(power_spectrum_tuple):
        """Mock chi^2 calculation comparing to 'data'"""
        # For IA_tt, extract just the first component (P_E)
        if isinstance(power_spectrum_tuple, tuple):
            power_spectrum = power_spectrum_tuple[0]
        else:
            power_spectrum = power_spectrum_tuple
            
        mock_data = power_spectrum * 1.05  # 5% difference
        mock_errors = power_spectrum * 0.1   # 10% errors
        return jnp.sum(((power_spectrum - mock_data) / mock_errors) ** 2)
    
    base_config.diff_type = "scalar"
    base_config.diff_method = "vjp"
    base_config.reduction_func = chi_squared
    validated_config = base_config.build_and_validate()
    
    chi2_value, gradient = jpt.diff(validated_config)
    
    assert not jnp.isnan(chi2_value)
    assert not jnp.isnan(gradient)
    # Chi2 should be positive
    assert chi2_value > 0


def test_custom_tangent_jvp(jpt, base_config):
    """Test JVP with different tangent vectors."""
    # Use IA_tt for simpler output structure
    base_config.function = "IA_tt"
    base_config.diff_type = "vector"
    base_config.diff_method = "jvp"
    
    # First with default tangent
    base_config.tangent = None
    default_config = base_config.build_and_validate()
    _, default_jvp = jpt.diff(default_config)
    
    # Then with custom tangent
    base_config.tangent = jnp.array(2.0)
    custom_config = base_config.build_and_validate()
    _, custom_jvp = jpt.diff(custom_config)
    
    # Extract the first component for comparison
    default_component = default_jvp[0] if isinstance(default_jvp, tuple) else default_jvp
    custom_component = custom_jvp[0] if isinstance(custom_jvp, tuple) else custom_jvp
    
    # The custom tangent should scale the JVP result
    ratio = custom_component / default_component
    assert jnp.allclose(ratio, 2.0, rtol=1e-5)


def test_windowing_effect(jpt, base_config):
    """Test the effect of window parameters on differentiation."""
    # Use IA_tt for simpler output structure
    base_config.function = "IA_tt"
    base_config.diff_type = "vector"
    base_config.diff_method = "jacfwd"
    
    # Without window
    base_config.P_window = None
    base_config.C_window = None
    no_window_config = base_config.build_and_validate()
    no_window_result = jpt.diff(no_window_config)
    
    # With window
    base_config.P_window = P_window
    base_config.C_window = C_window
    with_window_config = base_config.build_and_validate()
    with_window_result = jpt.diff(with_window_config)
    
    # Extract comparable components
    no_window_component = no_window_result[0] if isinstance(no_window_result, tuple) else no_window_result
    with_window_component = with_window_result[0] if isinstance(with_window_result, tuple) else with_window_result
    
    # Results should be different with windowing
    assert not jnp.allclose(no_window_component, with_window_component)


def test_vjp_vs_grad(jpt, base_config):
    """Compare vjp and grad methods for scalar differentiation."""
    # Use IA_tt for simpler output
    base_config.function = "IA_tt"
    base_config.diff_type = "scalar"
    
    # Create a reduction function that only works with the first component
    def sum_first_component(result):
        if isinstance(result, tuple):
            return jnp.sum(result[0])
        return jnp.sum(result)
    
    base_config.reduction_func = sum_first_component
    
    # Method 1: VJP
    base_config.diff_method = "vjp"
    vjp_config = base_config.build_and_validate()
    vjp_value, vjp_gradient = jpt.diff(vjp_config)
    
    # Method 2: grad
    base_config.diff_method = "grad" 
    grad_config = base_config.build_and_validate()
    grad_gradient = jpt.diff(grad_config)
    
    # The gradients should be close
    assert jnp.allclose(vjp_gradient, grad_gradient, rtol=1e-5)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])