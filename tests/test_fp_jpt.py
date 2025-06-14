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

def plot_comparison(term_name, component_name, jaxpt_result, fastpt_result):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy if needed
    j_arr = np.array(jaxpt_result)
    f_arr = np.array(fastpt_result)
    
    # Plot actual values
    plt.subplot(2, 1, 1)
    plt.loglog(k, j_arr, label='JAXPT')
    plt.loglog(k, f_arr, '--', label='FASTPT')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel(f'{component_name}')
    plt.title(f'{term_name} - {component_name}')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot relative difference
    plt.subplot(2, 1, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.abs((j_arr - f_arr) / f_arr)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)
    
    plt.loglog(k, rel_diff)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('Relative Difference |(JAXPT-FASTPT)/FASTPT|')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Add max difference info
    max_abs_diff = np.max(np.abs(j_arr - f_arr))
    max_rel_diff = np.max(rel_diff)
    plt.figtext(0.5, 0.01, f'Max abs diff: {max_abs_diff:.2e}, Max rel diff: {max_rel_diff:.2e}', 
                ha='center', bbox=dict(facecolor='lightgray', alpha=0.5))
    
    # Save figure
    plt.tight_layout()
    filename = f"{term_name}_{component_name.replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(f"../term_comparison_plots/{filename}")
    plt.close()

if __name__ == "__main__":
    from time import time
    import colorama
    from colorama import Fore, Style
    
    # Initialize colorama for colored terminal output
    colorama.init(autoreset=True)
    
    jpt = JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5 * len(k)))
    fpt = FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5 * len(k)))
    funcs = ["one_loop_dd_bias_b3nl", "one_loop_dd_bias_lpt_NL", "IA_tt", "IA_mix", "IA_ta", "IA_ct", "gI_ct", "gI_ta", "gI_tt", "kPol", "OV"]
    
    print("\n" + "="*80)
    print(f"{Style.BRIGHT}JAXPT vs FASTPT Comparison{Style.RESET_ALL}")
    print("="*80)
    
    all_success = True
    
    for func in funcs:
        print(f"\n{Style.BRIGHT}[Testing {func}]{Style.RESET_ALL}")
        print("-"*80)
        
        # Run both implementations and time them
        t0 = time()
        result = getattr(jpt, func)(P, P_window=P_window, C_window=C_window)
        t1 = time()
        r2 = getattr(fpt, func)(P, P_window=np.array([0.2,0.2]), C_window=C_window)
        t2 = time()
        
        # Print timing information
        jaxpt_time = t1 - t0
        fastpt_time = t2 - t1
        speedup = fastpt_time / jaxpt_time if jaxpt_time > 0 else float('inf')
        
        print(f"⏱️  JAXPT: {jaxpt_time:.4f}s | FASTPT: {fastpt_time:.4f}s | Speedup: {speedup:.2f}x")
        
        # For OV which returns a single array
        if func == "OV":
            close = np.allclose(result, r2)
            status = f"{Fore.GREEN}✓ MATCH" if close else f"{Fore.RED}✗ MISMATCH"
            print(f"\n{status}: {func}{Style.RESET_ALL}")
            
            if not close:
                all_success = False
                max_diff = np.max(np.abs(result - r2))
                rel_diff = np.max(np.abs((result - r2) / result))
                print(f"   Max absolute difference: {Fore.YELLOW}{max_diff:.2e}{Style.RESET_ALL}")
                print(f"   Max relative difference: {Fore.YELLOW}{rel_diff:.2e}{Style.RESET_ALL}")
        
        # For functions returning multiple arrays
        else:
            print(f"\n{Style.BRIGHT}Component-wise comparison:{Style.RESET_ALL}")
            all_match = True
            
            for i in range(len(result)):
                close = np.allclose(result[i], r2[i])
                component_status = f"{Fore.GREEN}✓" if close else f"{Fore.RED}✗"
                
                if not close:
                    all_success = False
                    all_match = False
                    max_diff = np.max(np.abs(result[i] - r2[i]))
                    rel_diff = np.max(np.abs((result[i] - r2[i]) / result[i]))
                    
                    print(f"   {component_status} Component {i}: {Fore.RED}MISMATCH{Style.RESET_ALL}")
                    print(f"      Max absolute difference: {Fore.YELLOW}{max_diff:.2e}{Style.RESET_ALL}")
                    print(f"      Max relative difference: {Fore.YELLOW}{rel_diff:.2e}{Style.RESET_ALL}")

                    # Plotting the comparison
                    # plot_comparison(func, f"Component {i}", result[i], r2[i])
                else:
                    print(f"   {component_status} Component {i}: {Fore.GREEN}MATCH{Style.RESET_ALL}")
            
            # Summary for this function
            overall_status = f"{Fore.GREEN}✓ ALL MATCH" if all_match else f"{Fore.RED}✗ SOME MISMATCH"
            print(f"\n{overall_status}{Style.RESET_ALL}")
    
    # Final summary
    print("\n" + "="*80)
    if all_success:
        print(f"{Fore.GREEN}{Style.BRIGHT}✓ ALL TESTS PASSED: JAXPT matches FASTPT for all functions{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}{Style.BRIGHT}✗ SOME TESTS FAILED: Check the details above{Style.RESET_ALL}")
    print("="*80 + "\n")


@pytest.fixture
def jpt():
    n_pad = int(0.5 * len(k))
    return JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad, warmup=True)

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
