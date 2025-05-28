import pytest
import numpy as np
import jax.numpy as jnp
from jaxpt import FP_JAXPT as JAXPT
import os
from fastpt import FASTPT, FPTHandler
from jax import vjp
import jax 
from time import time

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
d = np.loadtxt(data_path)
P = d[:, 1]
k = d[:, 0]
P_window = jnp.array([0.2, 0.2])
C_window = 0.75


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
    return JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad, warmup=False)

@pytest.fixture
def fpt():
    n_pad = int(0.5 * len(k))
    return FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)

############## Equality Tests ##############
## While they do test the same thing as the benchmarks, this will test whatever version of fastpt is installed specifically
## (If something is changed in fastpt, its effects will be shown here)
def test_one_loop_dd_bias_b3nl(jpt, fpt):
    old = fpt.one_loop_dd_bias_b3nl(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.one_loop_dd_bias_b3nl(P, P_window=P_window, C_window=C_window)
    if len(old) == len(new):
        for i in range(len(old)):
            assert np.allclose(old[i], new[i]), f"Arrays at index {i} are not equal"
    else:
        assert False, f"Arrays are not the same length, old: {len(old)} new: {len(new)}"

def test_one_loop_dd_bias_lpt_NL(jpt, fpt):
    old = fpt.one_loop_dd_bias_lpt_NL(P, P_window=np.array([0.2, 0.2]), C_window=C_window)
    new = jpt.one_loop_dd_bias_lpt_NL(P, P_window=P_window, C_window=C_window)
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


##### Timing Tests #####
@pytest.fixture
def jpt_warm():
    n_pad = int(0.5 * len(k))
    jpt = JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad, warmup=True)
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
