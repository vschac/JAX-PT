"""
This example demonstrates the key JAX-PT functionalities:
1. Basic function call to a Fast-PT mirrored function (IA_tt)
2. Single parameter differentiation
3. Multi-parameter differentiation with timing
"""

import jax.numpy as jnp
import numpy as np
import time
from jaxpt import JAXPT
from jaxpt.device_utils import device_info

data = np.loadtxt('Pk_test.dat')
P_linear = jnp.array(data[:, 1])
k = jnp.array(data[:, 0])

def main():
    """Simple demonstration of JAX-PT main features"""
    print("JAX-PT Simple Example")
    print("="*50)
    
    # Show device information
    print("\nDevice Information:")
    device_info()

    # Initialize JAX-PT with moderate warmup for better performance
    print("\nInitializing JAX-PT...")
    jpt = JAXPT(k, low_extrap=-5, high_extrap=3, warmup="moderate")
    print(f"Device: {jpt.get_device_info()['device']}")

    # Window parameters for numerical stability
    window_params = {"P_window": jnp.array([0.2, 0.2]), "C_window": 0.5}
    
    # ================================================================
    # 1. BASIC INTRINSIC ALIGNMENT CALCULATION
    # ================================================================
    print(f"\n1. INTRINSIC ALIGNMENT CALCULATION")
    print("-" * 40)
    
    t0 = time.time()
    P_E, P_B = jpt.IA_tt(P_linear, **window_params)
    t1 = time.time()
    
    print(f"✓ IA_tt computation: {t1-t0:.3f}s")
    print(f"  P_E shape: {P_E.shape}, range: {P_E.min():.2e} to {P_E.max():.2e}")
    print(f"  P_B shape: {P_B.shape}, range: {P_B.min():.2e} to {P_B.max():.2e}")
    
    # ================================================================
    # 2. SINGLE PARAMETER DIFFERENTIATION
    # ================================================================
    print(f"\n2. SINGLE PARAMETER DIFFERENTIATION") 
    print("-" * 40)
    
    # Define cosmological parameters for jax-cosmo
    cosmo_params = {
        'Omega_c': 0.12,
        'Omega_b': 0.022,
        'h': 0.7,
        'n_s': 0.96,
        'sigma8': 0.8
    }
    
    print("Computing dIA_tt/dOmega_c using automatic differentiation...")
    
    try:
        t0 = time.time()
        
        # Compute derivative of IA_tt with respect to Omega_c
        dP_E_dOmega_c, dP_B_dOmega_c = jpt.diff(
            pk_method='jax-cosmo',
            pk_params=cosmo_params,
            pk_diff_param='Omega_c',
            function='IA_tt',
            diff_method='jacfwd',
            **window_params
        )
        
        t1 = time.time()
        print(f"✓ Single parameter diff: {t1-t0:.3f}s")
        print(f"  dP_E/dOmega_c shape: {dP_E_dOmega_c.shape}")
        print(f"  dP_B/dOmega_c shape: {dP_B_dOmega_c.shape}")
        print(f"  dP_E/dOmega_c range: {dP_E_dOmega_c.min():.2e} to {dP_E_dOmega_c.max():.2e}")
        
    except ImportError:
        print("!! jax-cosmo not available, skipping differentiation examples")
        print("Install with: pip install jax-cosmo")
        return
    
    # ================================================================
    # 3. MULTI-PARAMETER DIFFERENTIATION WITH TIMING
    # ================================================================
    print(f"\n3. MULTI-PARAMETER DIFFERENTIATION")
    print("-" * 40)
    
    # Parameters to differentiate with respect to
    diff_params = ['Omega_c', 'sigma8', 'n_s', 'h']
    print(f"Computing derivatives w.r.t. {len(diff_params)} parameters: {diff_params}")
    
    t0 = time.time()
    
    # Compute all derivatives at once using multi_param_diff
    derivatives = jpt.multi_param_diff(
        pk_method='jax-cosmo',
        pk_params=cosmo_params, 
        pk_diff_params=diff_params,
        function='IA_tt',
        diff_method='jacfwd',
        **window_params
    )
    
    t1 = time.time()
    multi_diff_time = t1 - t0
    
    print(f"✓ Multi-parameter diff: {multi_diff_time:.3f}s")
    print(f"  Computed {len(diff_params)} derivatives for 2 outputs simultaneously")
    
    # Show the structure of results
    print(f"\nDerivative structure:")
    for output_name, param_dict in derivatives.items():
        print(f"  {output_name}:")
        for param_name, deriv_array in param_dict.items():
            print(f"    d{output_name}/d{param_name}: shape {deriv_array.shape}")
    
        # Compare timing: multi-diff vs individual diffs
    print(f"\nTiming comparison:")
    print(f"  Multi-parameter diff (4 params): {multi_diff_time:.3f}s")
    
    # Time ALL individual derivatives for actual comparison
    print(f"  Computing individual derivatives for comparison...")
    individual_total_time = 0
    individual_times = []
    
    for i, param in enumerate(diff_params):
        print(f"    Computing derivative {i+1}/{len(diff_params)}: d/d{param}")
        t0 = time.time()
        _ = jpt.diff('jax-cosmo', cosmo_params, param, 'IA_tt', 
                    diff_method='jacfwd', **window_params)
        t1 = time.time()
        param_time = t1 - t0
        individual_times.append(param_time)
        individual_total_time += param_time
        print(f"      Time: {param_time:.3f}s")
    
    print(f"  Individual diffs (actual total for 4 params): {individual_total_time:.3f}s")
    print(f"  Speedup from multi-diff: {individual_total_time/multi_diff_time:.1f}x")
    
    print(f"\n" + "="*50)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"\nKey results:")
    print(f"• IA_tt computation: Fast intrinsic alignment power spectra")
    print(f"• Single parameter differentiation: Automatic derivatives w.r.t. cosmology")
    print(f"• Multi-parameter differentiation: {individual_total_time/multi_diff_time:.1f}x faster than individual")
    print(f"• All computations are GPU-accelerated (if available)")

if __name__ == "__main__":
    main()