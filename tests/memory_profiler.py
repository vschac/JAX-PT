import gc
import numpy as np
import psutil
import os
import tracemalloc
from time import time
from fastpt import FASTPT, FPTHandler
import random
from classy import Class

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

def generate_power_spectrum(omega_cdm=0.12, h=0.67, omega_b=0.022, z=0.0, k_min=1e-4, k_max=1.0, n_points=2000):
    params = {
        'output': 'mPk',
        'P_k_max_1/Mpc': k_max * 1.1,
        'z_max_pk': z,
        'h': h,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm
    }

    # Wavenumbers at which to compute P(k)
    k_values = np.logspace(np.log10(k_min), np.log10(k_max), n_points)

    # Initialize CLASS
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    # Compute the power spectrum
    pk_values = np.array([cosmo.pk(k, z) for k in k_values])

    # Clean up CLASS instance
    cosmo.struct_cleanup()
    cosmo.empty()
    
    return k_values, pk_values


def random_cosmology():
    """Generate random cosmological parameters within reasonable ranges"""
    return {
        'omega_cdm': random.uniform(0.1, 0.14),     # Vary CDM density
        'h': random.uniform(0.65, 0.75),            # Vary Hubble parameter
        'omega_b': random.uniform(0.02, 0.025),     # Vary baryon density
        'z': random.uniform(0.0, 1.0)               # Vary redshift
    }

def profile_single_function(fpt, func_name, pk, iterations=10):
    """Profile memory usage of a single FASTPT function"""
    print(f"\nProfiling {func_name}...")
    
    # Force garbage collection
    gc.collect()
    start_mem = get_memory_usage()
    print(f"  Initial memory: {start_mem:.2f} MB")
    
    # Start tracemalloc
    tracemalloc.start()
    
    # Get function
    func = getattr(fpt, func_name)
    
    # Run function multiple times to ensure any caching effects
    results = []
    for i in range(iterations):
        results.append(func(pk))
        if i % 5 == 0:
            current_mem = get_memory_usage()
            print(f"  After {i+1} iterations: {current_mem:.2f} MB (delta: {current_mem - start_mem:.2f} MB)")
    
    # Get tracemalloc snapshot
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    # Print top memory allocations
    print("  Top memory allocations by line:")
    top_stats = snapshot.statistics('lineno')
    for stat in top_stats[:5]:  # Show top 5
        print(f"    {stat.count} blocks: {stat.size/1024/1024:.2f} MB - {stat.traceback.format()[0]}")
    
    final_mem = get_memory_usage()
    print(f"  Final memory: {final_mem:.2f} MB (total increase: {final_mem - start_mem:.2f} MB)")
    
    # Check if memory usage decreased after results are no longer referenced
    del results
    gc.collect()
    post_delete_mem = get_memory_usage()
    print(f"  Memory after deleting results: {post_delete_mem:.2f} MB (freed: {final_mem - post_delete_mem:.2f} MB)")
    
    return final_mem - start_mem

if __name__ == "__main__":
    # Generate a simple test spectrum
    cosmo_params = random_cosmology()
    k, pk = generate_power_spectrum(
        omega_cdm=cosmo_params['omega_cdm'],
        h=cosmo_params['h'],
        omega_b=cosmo_params['omega_b'],
        z=cosmo_params['z'],
        k_min=1e-4, 
        k_max=1.0, 
        n_points=3000
    )
    
    # Initialize FASTPT
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    fpt = FASTPT(k, n_pad=int(0.5*len(k)))
    print(f"After FASTPT initialization: {get_memory_usage():.2f} MB")
    
    # Profile individual functions
    functions_to_profile = ["one_loop_dd", "one_loop_dd_bias", "one_loop_dd_bias_b3nl", "one_loop_dd_bias_lpt_NL", 
            "IA_tt", "IA_mix", "IA_ta", "OV", "kPol"]
    
    function_memory = {}
    for func_name in functions_to_profile:
        memory_increase = profile_single_function(fpt, func_name, pk)
        function_memory[func_name] = memory_increase
        fpt._clear_cached_x_params()
    
    # Print summary
    print("\nMemory Usage Summary:")
    for func_name, memory in function_memory.items():
        print(f"  {func_name}: {memory:.2f} MB")
    
    # Test cache effectiveness
    print("\nTesting cache effectiveness...")
    gc.collect()
    start_mem = get_memory_usage()
    print(f"Memory before cached runs: {start_mem:.2f} MB")
    
    # Run all functions again (should use cache)
    for func_name in functions_to_profile:
        func = getattr(fpt, func_name)
        result = func(pk)
    
    cached_mem = get_memory_usage()
    print(f"Memory after cached runs: {cached_mem:.2f} MB (increase: {cached_mem - start_mem:.2f} MB)")
    
    # Print cache stats
    print("\nCache statistics:")
    print(fpt.cache.stats())