import os
import sys
import time
import numpy as np

# Add the parent directory to the path so we can import FASTPT
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastpt import FASTPT, FPTHandler
from fastpt.timing_utils import print_timing_report

# Import and modify the timing utils module
import fastpt.timing_utils as timing_utils

# Turn off automatic reporting at exit
if hasattr(timing_utils, 'atexit'):
    # Unregister the print_timing_report from atexit
    timing_utils.atexit.unregister(timing_utils.print_timing_report)

def generate_test_power_spectrum(k):
    """Generate a simple test power spectrum"""
    return 1e4 * k**(-2) * np.exp(-0.5 * ((np.log(k) - np.log(0.1))/0.3)**2)

def reset_timing_data():
    """Reset all timing data for a fresh run"""
    # Reset all the timing data storages
    if hasattr(timing_utils, '_timing_data'):
        timing_utils._timing_data.clear()
    if hasattr(timing_utils, '_start_times'):
        timing_utils._start_times.clear()
    if hasattr(timing_utils, '_call_counts'):
        timing_utils._call_counts.clear()
    if hasattr(timing_utils, '_call_stack'):
        timing_utils._call_stack.clear()

def run_test(dump_cache=True, n_spectra=10):
    """Run a test with timing instrumentation"""
    print(f"\n{'='*80}\nRUNNING TEST WITH dump_cache={dump_cache}\n{'='*80}")
    
    # Initialize FASTPT parameters
    k_min, k_max = 1e-3, 10
    n_points = 1000
    
    # Generate k array
    k = np.logspace(np.log10(k_min), np.log10(k_max), n_points)
    
    # Initialize FASTPT
    fpt = FASTPT(k, to_do=['all'], n_pad=int(0.5*len(k)))
    
    # Set cache dump behavior
    fpt.cache.dump_cache = dump_cache
    
    # Create handler
    handler = FPTHandler(fpt, P_window=np.array([0.2, 0.2]), C_window=0.75)
    
    # Time entire run
    start_time = time.time()
    
    # Run with multiple power spectra
    for i in range(n_spectra):
        print(f"\n--- Processing power spectrum {i+1}/{n_spectra} ---")
        
        # Generate slightly different power spectrum each time
        P = generate_test_power_spectrum(k) * (1 + 0.1 * np.sin(i))
        
        # Run function
        result = handler.run("IA_tt", P=P)
        
        # Stats after each run
        print(f"Cache hits: {fpt.cache.hits}, misses: {fpt.cache.misses}")
    
    # Total time
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.4f} seconds")
    
    # Return summary stats
    return {
        "total_time": elapsed,
        "hits": fpt.cache.hits,
        "misses": fpt.cache.misses,
        "hit_ratio": fpt.cache.hits / (fpt.cache.hits + fpt.cache.misses) if (fpt.cache.hits + fpt.cache.misses) > 0 else 0
    }

def compare_performance():
    """Run tests with and without dump cache and compare results"""
    # 1. First run with dump_cache=True
    reset_timing_data()
    stats_true = run_test(dump_cache=True)
    
    print("\n" + "="*80)
    print("TIMING REPORT FOR dump_cache=True")
    print("="*80)
    print_timing_report()
    
    # 2. Second run with dump_cache=False
    reset_timing_data()
    stats_false = run_test(dump_cache=False)
    
    print("\n" + "="*80)
    print("TIMING REPORT FOR dump_cache=False")
    print("="*80)
    print_timing_report()
    
    # 3. Comparison summary
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    
    print(f"WITH dump_cache=True:")
    print(f"  Total time: {stats_true['total_time']:.4f} seconds")
    print(f"  Cache hits: {stats_true['hits']}")
    print(f"  Cache misses: {stats_true['misses']}")
    print(f"  Hit ratio: {stats_true['hit_ratio']:.2%}")
    
    print(f"\nWITH dump_cache=False:")
    print(f"  Total time: {stats_false['total_time']:.4f} seconds")
    print(f"  Cache hits: {stats_false['hits']}")
    print(f"  Cache misses: {stats_false['misses']}")
    print(f"  Hit ratio: {stats_false['hit_ratio']:.2%}")
    
    # Calculate speedup
    speedup = stats_true['total_time'] / stats_false['total_time'] if stats_false['total_time'] > 0 else 0
    print(f"\nSPEEDUP WITHOUT DUMP CACHE: {speedup:.2f}x")
    print(f"PERFORMANCE IMPROVEMENT: {(speedup-1)*100:.2f}%")

if __name__ == "__main__":
    compare_performance()