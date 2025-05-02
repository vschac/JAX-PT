import time
import inspect
import functools
import atexit
import numpy as np
from collections import defaultdict

# Storage for all timing data
_timing_data = defaultdict(list)
_start_times = {}
_call_counts = defaultdict(int)
_call_stack = []

__all__ = [
    'timing_checkpoint', 
    'time_function', 
    'print_timing_report',
    '_timing_data',
    '_start_times',
    '_call_counts',
    '_call_stack'
]

def timing_checkpoint(label=None):
    """Record timing at this checkpoint"""
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split("/")[-1]
    func_name = frame.f_code.co_name
    line_no = frame.f_lineno
    
    if not label:
        label = f"{func_name}:{line_no}"
    
    checkpoint_id = f"{filename}:{label}"
    current_time = time.time()
    
    # If we have a start time for this checkpoint, calculate duration
    if checkpoint_id in _start_times:
        duration = current_time - _start_times[checkpoint_id]
        _timing_data[checkpoint_id].append(duration)
        
        # Calculate context (where in the call stack we are)
        context = ":".join(_call_stack) if _call_stack else "main"
        
        # Count this call
        _call_counts[checkpoint_id] += 1
    
    # Update start time for next measurement
    _start_times[checkpoint_id] = current_time
    return current_time

def time_function(func=None, *, detailed=False):
    """Decorator to time a function's execution
    
    Args:
        func: The function to time
        detailed: If True, also time each call to timing_checkpoint inside the function
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            global _call_stack
            
            # Function identifier
            if hasattr(f, "__qualname__"):
                func_id = f.__qualname__
            else:
                # Get full name including class for methods
                if args and hasattr(args[0].__class__, f.__name__):
                    func_id = f"{args[0].__class__.__name__}.{f.__name__}"
                else:
                    func_id = f.__name__
            
            # Track the call stack
            _call_stack.append(func_id)
            
            # Start timing
            start_time = time.time()
            
            # Execute the function
            try:
                result = f(*args, **kwargs)
            finally:
                # Always ensure we pop from call stack even if there's an exception
                if _call_stack:
                    _call_stack.pop()
            
            # Record timing
            duration = time.time() - start_time
            _timing_data[func_id].append(duration)
            _call_counts[func_id] += 1
            
            return result
        return wrapper
    
    # Handle both @time_function and @time_function(detailed=True)
    if func is None:
        return decorator
    return decorator(func)

def print_timing_report():
    """Print a detailed timing report"""
    print("\n" + "="*80)
    print("TIMING REPORT")
    print("="*80)
    
    # Sort by total time spent
    sorted_items = sorted(
        [(k, np.sum(_timing_data[k]), len(_timing_data[k]), _call_counts[k]) for k in _timing_data],
        key=lambda x: x[1],
        reverse=True
    )
    
    print(f"{'Function/Checkpoint':<50} {'Calls':<10} {'Total (s)':<12} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
    print("-"*110)
    
    for name, total, measurements, calls in sorted_items:
        avg = total / measurements if measurements > 0 else 0
        min_time = min(_timing_data[name]) if measurements > 0 else 0
        max_time = max(_timing_data[name]) if measurements > 0 else 0
        
        print(f"{name:<50} {calls:<10} {total:<12.6f} {avg:<12.6f} {min_time:<12.6f} {max_time:<12.6f}")

# Register to print report when program exits
atexit.register(print_timing_report)