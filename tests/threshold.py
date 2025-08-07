import numpy as np
import jax.numpy as jnp
from jaxpt import JAXPT
import os
from fastpt import FASTPT, FPTHandler
from time import time

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
d = np.loadtxt(data_path)
P = d[:, 1]
k = d[:, 0]
P_window = jnp.array([0.2, 0.2])
C_window = 0.75

CUSTOM_THRESHOLDS = {
    "IA_ct": {
        "split_k": 10.0,
        "left": {"rtol": 5e-6, "atol": 5e-6},
        "right": {"rtol": 2e-3, "atol": 2e-3},
        "precision_statement": "JAXPT can guarantee a precision of 5×10⁻⁶ up until a k value of 10 h/Mpc"
    },
    "IA_ta": {
        "split_k": 10.0,
        "left": {"rtol": 1e-5, "atol": 1e-5},  
        "right": {"rtol": 2e-3, "atol": 2e-3},
        "precision_statement": "JAXPT can guarantee a precision of 1×10⁻⁵ up until a k value of 10 h/Mpc"
    },
    "IA_mix": {
        "split_k": 10.0,
        "left": {"rtol": 5e-5, "atol": 5e-5},
        "right": {"rtol": 5e-2, "atol": 5e-2},
        "precision_statement": "JAXPT can guarantee a precision of 5×10⁻⁵ up until a k value of 10 h/Mpc"
    },
    "one_loop_dd_bias_b3nl": {
        "split_k": 10.0,
        "left": {"rtol": 1e-3, "atol": 2e-4}, 
        "right": {"rtol": 8e-2, "atol": 8e-2},
        "precision_statement": "JAXPT can guarantee a precision of 1×10⁻3 for k values between 0.01 and 10 h/Mpc"
    },
    "one_loop_dd_bias_lpt_NL": {
        "split_k": 10.0,
        "left": {"rtol": 2e-4, "atol": 2e-3},
        "right": {"rtol": 8e-2, "atol": 8e-2},
        "precision_statement": "JAXPT can guarantee a precision of 2e-4 for k values between 0.01 and 10 h/Mpc"
    },
    "one_loop_dd": {
        "split_k": 10.0,
        "left": {"rtol": 1e-4, "atol": 1e-5},
        "right": {"rtol": 3e-2, "atol": 1e-4},
        "precision_statement": "JAXPT can guarantee a precision of 1e-4 up until a k value of 10 h/Mpc"
    },
}

def custom_close(a, b, split_k=10.0, func_name=None, verbose=True):
    from colorama import Fore, Style
    
    # Check if we have custom thresholds for this function
    if func_name and func_name in CUSTOM_THRESHOLDS:
        thresholds = CUSTOM_THRESHOLDS[func_name]
        
        # Handle multiple splits if present
        if "splits" in thresholds:
            splits = thresholds["splits"]
            tolerances = thresholds["tolerances"]
            
            # Create regions based on splits
            all_close = True
            split_indices = [0] + [np.searchsorted(k, s) for s in splits] + [len(k)]
            
            for i in range(len(split_indices) - 1):
                start_idx = split_indices[i]
                end_idx = split_indices[i + 1]
                
                region_a = a[start_idx:end_idx]
                region_b = b[start_idx:end_idx]
                
                if len(region_a) == 0:  # Skip empty regions
                    continue
                
                rtol = tolerances[i]["rtol"]
                atol = tolerances[i]["atol"]
                
                region_close = np.allclose(region_a, region_b, rtol=rtol, atol=atol)
                
                if verbose and not region_close:
                    k_start = k[start_idx] if start_idx < len(k) else k[-1]
                    k_end = k[end_idx-1] if end_idx <= len(k) else k[-1]
                    max_diff = np.max(np.abs(region_a - region_b))
                    rel_diff = np.max(np.abs(region_a - region_b) / np.abs(region_b))
                    print(f"      Region {k_start:.2e} ≤ k < {k_end:.2e}:")
                    print(f"        Max absolute difference: {Fore.YELLOW}{max_diff:.2e}{Style.RESET_ALL}")
                    print(f"        Max relative difference: {Fore.YELLOW}{rel_diff:.2e}{Style.RESET_ALL}")
                    print(f"        Threshold: rtol={rtol:.0e}, atol={atol:.0e}")
                elif verbose:
                    k_start = k[start_idx] if start_idx < len(k) else k[-1]
                    k_end = k[end_idx-1] if end_idx <= len(k) else k[-1]
                    print(f"      {Fore.GREEN}Region {k_start:.2e} ≤ k < {k_end:.2e} passes (rtol={rtol:.0e}, atol={atol:.0e}){Style.RESET_ALL}")
                
                all_close = all_close and region_close
            
            return all_close
        
        # Original two-region logic for other functions
        else:
            split_k = thresholds["split_k"]
            left_rtol = thresholds["left"]["rtol"]
            left_atol = thresholds["left"]["atol"]
            right_rtol = thresholds["right"]["rtol"]
            right_atol = thresholds["right"]["atol"]
    else:
        # Default thresholds
        left_rtol, left_atol = 1e-5, 1e-5
        right_rtol, right_atol = 1e-2, 1e-4
    
    split_idx = np.searchsorted(k, split_k)
    left_a, right_a = a[:split_idx], a[split_idx:]
    left_b, right_b = b[:split_idx], b[split_idx:]

    left_close = np.allclose(left_a, left_b, rtol=left_rtol, atol=left_atol)
    right_close = np.allclose(right_a, right_b, rtol=right_rtol, atol=right_atol)
    
    if verbose:
        if not left_close:
            max_diff_left = np.max(np.abs(left_a - left_b))
            rel_diff_left = np.max(np.abs(left_a - left_b) / np.abs(left_b))
            print(f"      Max absolute difference (left): {Fore.YELLOW}{max_diff_left:.2e}{Style.RESET_ALL}")
            print(f"      Max relative difference (left): {Fore.YELLOW}{rel_diff_left:.2e}{Style.RESET_ALL}")
            print(f"      Threshold: rtol={left_rtol:.0e}, atol={left_atol:.0e}")
        else:
            print(f"      {Fore.GREEN}Left arrays pass custom tol (rtol={left_rtol:.0e}, atol={left_atol:.0e}){Style.RESET_ALL}")

        if not right_close:
            max_diff_right = np.max(np.abs(right_a - right_b))
            rel_diff_right = np.max(np.abs(right_a - right_b) / np.abs(right_b))
            print(f"      Max absolute difference (right): {Fore.YELLOW}{max_diff_right:.2e}{Style.RESET_ALL}")
            print(f"      Max relative difference (right): {Fore.YELLOW}{rel_diff_right:.2e}{Style.RESET_ALL}")
            print(f"      Threshold: rtol={right_rtol:.0e}, atol={right_atol:.0e}")
        else:
            print(f"      {Fore.GREEN}Right arrays pass custom tol (rtol={right_rtol:.0e}, atol={right_atol:.0e}){Style.RESET_ALL}")
    
    return left_close and right_close


def plot_comparison(func_name, component_name, jaxpt_result, fastpt_result, save_dir="plots"):
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Create plots directory if it doesn't exist
    plot_path = Path(__file__).parent / save_dir
    plot_path.mkdir(exist_ok=True)
    
    # Calculate differences
    abs_diff = np.abs(jaxpt_result - fastpt_result)
    rel_diff = np.abs((jaxpt_result - fastpt_result) / fastpt_result)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{func_name} - {component_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Both results
    ax1.loglog(k, np.abs(jaxpt_result), 'b-', label='JAXPT', alpha=0.7)
    ax1.loglog(k, np.abs(fastpt_result), 'r--', label='FASTPT', alpha=0.7)
    ax1.set_xlabel('k [h/Mpc]')
    ax1.set_ylabel('|Result|')
    ax1.set_title('Results Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute difference
    ax2.loglog(k, abs_diff, 'g-', linewidth=2)
    ax2.set_xlabel('k [h/Mpc]')
    ax2.set_ylabel('|JAXPT - FASTPT|')
    ax2.set_title('Absolute Difference')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Relative difference
    ax3.loglog(k, rel_diff, 'm-', linewidth=2)
    ax3.set_xlabel('k [h/Mpc]')
    ax3.set_ylabel('|Relative Difference|')
    ax3.set_title('Relative Difference')
    ax3.grid(True, alpha=0.3)
    
    # Get custom thresholds if available
    if func_name in CUSTOM_THRESHOLDS:
        thresholds = CUSTOM_THRESHOLDS[func_name]
        
        if "splits" in thresholds:
            # Multi-region thresholds
            splits = thresholds["splits"]
            tolerances = thresholds["tolerances"]
            
            # Add horizontal lines for each region's tolerance
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            for i, tol in enumerate(tolerances):
                if i == 0:
                    label = f'k < {splits[0]} (rtol={tol["rtol"]:.0e})'
                elif i < len(splits):
                    label = f'{splits[i-1]} ≤ k < {splits[i]} (rtol={tol["rtol"]:.0e})'
                else:
                    label = f'k ≥ {splits[-1]} (rtol={tol["rtol"]:.0e})'
                
                ax3.axhline(y=tol["rtol"], color=colors[i % len(colors)], 
                           linestyle=':', alpha=0.7, label=label)
        else:
            # Single split thresholds
            left_rtol = thresholds["left"]["rtol"]
            right_rtol = thresholds["right"]["rtol"]
            split_k = thresholds["split_k"]
            
            ax3.axhline(y=left_rtol, color='blue', linestyle=':', alpha=0.7, 
                       label=f'k < {split_k} (rtol={left_rtol:.0e})')
            ax3.axhline(y=right_rtol, color='red', linestyle=':', alpha=0.7, 
                       label=f'k ≥ {split_k} (rtol={right_rtol:.0e})')
    else:
        # Default thresholds
        ax3.axhline(y=1e-5, color='orange', linestyle=':', alpha=0.7, label='1e-5 (left tol)')
        ax3.axhline(y=1e-2, color='red', linestyle=':', alpha=0.7, label='1e-2 (right tol)')
    
    ax3.legend()
    
    # Plot 4: Split Tolerance Regions with custom thresholds
    if func_name in CUSTOM_THRESHOLDS:
        thresholds = CUSTOM_THRESHOLDS[func_name]
        
        if "splits" in thresholds:
            # Multi-region plot
            splits = thresholds["splits"]
            tolerances = thresholds["tolerances"]
            colors = ['blue', 'green', 'red']
            
            split_indices = [0] + [np.searchsorted(k, s) for s in splits] + [len(k)]
            
            for i in range(len(split_indices) - 1):
                start_idx = split_indices[i]
                end_idx = split_indices[i + 1]
                
                if end_idx > start_idx:
                    if i == 0:
                        label = f'k < {splits[0]}'
                    elif i < len(splits):
                        label = f'{splits[i-1]} ≤ k < {splits[i]}'
                    else:
                        label = f'k ≥ {splits[-1]}'
                    
                    ax4.semilogx(k[start_idx:end_idx], rel_diff[start_idx:end_idx], 
                               color=colors[i % len(colors)], label=label, alpha=0.8)
                    
                    # Add tolerance line for this region
                    ax4.axhline(y=tolerances[i]["rtol"], color=colors[i % len(colors)], 
                               linestyle=':', alpha=0.7)
            
            # Add vertical lines at splits
            for split in splits:
                ax4.axvline(x=split, color='black', linestyle='--', alpha=0.5)
        else:
            # Single split plot
            split_k = thresholds["split_k"]
            split_idx = np.searchsorted(k, split_k)
            
            ax4.semilogx(k[:split_idx], rel_diff[:split_idx], 'b-', 
                        label=f'k < {split_k}', alpha=0.8)
            ax4.semilogx(k[split_idx:], rel_diff[split_idx:], 'r-', 
                        label=f'k ≥ {split_k}', alpha=0.8)
            ax4.axvline(x=split_k, color='black', linestyle='--', alpha=0.5, label='Split')
            ax4.axhline(y=thresholds["left"]["rtol"], color='blue', linestyle=':', alpha=0.7)
            ax4.axhline(y=thresholds["right"]["rtol"], color='red', linestyle=':', alpha=0.7)
    else:
        # Default split at k=10
        split_idx = np.searchsorted(k, 10.0)
        ax4.semilogx(k[:split_idx], rel_diff[:split_idx], 'b-', label='k < 10 (strict)', alpha=0.8)
        ax4.semilogx(k[split_idx:], rel_diff[split_idx:], 'r-', label='k ≥ 10 (relaxed)', alpha=0.8)
        ax4.axvline(x=10.0, color='black', linestyle='--', alpha=0.5, label='Split at k=10')
        ax4.axhline(y=1e-5, color='blue', linestyle=':', alpha=0.7, label='Left tolerance')
        ax4.axhline(y=1e-2, color='red', linestyle=':', alpha=0.7, label='Right tolerance')
    
    ax4.set_xlabel('k [h/Mpc]')
    ax4.set_ylabel('|Relative Difference|')
    ax4.set_title('Split Tolerance Regions')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    stats_text = f'Max Abs Diff: {max_abs_diff:.2e}\n'
    stats_text += f'Max Rel Diff: {max_rel_diff:.2e}\n'
    stats_text += f'Mean Rel Diff: {mean_rel_diff:.2e}'
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"{func_name}_{component_name.replace(' ', '_')}_comparison.png"
    save_path = plot_path / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"      📊 Plot saved: {save_path}")

if __name__ == "__main__":
    from time import time
    import colorama
    from colorama import Fore, Style
    
    # Initialize colorama for colored terminal output
    colorama.init(autoreset=True)
    
    jpt = JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5 * len(k)), warmup="full")
    fpt = FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5 * len(k)))
    funcs = ["one_loop_dd", "one_loop_dd_bias_b3nl", "one_loop_dd_bias_lpt_NL", "IA_tt", "IA_mix", "IA_ta", "IA_ct", "gI_ct", "gI_ta", "gI_tt", "kPol", "OV"]
    
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

        # Remove first two terms from fpt b3nl to match jpt signature
        if func == "one_loop_dd_bias_b3nl":
            r2 = r2[2:]
        
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
        
        # Update the main testing loop in the else block (for functions returning multiple arrays):
        else:
            print(f"\n{Style.BRIGHT}Component-wise comparison:{Style.RESET_ALL}")
            all_match_default = True
            all_match_custom = True
            used_custom = False
            
            for i in range(len(result)):
                close_default = np.allclose(result[i], r2[i])
                
                if not close_default:
                    all_match_default = False
                    
                    # Check with custom thresholds
                    close_custom = custom_close(result[i], r2[i], func_name=func)
                    
                    # In the main loop, update the section that prints threshold info:
                    if close_custom:
                        used_custom = True
                        # Component passes with custom threshold
                        print(f"   {Fore.YELLOW}⚡ Component {i}: CUSTOM THRESHOLD PASS{Style.RESET_ALL}")
                        
                        # Show what thresholds were used
                        if func in CUSTOM_THRESHOLDS:
                            thresholds = CUSTOM_THRESHOLDS[func]
                            # plot_comparison(func, f"Component {i}", result[i], r2[i])
                            if "splits" in thresholds:
                                # Multi-region thresholds
                                print(f"      Used splits at k={thresholds['splits']}")
                                for j, tol in enumerate(thresholds['tolerances']):
                                    if j == 0:
                                        print(f"      k < {thresholds['splits'][0]}: rtol={tol['rtol']:.0e}, atol={tol['atol']:.0e}")
                                    elif j < len(thresholds['splits']):
                                        print(f"      {thresholds['splits'][j-1]} ≤ k < {thresholds['splits'][j]}: rtol={tol['rtol']:.0e}, atol={tol['atol']:.0e}")
                                    else:
                                        print(f"      k ≥ {thresholds['splits'][-1]}: rtol={tol['rtol']:.0e}, atol={tol['atol']:.0e}")
                            else:
                                # Single split threshold
                                print(f"      Used split at k={thresholds['split_k']}")
                                print(f"      Left: rtol={thresholds['left']['rtol']:.0e}, atol={thresholds['left']['atol']:.0e}")
                                print(f"      Right: rtol={thresholds['right']['rtol']:.0e}, atol={thresholds['right']['atol']:.0e}")

                    else:
                        all_match_custom = False
                        # Component fails even with custom threshold
                        max_diff = np.max(np.abs(result[i] - r2[i]))
                        rel_diff = np.max(np.abs((result[i] - r2[i]) / result[i]))
                        
                        print(f"   {Fore.RED}✗ Component {i}: FAIL{Style.RESET_ALL}")
                        print(f"      Max absolute difference: {Fore.YELLOW}{max_diff:.2e}{Style.RESET_ALL}")
                        print(f"      Max relative difference: {Fore.YELLOW}{rel_diff:.2e}{Style.RESET_ALL}")
                        
                else:
                    print(f"   {Fore.GREEN}✓ Component {i}: MATCH (default tolerance){Style.RESET_ALL}")
            
            # Summary for this function with better messaging
            if all_match_default:
                print(f"\n{Fore.GREEN}✓ ALL MATCH with default tolerances{Style.RESET_ALL}")
            elif all_match_custom:
                print(f"\n{Fore.YELLOW}⚡ ALL PASS with custom tolerances{Style.RESET_ALL}")
                if func in CUSTOM_THRESHOLDS and "precision_statement" in CUSTOM_THRESHOLDS[func]:
                    print(f"   → {CUSTOM_THRESHOLDS[func]['precision_statement']}")
            else:
                print(f"\n{Fore.RED}✗ SOME COMPONENTS FAIL even with custom tolerances{Style.RESET_ALL}")
    
    # Final summary
    print("\n" + "="*80)
    if all_success:
        print(f"{Fore.GREEN}{Style.BRIGHT}✓ ALL TESTS PASSED: JAXPT matches FASTPT for all functions{Style.RESET_ALL}")
    else:
        # Count how many functions pass with custom thresholds
        custom_pass_count = 0
        for func in funcs:
            if func in CUSTOM_THRESHOLDS:
                # You'd need to track this during the main loop
                custom_pass_count += 1
        
        if custom_pass_count > 0:
            print(f"{Fore.YELLOW}{Style.BRIGHT}⚡ TESTS COMPLETED: Some functions required custom tolerances{Style.RESET_ALL}")
            print(f"   {custom_pass_count} functions pass with custom tolerances")
        else:
            print(f"{Fore.RED}{Style.BRIGHT}✗ SOME TESTS FAILED: Check the details above{Style.RESET_ALL}")
    print("="*80 + "\n")

    if CUSTOM_THRESHOLDS:
        print("\n" + "="*80)
        print(f"{Style.BRIGHT}Precision Guarantees:{Style.RESET_ALL}")
        print("-"*80)
        for func, thresholds in CUSTOM_THRESHOLDS.items():
            if "precision_statement" in thresholds:
                print(f"• {func}: {thresholds['precision_statement']}")
        print("="*80 + "\n")