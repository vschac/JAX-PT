import jax.numpy as jnp
from jax import jit, vjp, jvp, jacfwd, grad
import numpy as np
from jax import config
import jax
from fastpt import FASTPT as FPT
from jaxpt.jax_utils import p_window, c_window, jax_k_extend
from jaxpt.jax_utils import P_13_reg, Y1_reg_NL, Y2_reg_NL, P_IA_B, P_IA_deltaE2, P_IA_13F, P_IA_13G
config.update("jax_enable_x64", True)
import functools
from jax.numpy.fft import ifft, irfft
from dataclasses import dataclass
from typing import Optional, Any
from functools import partial
from time import time


def process_x_term(X):
    """Process X term for JAX compatibility, preserving complex values and handling nested arrays."""
    processed_X = []
    
    for term in X:
        if isinstance(term, np.ndarray):
            if term.dtype == np.dtype('O') or (term.size > 0 and isinstance(term.flat[0], np.ndarray)):
                nested_arrays = []
                for item in term:
                    if isinstance(item, np.ndarray):
                        if np.iscomplexobj(item):
                            nested_arrays.append(jnp.asarray(item, dtype=jnp.complex128))
                        else:
                            nested_arrays.append(jnp.asarray(item, dtype=jnp.float64))
                    else:
                        # Handle direct values (unlikely but possible)
                        nested_arrays.append(jnp.asarray(item, dtype=jnp.float64))
                
                term = jnp.asarray(nested_arrays)
            
            elif term.dtype == np.float64:
                term = jnp.asarray(term, dtype=jnp.float64)
            elif term.dtype == np.complex128:
                term = jnp.asarray(term, dtype=jnp.complex128)
            # Fallback for other array types
            else:
                print("fallback")
                try:
                    # Sample the first element to determine the type
                    if term.size > 0:
                        sample = term.flat[0]
                        if isinstance(sample, complex) or (hasattr(sample, 'imag') and sample.imag != 0):
                            term = jnp.asarray(term, dtype=jnp.complex128)
                        else:
                            term = jnp.asarray(term, dtype=jnp.float64)
                    else:
                        # Empty array, default to float64
                        term = jnp.asarray(term, dtype=jnp.float64)
                except:
                    print(f"Warning: Could not determine type for array of dtype {term.dtype}")
                    term = jnp.asarray(term)
        
        processed_X.append(term)
    
    return tuple(processed_X)

def jax_cached_property(method):
    prop_name = '_' + method.__name__

    @functools.wraps(method)
    def wrapper(self):
        # When being traced, always compute the result fresh without caching
        if isinstance(self, jax.core.Tracer):
            result = method(self)
            if isinstance(result, tuple):
                return process_x_term(result)
            return result
            
        # Only access cached properties when not being traced by JAX
        if not hasattr(self, prop_name):
            result = method(self)
            # Process X terms for JAX compatibility
            if isinstance(result, tuple):
                converted = process_x_term(result)
            else:
                converted = result
            setattr(self, prop_name, converted)
        
        return getattr(self, prop_name)
    return property(wrapper)

class JAXPT: 
    def __init__(self, k, low_extrap=None, high_extrap=None, n_pad=None, warmup=True):
        
        if (k is None or len(k) == 0):
            raise ValueError('You must provide an input k array.')        
        if not isinstance(k, jnp.ndarray):
            try:
                k = jnp.asarray(k, dtype=jnp.float64)
            except:
                raise ValueError('Input k array must be a jax numpy array, automatic conversion failed.')
            
        self.__k_original = k
        self.temp_fpt = FPT(k.copy(), low_extrap=low_extrap, high_extrap=high_extrap, n_pad=n_pad)
        self.extrap = False
        self.EK = None
        if (low_extrap is not None or high_extrap is not None):
            if (high_extrap < low_extrap):
                raise ValueError('high_extrap must be greater than low_extrap')
            self.EK = jax_k_extend(k, low_extrap, high_extrap)
            k = self.EK.extrap_k()
            self.extrap = True

        self.low_extrap = low_extrap
        self.high_extrap = high_extrap
        self.__k_extrap = k #K extrapolation not padded

        dk = jnp.diff(jnp.log(k))
        delta_L = (jnp.log(k[-1]) - jnp.log(k[0])) / (k.size - 1)
        dk_test = jnp.ones_like(dk) * delta_L

        log_sample_test = 'ERROR! FASTPT will not work if your in put (k,Pk) values are not sampled evenly in log space!'
        np.testing.assert_array_almost_equal(dk, dk_test, decimal=4, err_msg=log_sample_test, verbose=False)

        if (k.size % 2 != 0):
            raise ValueError('Input array must contain an even number of elements.')

        if n_pad is None:
            n_pad = int(0.5 * len(k))
        self.n_pad = n_pad
        if (n_pad > 0):
            if not isinstance(n_pad, int):
                n_pad = int(n_pad)
            self.n_pad = n_pad
            self.id_pad = jnp.arange(k.size) + n_pad
            d_logk = delta_L
            k_pad = jnp.log(k[0]) - jnp.arange(1, n_pad + 1) * d_logk
            k_pad = jnp.exp(k_pad)
            k_left = k_pad[::-1]

            k_pad = jnp.log(k[-1]) + jnp.arange(1, n_pad + 1) * d_logk
            k_right = jnp.exp(k_pad)
            k = jnp.hstack((k_left, k, k_right))
            n_pad_check = int(jnp.log(2) / delta_L) + 1
            if (n_pad < n_pad_check):
                print('*** Warning ***')
                print(f'You should consider increasing your zero padding to at least {n_pad_check}')
                print('to ensure that the minimum k_output is > 2k_min in the FASTPT universe.')
                print(f'k_min in the FASTPT universe is {k[0]} while k_min_input is {self.k_extrap[0]}')

        self.__k_final = k #log spaced k, with padding and extrap
        self.k_size = k.size
        self.N = k.size

        # define eta_m and eta_n=eta_m
        omega = 2 * jnp.pi / (float(self.N) * delta_L)
        self.m = jnp.arange(-self.N // 2, self.N // 2 + 1)
        self.eta_m = omega * self.m

        # define l and tau_l
        self.n_l = self.m.size + self.m.size - 1
        self.l = jnp.arange(-self.n_l // 2 + 1, self.n_l // 2 + 1)
        self.tau_l = omega * self.l


        self._static_config = StaticConfig(
            k_size=self.k_size,
            n_pad=self.n_pad,
            N=self.N,
            low_extrap=self.low_extrap,
            high_extrap=self.high_extrap,
            EK=self.EK
        )

        #These cannot be cached properties since they would be accessed twice in one function call (the one loop functions)
        #Therefore producing a side affect as the second access is done via cache and breaking differentiability
        # self.X_spt = process_x_term(self.temp_fpt.X_spt)
        # self.X_lpt = process_x_term(self.temp_fpt.X_lpt)
        #TODO may be able to add these back as cached properties, check differentiability

        if warmup: self._simple_warmup()

    def _simple_warmup(self):
        """Streamlined JIT warm-up focused on top-level API functions and internal computation methods."""
        print("Starting JIT warm-up...")
        
        # Prepare test inputs
        dummy_P = jnp.ones_like(self.k_original)
        window_settings = [
            {"P_window": None, "C_window": None},
            {"P_window": jnp.array([0.2, 0.2]), "C_window": 0.5},
            {"P_window": None, "C_window": 0.5},
            {"P_window": jnp.array([0.2, 0.2]), "C_window": None}
        ]
        
        # Warm up all the top-level API functions
        api_functions = [
            "one_loop_dd_bias_b3nl", 
            "one_loop_dd_bias_lpt_NL", 
            "IA_tt", 
            "IA_mix", 
            "IA_ta", 
            "IA_ct", 
            "gI_ct", 
            "gI_ta", 
            "gI_tt", 
            "kPol",
            "OV"
        ]
        
        # Warm up each function with each window setting
        for func_name in api_functions:
            func = getattr(self, func_name)
            for settings in window_settings:
                _ = func(dummy_P, **settings)
        
        # # Warm up _get_ methods used by SPT calculations
        # spt_get_methods = [
        #     '_get_1loop', '_get_sig4', '_get_Pd1d2', '_get_Pd2d2', 
        #     '_get_Pd1s2', '_get_Pd2s2', '_get_Ps2s2', '_get_sig3nl'
        # ]
        
        # for method_name in spt_get_methods:
        #     for settings in window_settings:
        #         try:
        #             _ = globals()[method_name](
        #                 dummy_P, self.X_spt, self._static_config,
        #                 self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
        #                 **settings
        #             )
        #         except Exception:
        #             pass  # Silently continue if method doesn't exist or has errors
        
        # # Warm up _get_ methods used by LPT calculations
        # lpt_get_methods = ['_get_Pb1L', '_get_Pb1L_2', '_get_Pb1L_b2L', '_get_Pb2L', '_get_Pb2L_2']
        
        # for method_name in lpt_get_methods:
        #     for settings in window_settings:
        #         try:
        #             _ = globals()[method_name](
        #                 dummy_P, self.X_lpt, self._static_config,
        #                 self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
        #                 **settings
        #             )
        #         except Exception:
        #             pass
        
        # # Warm up tensor term get methods
        # special_get_methods = [
        #     {'name': '_get_P_0EtE', 'X': [self.X_IA_tij_feG2, self.X_IA_deltaE1]},
        #     {'name': '_get_P_E2tE', 'X': [self.X_IA_tij_heG2, self.X_IA_A]},
        #     {'name': '_get_P_tEtE', 'X': [self.X_IA_tij_F2F2, self.X_IA_tij_G2G2, self.X_IA_tij_F2G2]},
        #     {'name': '_get_P_d2tE', 'X': [self.X_IA_gb2_F2, self.X_IA_gb2_G2]},
        #     {'name': '_get_P_s2tE', 'X': [self.X_IA_gb2_S2F2, self.X_IA_gb2_S2G2]}
        # ]
        
        # for method_info in special_get_methods:
        #     for settings in window_settings:
        #         try:
        #             _ = globals()[method_info['name']](
        #                 dummy_P, method_info['X'], self._static_config,
        #                 self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
        #                 **settings
        #             )
        #         except Exception:
        #             pass
        
        # # Warm up _get_P_0tE which has a different signature
        # try:
        #     _ = globals()['_get_P_0tE'](
        #         dummy_P, [self.X_spt, self.X_sptG], self._static_config, 
        #         self.k_original, self.k_extrap, self.k_final, 
        #         self.id_pad, self.l, self.m, 
        #         P_window=None, C_window=None
        #     )
        # except Exception:
        #     pass
        
        # # Warm up methods that don't use window parameters
        # try:
        #     _ = globals()['_get_P_Btype2'](dummy_P, self.k_original)
        #     _ = globals()['_get_P_deltaE2'](dummy_P, self.k_original)
        # except Exception:
        #     pass
        
        # # Warm up compute_term with different X parameters and operations
        # key_tensors = [
        #     self.X_IA_E, self.X_kP1, self.X_kP2, self.X_kP3
        # ]
        
        # operations = [None, lambda x: 2.0 * x, lambda x: x / (80 * jnp.pi ** 2)]
        
        # for X in key_tensors:
        #     for op in operations:
        #         for settings in window_settings:
        #             try:
        #                 _ = compute_term(
        #                     dummy_P, X, self._static_config, self.k_extrap, self.k_final, 
        #                     self.id_pad, self.l, self.m, operation=op, **settings
        #                 )
        #             except Exception:
        #                 pass
        
        # # Specifically warm up kPol which was showing performance issues
        # _ = self.kPol(dummy_P, P_window=None, C_window=None)
        # _ = self.kPol(dummy_P, P_window=jnp.array([0.2, 0.2]), C_window=None)
        
        print("JIT warm-up completed.")

    def _warm_up_jit_functions(self):
        """Calls JIT-compiled functions with dummy arguments to force compilation."""
        print("Starting JIT warm-up...")

        # Dummy Power Spectra
        dummy_P_orig = jnp.ones_like(self.k_original, dtype=jnp.float64)
        
        # For functions that expect P already extrapolated, create a dummy directly
        # of the right size to match k_extrap after transformation
        if self.EK is not None:
            # Method 1: Create a simple array of the right size
            dummy_P_extrap_raw = jnp.ones_like(self.k_extrap, dtype=jnp.float64)
            # Apply extrapolation to get the right size for functions that expect
            # to transform the input P themselves
            dummy_P_for_transforms = self.EK.PK_original(dummy_P_extrap_raw)[1]
        else:
            dummy_P_for_transforms = dummy_P_orig
            dummy_P_extrap_raw = dummy_P_orig

        dummy_P_final_grid_input = jnp.ones_like(self.k_final, dtype=jnp.float64)

        # Setup window parameters for all combinations:
        # 1. Both None (already in the original code)
        # 2. Only P_window
        # 3. Only C_window
        # 4. Both provided
        
        # Define sample window values
        p_window_value = (0.2, 0.3)  # Sample values for p_window parameters
        c_window_value = 0.5         # Sample value for c_window parameter
        
        # Create pre-computed window arrays
        p_window_array = p_window(self.k_extrap, p_window_value[0], p_window_value[1])
        
        # Define parameter combinations for warm-up
        window_combinations = [
            {"P_window": None, "C_window": None, "label": "no windows"},
            {"P_window": p_window_array, "C_window": None, "label": "only P_window"},
            {"P_window": None, "C_window": c_window_value, "label": "only C_window"},
            {"P_window": p_window_array, "C_window": c_window_value, "label": "both windows"}
        ]
        
        # Common functions to warm up with all window combinations
        core_functions = [
            # Tuple format: (function_name, args excluding window params)
            ("J_k_scalar", [dummy_P_for_transforms, self.X_spt, self._static_config, 
                            self.k_extrap, self.k_final, self.id_pad, self.l, self.m]),
            ("J_k_tensor", [dummy_P_for_transforms, self.X_IA_E, self._static_config, 
                        self.k_extrap, self.k_final, self.id_pad, self.l, self.m]),
            ("compute_term", [dummy_P_for_transforms, self.X_IA_E, self._static_config, 
                            self.k_extrap, self.k_final, self.id_pad, self.l, self.m], 
                            {"operation": lambda x: 2.0 * x})
        ]
        
        # --- Function specific warm-up calls ---

        # First warm up core functions with all window combinations
        for combo in window_combinations:
            for func_info in core_functions:
                if len(func_info) == 2:
                    func_name, args = func_info
                    kwargs = {}
                else:
                    func_name, args, kwargs = func_info
                    
                # Add window parameters
                kwargs["P_window"] = combo["P_window"]
                kwargs["C_window"] = combo["C_window"]
                
                # Call the function
                _ = globals()[func_name](*args, **kwargs)
        
        # Continue with existing warm-up calls for other components
        _ = _apply_extrapolation(dummy_P_orig, EK=self._static_config.EK)
        _ = _apply_extrapolation(dummy_P_extrap_raw, EK=self._static_config.EK)

        # P_b should be on the k_final grid (size N)
        dummy_P_b_for_fourier = dummy_P_final_grid_input * self.k_final**(-2.0) # Example nu=-2
        
        # Warm up fourier_coefficients with both window settings
        _ = fourier_coefficients(dummy_P_b_for_fourier, self.m, self.N, C_window=None)
        _ = fourier_coefficients(dummy_P_b_for_fourier, self.m, self.N, C_window=c_window_value)
        
        dummy_c_m_output = fourier_coefficients(dummy_P_b_for_fourier, self.m, self.N, C_window=None)

        # convolution (window parameters don't apply directly)
        if self.X_spt and len(self.X_spt) >= 6:
            g_m_dummy_conv = self.X_spt[2][0]
            g_n_dummy_conv = self.X_spt[3][0]
            h_l_dummy_conv = self.X_spt[5][0]
            two_part_l_dummy_conv = self.X_spt[4][0] if self.X_spt[4] is not None and len(self.X_spt[4]) > 0 else None
            _ = convolution(dummy_c_m_output, dummy_c_m_output, g_m_dummy_conv, g_n_dummy_conv, h_l_dummy_conv, two_part_l_dummy_conv)

        # Warm up special function groups with different window combinations
        spt_kernel_funcs = ['_get_1loop', '_get_sig4', '_get_Pd1d2', '_get_Pd2d2', '_get_Pd1s2', '_get_Pd2s2', '_get_Ps2s2', '_get_sig3nl']
        
        for func_name in spt_kernel_funcs:
            # Warm up with no windows and with both windows (representative cases)
            _ = globals()[func_name](dummy_P_for_transforms, self.X_spt, self._static_config, 
                                    self.k_extrap, self.k_final, self.id_pad, self.l, self.m, 
                                    P_window=None, C_window=None)
            _ = globals()[func_name](dummy_P_for_transforms, self.X_spt, self._static_config, 
                                    self.k_extrap, self.k_final, self.id_pad, self.l, self.m, 
                                    P_window=p_window_array, C_window=c_window_value)
        
        lpt_kernel_funcs = ['_get_Pb1L', '_get_Pb1L_2', '_get_Pb1L_b2L', '_get_Pb2L', '_get_Pb2L_2']
        
        for func_name in lpt_kernel_funcs:
            # Warm up with no windows and with both windows
            _ = globals()[func_name](dummy_P_for_transforms, self.X_lpt, self._static_config, 
                                    self.k_extrap, self.k_final, self.id_pad, self.l, self.m, 
                                    P_window=None, C_window=None)
            _ = globals()[func_name](dummy_P_for_transforms, self.X_lpt, self._static_config, 
                                    self.k_extrap, self.k_final, self.id_pad, self.l, self.m, 
                                    P_window=p_window_array, C_window=c_window_value)

        # Warm up OV
        _ = globals()['_get_OV'](dummy_P_for_transforms, self.X_OV, 
                                self._static_config, self.k_extrap, self.k_final, 
                                self.id_pad, self.l, self.m, 
                                P_window=None, C_window=None)
        _ = globals()['_get_OV'](dummy_P_for_transforms, self.X_OV, 
                                self._static_config, self.k_extrap, self.k_final, 
                                self.id_pad, self.l, self.m, 
                                P_window=p_window_array, C_window=c_window_value)

        # Special cases with list-based X parameters
        special_funcs = [
            {'name': '_get_P_0EtE', 'X': [self.X_IA_tij_feG2, self.X_IA_deltaE1]},
            {'name': '_get_P_E2tE', 'X': [self.X_IA_tij_heG2, self.X_IA_A]},
            {'name': '_get_P_tEtE', 'X': [self.X_IA_tij_F2F2, self.X_IA_tij_G2G2, self.X_IA_tij_F2G2]},
            {'name': '_get_P_d2tE', 'X': [self.X_IA_gb2_F2, self.X_IA_gb2_G2]},
            {'name': '_get_P_s2tE', 'X': [self.X_IA_gb2_S2F2, self.X_IA_gb2_S2G2]}
        ]
        
        for func_info in special_funcs:
            # Warm up with no windows and with both windows
            _ = globals()[func_info['name']](dummy_P_for_transforms, func_info['X'], 
                                            self._static_config, self.k_extrap, self.k_final, 
                                            self.id_pad, self.l, self.m, 
                                            P_window=None, C_window=None)
            _ = globals()[func_info['name']](dummy_P_for_transforms, func_info['X'], 
                                            self._static_config, self.k_extrap, self.k_final, 
                                            self.id_pad, self.l, self.m, 
                                            P_window=p_window_array, C_window=c_window_value)
        
        # _get_P_0tE has a different signature with k_original
        _ = globals()['_get_P_0tE'](dummy_P_for_transforms, [self.X_spt, self.X_sptG], 
                                self._static_config, self.k_original, self.k_extrap, 
                                self.k_final, self.id_pad, self.l, self.m, 
                                P_window=None, C_window=None)
        _ = globals()['_get_P_0tE'](dummy_P_for_transforms, [self.X_spt, self.X_sptG], 
                                self._static_config, self.k_original, self.k_extrap, 
                                self.k_final, self.id_pad, self.l, self.m, 
                                P_window=p_window_array, C_window=c_window_value)

        # Functions with (P_orig, k_original) signature - these don't use window parameters
        _ = globals()['_get_P_Btype2'](dummy_P_orig, self.k_original)
        _ = globals()['_get_P_deltaE2'](dummy_P_orig, self.k_original)


        # Warm up core JPT functions
        # Warm up _b3nl_core with different window combinations 
        _ = _b3nl_core(self.X_spt, self._static_config, self.k_extrap, self.k_final, 
                    self.id_pad, self.l, self.m, dummy_P_for_transforms, 
                    P_window=None, C_window=None)
        _ = _b3nl_core(self.X_spt, self._static_config, self.k_extrap, self.k_final, 
                    self.id_pad, self.l, self.m, dummy_P_for_transforms, 
                    P_window=p_window_array, C_window=c_window_value)
                    
        # Warm up _lpt_NL_core
        _ = _lpt_NL_core(self.X_lpt, self.X_spt, self._static_config, self.k_extrap, 
                        self.k_final, self.id_pad, self.l, self.m, dummy_P_for_transforms, 
                        P_window=None, C_window=None)
        _ = _lpt_NL_core(self.X_lpt, self.X_spt, self._static_config, self.k_extrap, 
                        self.k_final, self.id_pad, self.l, self.m, dummy_P_for_transforms, 
                        P_window=p_window_array, C_window=c_window_value)
                        
        # Warm up _IA_tt_core
        _ = _IA_tt_core(self.X_IA_E, self.X_IA_B, self._static_config, self.k_extrap, 
                    self.k_final, self.id_pad, self.l, self.m, dummy_P_for_transforms, 
                    P_window=None, C_window=None)
        _ = _IA_tt_core(self.X_IA_E, self.X_IA_B, self._static_config, self.k_extrap, 
                    self.k_final, self.id_pad, self.l, self.m, dummy_P_for_transforms, 
                    P_window=p_window_array, C_window=c_window_value)
                    
        # Warm up _IA_mix_core
        _ = _IA_mix_core(self.X_IA_A, self.X_IA_DEE, self.X_IA_DBB, self._static_config, 
                        self.k_original, self.k_extrap, self.k_final, self.id_pad, 
                        self.l, self.m, dummy_P_for_transforms, 
                        P_window=None, C_window=None)
        _ = _IA_mix_core(self.X_IA_A, self.X_IA_DEE, self.X_IA_DBB, self._static_config, 
                        self.k_original, self.k_extrap, self.k_final, self.id_pad, 
                        self.l, self.m, dummy_P_for_transforms, 
                        P_window=p_window_array, C_window=c_window_value)
        
        # Warm up _IA_ta_core
        _ = _IA_ta_core(self.X_IA_deltaE1, self.X_IA_0E0E, self.X_IA_0B0B, self._static_config, 
                        self.k_original, self.k_extrap, self.k_final, 
                        self.id_pad, self.l, self.m, dummy_P_for_transforms,
                        P_window=None, C_window=None)
        _ = _IA_ta_core(self.X_IA_deltaE1, self.X_IA_0E0E, self.X_IA_0B0B, self._static_config, 
                        self.k_original, self.k_extrap, self.k_final, 
                        self.id_pad, self.l, self.m, dummy_P_for_transforms,
                        P_window=p_window_array, C_window=c_window_value)
        
        # Warm up _IA_ct_core
        _ = _IA_ct_core(self.X_spt, self.X_sptG, self.X_IA_tij_feG2, self.X_IA_tij_heG2, self.X_IA_A, self.X_IA_tij_F2F2, self.X_IA_deltaE1, self.X_IA_tij_G2G2, self.X_IA_tij_F2G2, 
                        self._static_config, self.k_original, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                        dummy_P_for_transforms, P_window=None, C_window=None)
        _ = _IA_ct_core(self.X_spt, self.X_sptG, self.X_IA_tij_feG2, self.X_IA_tij_heG2, self.X_IA_A, self.X_IA_tij_F2F2, self.X_IA_deltaE1, self.X_IA_tij_G2G2, self.X_IA_tij_F2G2, 
                        self._static_config, self.k_original, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                        dummy_P_for_transforms, P_window=p_window_array, C_window=c_window_value)
        
        # Warm up _gI_ct_core
        _ = _gI_ct_core(self.X_IA_gb2_F2, self.X_IA_gb2_G2, self.X_IA_gb2_S2F2, self.X_IA_gb2_S2G2, 
                        self._static_config, self.k_original, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                        dummy_P_for_transforms, P_window=None, C_window=None)
        _ = _gI_ct_core(self.X_IA_gb2_F2, self.X_IA_gb2_G2, self.X_IA_gb2_S2F2, self.X_IA_gb2_S2G2, 
                        self._static_config, self.k_original, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                        dummy_P_for_transforms, P_window=p_window_array, C_window=c_window_value)
        
        # Warm up _gI_ta_core
        _ = _gI_ta_core(self.X_IA_gb2_F2, self.X_IA_gb2_fe, self.X_IA_gb2_S2F2, self.X_IA_gb2_S2fe,
                        self._static_config, self.k_original, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                        dummy_P_for_transforms, P_window=None, C_window=None)
        _ = _gI_ta_core(self.X_IA_gb2_F2, self.X_IA_gb2_fe, self.X_IA_gb2_S2F2, self.X_IA_gb2_S2fe,    
                        self._static_config, self.k_original, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                        dummy_P_for_transforms, P_window=p_window_array, C_window=c_window_value)        
        
        # Warm up _gI_tt_core
        _ = _gI_tt_core(self.X_IA_gb2_S2he, self.X_IA_gb2_he, self._static_config, 
                        self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                        dummy_P_for_transforms, P_window=None, C_window=None)
        _ = _gI_tt_core(self.X_IA_gb2_S2he, self.X_IA_gb2_he, self._static_config, 
                        self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                        dummy_P_for_transforms, P_window=p_window_array, C_window=c_window_value)
        
        # Warm up _kPol_core
        # _ = _kPol_core(self.X_kP1, self.X_kP2, self.X_kP3, self._static_config,
        #                 self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
        #                 dummy_P_for_transforms, P_window=None, C_window=None)
        # _ = _kPol_core(self.X_kP1, self.X_kP2, self.X_kP3, self._static_config, 
        #                 self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
        #                 dummy_P_for_transforms, P_window=p_window_array, C_window=c_window_value)
        _ = self.kPol(dummy_P_orig, P_window=None, C_window=None)
        _ = self.kPol(dummy_P_orig, P_window=p_window_array, C_window=c_window_value)
        print("JIT warm-up completed.")

    @jax_cached_property
    def X_spt(self):
        return self.temp_fpt.X_spt
    @jax_cached_property
    def X_lpt(self):
        return self.temp_fpt.X_lpt
    @jax_cached_property
    def X_sptG(self):
        return self.temp_fpt.X_sptG
    @jax_cached_property
    def X_IA_A(self):
        return self.temp_fpt.X_IA_A
    @jax_cached_property
    def X_IA_B(self):
        return self.temp_fpt.X_IA_B
    @jax_cached_property
    def X_IA_E(self):
        return self.temp_fpt.X_IA_E
    @jax_cached_property
    def X_IA_DEE(self):
        return self.temp_fpt.X_IA_DEE
    @jax_cached_property
    def X_IA_DBB(self):
        return self.temp_fpt.X_IA_DBB
    @jax_cached_property
    def X_IA_deltaE1(self):
        return self.temp_fpt.X_IA_deltaE1
    @jax_cached_property
    def X_IA_0E0E(self):
        return self.temp_fpt.X_IA_0E0E
    @jax_cached_property
    def X_IA_0B0B(self):
        return self.temp_fpt.X_IA_0B0B
    @jax_cached_property
    def X_IA_gb2_fe(self):
        return self.temp_fpt.X_IA_gb2_fe
    @jax_cached_property
    def X_IA_gb2_he(self):
        return self.temp_fpt.X_IA_gb2_he
    @jax_cached_property
    def X_IA_tij_feG2(self):
        return self.temp_fpt.X_IA_tij_feG2
    @jax_cached_property
    def X_IA_tij_heG2(self):
        return self.temp_fpt.X_IA_tij_heG2
    @jax_cached_property
    def X_IA_tij_F2F2(self):
        return self.temp_fpt.X_IA_tij_F2F2
    @jax_cached_property
    def X_IA_tij_G2G2(self):
        return self.temp_fpt.X_IA_tij_G2G2
    @jax_cached_property
    def X_IA_tij_F2G2(self):
        return self.temp_fpt.X_IA_tij_F2G2
    @jax_cached_property
    def X_IA_gb2_F2(self):
        return self.temp_fpt.X_IA_gb2_F2
    @jax_cached_property
    def X_IA_gb2_G2(self):
        return self.temp_fpt.X_IA_gb2_G2
    @jax_cached_property
    def X_IA_gb2_S2F2(self):
        return self.temp_fpt.X_IA_gb2_S2F2
    @jax_cached_property
    def X_IA_gb2_S2fe(self):
        return self.temp_fpt.X_IA_gb2_S2fe
    @jax_cached_property
    def X_IA_gb2_S2he(self):
        return self.temp_fpt.X_IA_gb2_S2he
    @jax_cached_property
    def X_IA_gb2_S2G2(self):
        return self.temp_fpt.X_IA_gb2_S2G2
    @jax_cached_property
    def X_OV(self):
        return self.temp_fpt.X_OV
    @jax_cached_property
    def X_kP1(self):
        return self.temp_fpt.X_kP1
    @jax_cached_property
    def X_kP2(self):
        return self.temp_fpt.X_kP2
    @jax_cached_property
    def X_kP3(self):
        return self.temp_fpt.X_kP3


        
    @property
    def k_original(self):
        return self.__k_original
    
    @property
    def k_extrap(self):
        return self.__k_extrap
    
    @property
    def k_final(self):
        return self.__k_final
        
    
    def one_loop_dd_bias_b3nl(self, P, P_window=None, C_window=None):
        return _b3nl_core(self.X_spt, self._static_config, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                          P, P_window=P_window, C_window=C_window)

    def one_loop_dd_bias_lpt_NL(self, P, P_window=None, C_window=None):
        return _lpt_NL_core(self.X_lpt, self.X_spt, self._static_config, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                          P, P_window=P_window, C_window=C_window)
    
    def IA_tt(self, P, P_window=None, C_window=None):
        return _IA_tt_core(self.X_IA_E, self.X_IA_B, self._static_config, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                          P, P_window=P_window, C_window=C_window)

    def IA_mix(self, P, P_window=None, C_window=None):
        return _IA_mix_core(self.X_IA_A, self.X_IA_DEE, self.X_IA_DBB, self._static_config, self.k_original, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                          P, P_window=P_window, C_window=C_window)

    def IA_ta(self, P, P_window=None, C_window=None):
        return _IA_ta_core(self.X_IA_deltaE1, self.X_IA_0E0E, self.X_IA_0B0B, self._static_config, self.k_original, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                            P, P_window=P_window, C_window=C_window)

    def IA_ct(self, P, P_window=None, C_window=None):
        return _IA_ct_core(self.X_spt, self.X_sptG, self.X_IA_tij_feG2, self.X_IA_tij_heG2, self.X_IA_A, self.X_IA_tij_F2F2, self.X_IA_deltaE1, self.X_IA_tij_G2G2, self.X_IA_tij_F2G2, 
                           self._static_config, self.k_original, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                           P, P_window=P_window, C_window=C_window)

    def gI_ct(self, P, P_window=None, C_window=None):
        return _gI_ct_core(self.X_IA_gb2_F2, self.X_IA_gb2_G2, self.X_IA_gb2_S2F2, self.X_IA_gb2_S2G2, 
                           self._static_config, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                           P, P_window=P_window, C_window=C_window)

    def gI_ta(self, P, P_window=None, C_window=None):
        return _gI_ta_core(self.X_IA_gb2_F2, self.X_IA_gb2_fe, self.X_IA_gb2_S2F2, self.X_IA_gb2_S2fe,
                            self._static_config, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                            P, P_window=P_window, C_window=C_window)

    def gI_tt(self, P, P_window=None, C_window=None):
        return _gI_tt_core(self.X_IA_gb2_S2he, self.X_IA_gb2_he, self._static_config, 
                           self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                           P, P_window=P_window, C_window=C_window)

    def OV(self, P, P_window=None, C_window=None):
        return _OV_core(P, self.X_OV, self._static_config, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                       P_window=P_window, C_window=C_window)

    def IA_der(self, P):
        return (self.k_original**2)*P

    def kPol(self, P, P_window=None, C_window=None):
        return _kPol_core(self.X_kP1, self.X_kP2, self.X_kP3, self._static_config,
                          self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                          P, P_window=P_window, C_window=C_window)
    
    

    def diff(self, pk_method='jax-cosmo', pk_params={}, pk_diff_param='Omega_c', 
         function=None, P_window=None, C_window=None, diff_method=None, tangent=None):    
        
        func = getattr(self, function)
        def diff_func(param_value):
            P = self._pk_generator(pk_method, param_value, pk_diff_param, pk_params)
            P_new = func(P, P_window=P_window, C_window=C_window)
            return P_new
        
        param_value = pk_params.get(pk_diff_param, 0.12)  # Default value for Omega_c if not provided
        param_value = jnp.array(param_value, dtype=jnp.float64)
        if diff_method == 'jacfwd':
            result = jacfwd(diff_func)(param_value)
            return result
        elif diff_method == 'jvp':
            if tangent is None:
                tangent = jnp.ones_like(param_value, dtype=jnp.float64)
            primal_out, jvp_result = jvp(diff_func, (param_value,), (tangent,))
            return jvp_result, primal_out
        elif diff_method == 'vjp':
            primal_out, vjp_fn = vjp(diff_func, param_value)
            if tangent is None or not isinstance(tangent, tuple):
                tangent = tuple(jnp.ones_like(r) for r in primal_out)
            vjp_result = vjp_fn(tangent)[0]
            return vjp_result, primal_out
        else:
            raise ValueError(f"Unsupported differentiation method: {diff_method}. Use 'jacfwd', 'jvp', or 'vjp'.")

    def _pk_generator(self, pk_method, param_value, diff_param, P_params):
        if pk_method == 'jax-cosmo':
            return self._jax_cosmo_pk_generator(param_value, diff_param, P_params)
        elif pk_method == 'discoeb':
            return self._discoeb_pk_generator(param_value, diff_param, P_params)
        else:
            raise ValueError(f"Unsupported power spectrum generation method: {pk_method}")


    def _jax_cosmo_pk_generator(self, param_value, diff_param, P_params):
        """Generate power spectrum using jax-cosmo"""
        from jax_cosmo import Cosmology, power
        
        cosmo_dict = {
            'Omega_c': P_params.get('Omega_c', 0.12), 
            'Omega_b': P_params.get('Omega_b', 0.022),  
            'h': P_params.get('h', 0.69),
            'n_s': P_params.get('n_s', 0.96),
            'sigma8': P_params.get('sigma8', 0.8),
            'Omega_k': P_params.get('Omega_k', 0.0), 
            'w0': P_params.get('w0', -1.0),
            'wa': P_params.get('wa', 0.0),
        }

        if diff_param not in cosmo_dict:
            raise ValueError(f"Parameter '{diff_param}' not found in cosmology parameters.")
        
        # Explicitly use the differentiation parameter
        cosmo_dict[diff_param] = param_value
        new_cosmo = Cosmology(**cosmo_dict)
        return power.linear_matter_power(new_cosmo, self.k_original)
    
    def get_filter_jit():
        """Lazily import equinox.filter_jit only when needed."""
        import equinox
        return equinox.filter_jit

    @get_filter_jit()
    def _discoeb_pk_generator(self, param_value, diff_param, P_params):
        from discoeb.background import evolve_background
        from discoeb.perturbations import evolve_perturbations, get_power
        
        cosmo_dict = {
            'Omegam'  : P_params.get('Omegam', 0.3099),            # Total matter density parameter
            'Omegab'  : P_params.get('Omegab', 0.0488911),         # Baryon density parameter
            'w_DE_0'  : P_params.get('w_DE_0', -0.99),             # Dark energy equation of state parameter today
            'w_DE_a'  : P_params.get('w_DE_a', 0.0),              # Dark energy equation of state parameter time derivative
            'cs2_DE'  : P_params.get('cs2_DE', 1.0),               # Dark energy sound speed squared
            'Omegak'  : P_params.get('Omegak', 0.0),             # Curvature density parameter
            'A_s'     : P_params.get('A_s', 2.1e-9),        # Scalar amplitude of the primordial power spectrum
            'n_s'     : P_params.get('n_s', 0.96822),           # Scalar spectral index
            'H0'      : P_params.get('H0', 67.742),            # Hubble constant today in units of 100 km/s/Mpc
            'Tcmb'    : P_params.get('Tcmb', 2.7255),            # CMB temperature today in K
            'YHe'     : P_params.get('YHe', 0.248),             # Helium mass fraction
            'Neff'    : P_params.get('Neff', 2.046),             # Effective number of ultrarelativistic neutrinos
                                                        # -1 if massive neutrino present
            'Nmnu'    : P_params.get('Nmnu', 1),                # Number of massive neutrinos (must be 1 currently)
            'mnu'     : P_params.get('mnu', 0.06),              # Sum of neutrino masses in eV 
            'k_p'     : P_params.get('k_p', 0.05),              # Pivot scale in 1/Mpc
            }
        
        if diff_param not in cosmo_dict:
            raise ValueError(f"Parameter '{diff_param}' not found in cosmology parameters.")
        cosmo_dict[diff_param] = param_value
        #### This code comes directly from the discoeb example notebook ####

        # modes to sample
        nmodes = 512                         # number of modes to sample
        kmin   = jnp.min(self.k_original)                        # minimum k in 1/Mpc
        kmax   = jnp.max(self.k_original)                          # maximum k in 1/Mpc
        aexp   = 1.0                         # scale factor at which to evaluate the power spectrum
        
        ## Compute Background+thermal evolution
        param = evolve_background(param=cosmo_dict, thermo_module='RECFAST')

        # compute perturbation evolution
        aexp_out = jnp.array([aexp])
        # jax.profiler.start_trace("/tmp/tensorboard")
        y, kmodes = evolve_perturbations( param=param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out, 
                                            rtol=1e-3, atol=1e-3 , 
                                        )
        # jax.profiler.stop_trace()
        # turn perturbations into power spectra
        Pkm = get_power( k=kmodes, y=y[:,0,:], idx=4, param=param )

        return Pkm

        

@dataclass(frozen=True)
class StaticConfig:
    k_size: int
    n_pad: int
    N: int
    low_extrap: Optional[float] = None
    high_extrap: Optional[float] = None
    EK: Optional[Any] = None


@partial(jit, static_argnames=["EK"])
def _apply_extrapolation(*args, EK=None):
    if EK is None:
        return args if len(args) > 1 else args[0]
    return [EK.PK_original(var)[1] for var in args] if len(args) > 1 else EK.PK_original(args[0])[1]

##### Core for original functions #####
@partial(jit, static_argnames=["static_cfg"])
def _b3nl_core(X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
               P, P_window=None, C_window=None):
    P22_coef = jnp.array([2*1219/1470., 2*671/1029., 2*32/1715., 2*1/3., 2*62/35., 2*8/35., 1/3.])
    Ps, mat = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P22_mat = jnp.multiply(P22_coef, jnp.transpose(mat))
    P22 = jnp.sum(P22_mat, axis=1)
    P13 = P_13_reg(k_extrap, Ps)

    P_1loop = P22 + P13
    Pd1d2 = 2. * (17. / 21 * mat[0, :] + mat[4, :] + 4. / 21 * mat[1, :])
    Pd2d2 = 2. * (mat[0, :])
    Pd1s2 = 2. * (8. / 315 * mat[0, :] + 4. / 15 * mat[4, :] + 254. / 441 * mat[1, :] + 2. / 5 * mat[5,:] + 16. / 245 * mat[2,:])
    Pd2s2 = 2. * (2. / 3 * mat[1, :])
    Ps2s2 = 2. * (4. / 45 * mat[0, :] + 8. / 63 * mat[1, :] + 8. / 35 * mat[2, :])
    sig4 = jax.scipy.integrate.trapezoid(k_extrap ** 3 * Ps ** 2, x=jnp.log(k_extrap)) / (2. * jnp.pi ** 2)
    sig3nl = Y1_reg_NL(k_extrap, Ps)
    # return Ps
    P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig3nl = _apply_extrapolation(
        P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig3nl, EK=static_cfg.EK)
    
    return P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4, sig3nl

@partial(jit, static_argnames=["static_cfg"])
def _lpt_NL_core(X_lpt, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                 P, P_window=None, C_window=None):
    Ps, mat = J_k_scalar(P, X_lpt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    
    X1 = ((144. / 245.) * j000 - (176. / 343.) * j002 - (128. / 1715.) * j004 + (16. / 35.) * j1n11 - (
            16. / 35.) * j1n13)
    Y1 = Y1_reg_NL(k_extrap, Ps)
    Pb1L = X1 + Y1
    X2 = ((16. / 21.) * j000 - (16. / 21.) * j002 + (16. / 35.) * j1n11 - (16. / 35.) * j1n13)
    Y2 = Y2_reg_NL(k_extrap, Ps)
    Pb1L_2 = X2 + Y2
    X3 = (50. / 21.) * j000 + 2. * j1n11 - (8. / 21.) * j002
    Pb1L_b2L = X3
    X4 = (34. / 21.) * j000 + 2. * j1n11 + (8. / 21.) * j002
    Pb2L = X4
    X5 = j000
    Pb2L_2 = X5
    Ps, _ = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    sig4 = jax.scipy.integrate.trapezoid(k_extrap ** 3 * Ps ** 2, x=jnp.log(k_extrap)) / (2. * jnp.pi ** 2)

    Ps, Pb1L, Pb1L_2, Pb1L_b2L, Pb2L, Pb2L_2 = _apply_extrapolation(Ps, Pb1L, Pb1L_2, Pb1L_b2L, Pb2L, Pb2L_2, EK=static_cfg.EK)
    return Ps, Pb1L, Pb1L_2, Pb1L_b2L, Pb2L, Pb2L_2, sig4

@partial(jit, static_argnames=["static_cfg"])
def _IA_tt_core(X_IA_E, X_IA_B, static_cfg, k_extrap, k_final, id_pad, l, m,
                P, P_window=None, C_window=None):
    P_E, _ = J_k_tensor(P, X_IA_E, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_B, _ = J_k_tensor(P, X_IA_B, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_E, P_B = _apply_extrapolation(P_E, P_B, EK=static_cfg.EK)
    return 2 * P_E, 2 * P_B

@partial(jit, static_argnames=["static_cfg"])
def _IA_mix_core(X_IA_A, X_IA_DEE, X_IA_DBB, static_cfg, k_original, k_extrap, k_final, id_pad, l, m,
                P, P_window=None, C_window=None):
    P_A, _ = J_k_tensor(P, X_IA_A, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_DEE, _ = J_k_tensor(P, X_IA_DEE, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_DBB, _ = J_k_tensor(P, X_IA_DBB, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_A, P_DEE, P_DBB = _apply_extrapolation(P_A, P_DEE, P_DBB, EK=static_cfg.EK)
    P_Btype2 = P_IA_B(k_original, P)
    return 2 * P_A, 4 * P_Btype2, 2 * P_DEE, 2 * P_DBB

@partial(jit, static_argnames=["static_cfg"])
def _IA_ta_core(X_IA_deltaE1, X_IA_0E0E, X_IA0B0B, static_cfg, k_original, k_extrap, k_final, id_pad, l, m,
                P, P_window=None, C_window=None):
    P_deltaE1, _ = J_k_tensor(P, X_IA_deltaE1, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_deltaE2 = P_IA_deltaE2(k_original, P)
    P_0E0E, _ = J_k_tensor(P, X_IA_0E0E, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_0B0B, _ = J_k_tensor(P, X_IA0B0B, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_deltaE1, P_0E0E, P_0B0B = _apply_extrapolation(P_deltaE1, P_0E0E, P_0B0B, EK=static_cfg.EK)
    return 2 * P_deltaE1, 2 * P_deltaE2, P_0E0E, P_0B0B

@partial(jit, static_argnames=["static_cfg"])
def _IA_ct_core(X_spt, X_sptG, X_IA_tij_feG2, X_IA_tij_heG2, X_IA_A, X_IA_tij_F2F2, X_IA_deltaE1, X_IA_tij_G2G2, X_IA_tij_F2G2, 
                static_cfg, k_original, k_extrap, k_final, id_pad, l, m,
                P, P_window=None, C_window=None):
    
    Ps, mat = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    one_loop_coef = jnp.array(
        [2 * 1219 / 1470., 2 * 671 / 1029., 2 * 32 / 1715., 2 * 1 / 3., 2 * 62 / 35., 2 * 8 / 35., 1 / 3.])
    P22_mat = jnp.multiply(one_loop_coef, jnp.transpose(mat))
    P_22F = jnp.sum(P22_mat, 1)

    one_loop_coefG= jnp.array(
        [2*1003/1470, 2*803/1029, 2*64/1715, 2*1/3, 2*58/35, 2*12/35, 1/3])
    PsG, matG = J_k_scalar(P, X_sptG, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P22G_mat = jnp.multiply(one_loop_coefG, jnp.transpose(matG))
    P_22G = jnp.sum(P22G_mat, 1)
    P_22F, P_22G = _apply_extrapolation(P_22F, P_22G, EK=static_cfg.EK)
    P_13G = P_IA_13G(k_original,P,)
    P_13F = P_IA_13F(k_original, P)
    P_0tE = P_22G-P_22F+P_13G-P_13F
    P_0tE = 2*P_0tE

    P_feG2, A = J_k_tensor(P, X_IA_tij_feG2, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_feG2 = _apply_extrapolation(P_feG2, EK=static_cfg.EK)
    # P_A00E = compute_term(P, X_IA_deltaE1, static_cfg, k_extrap, k_final, id_pad, l, m,
    #                      P_window=P_window, C_window=C_window, operation=lambda x: 2 * x)
    P_A00E = J_k_tensor(P, X_IA_deltaE1, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)[0]
    P_A00E = _apply_extrapolation(P_A00E, EK=static_cfg.EK)
    P_A00E = jnp.multiply(P_A00E, 2)
    P_0EtE = jnp.subtract(P_feG2,(1/2)*P_A00E)
    P_0EtE = 2*P_0EtE

    P_heG2, A = J_k_tensor(P, X_IA_tij_heG2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_heG2 = _apply_extrapolation(P_heG2, EK=static_cfg.EK)
    # P_A0E2 = compute_term(P, X_IA_A, static_cfg, k_extrap, k_final, id_pad, l, m,
    #                      P_window=P_window, C_window=C_window, operation=lambda x: 2 * x)
    P_A0E2 = J_k_tensor(P, X_IA_A, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)[0]
    P_A0E2 = _apply_extrapolation(P_A0E2, EK=static_cfg.EK)
    P_A0E2 = jnp.multiply(P_A0E2, 2)
    P_E2tE = jnp.subtract(P_heG2,(1/2)*P_A0E2)
    P_E2tE = 2*P_E2tE

    P_F2F2, A = J_k_tensor(P,X_IA_tij_F2F2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_G2G2, A = J_k_tensor(P,X_IA_tij_G2G2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_F2G2, A = J_k_tensor(P,X_IA_tij_F2G2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_F2F2, P_G2G2, P_F2G2 = _apply_extrapolation(P_F2F2, P_G2G2, P_F2G2, EK=static_cfg.EK)
    P_tEtE = P_F2F2+P_G2G2-2*P_F2G2
    P_tEtE = 2*P_tEtE

    return P_0tE, P_0EtE, P_E2tE, P_tEtE

def _gI_ct_core(X_IA_gb2_F2, X_IA_gb2_G2, X_IA_gb2_S2F2, X_IA_gb2_S2G2, 
                static_cfg, k_extrap, k_final, id_pad, l, m,
                P, P_window=None, C_window=None):
    P_F2, _ = J_k_tensor(P, X_IA_gb2_F2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_G2, _ = J_k_tensor(P, X_IA_gb2_G2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_F2 = _apply_extrapolation(P_F2, EK=static_cfg.EK)
    P_G2 = _apply_extrapolation(P_G2, EK=static_cfg.EK)
    P_d2tE = 2 * (P_G2 - P_F2)

    P_S2F2, _ = J_k_tensor(P, X_IA_gb2_S2F2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_S2G2, _ = J_k_tensor(P, X_IA_gb2_S2G2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_S2F2 = _apply_extrapolation(P_S2F2, EK=static_cfg.EK)
    P_S2G2 = _apply_extrapolation(P_S2G2, EK=static_cfg.EK)
    P_s2tE = 2 * (P_S2G2 - P_S2F2)

    return P_d2tE, P_s2tE

def _gI_ta_core(X_IA_gb2_F2, X_IA_gb2_fe, X_IA_gb2_S2F2, X_IA_gb2_S2fe, 
                static_cfg, k_extrap, k_final, id_pad, l, m,
                P, P_window=None, C_window=None):
    
    P_d2E, _ = J_k_tensor(P, X_IA_gb2_F2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_d20E, _ = J_k_tensor(P, X_IA_gb2_fe, static_cfg, k_extrap, k_final, id_pad, l, m,
                            P_window=P_window, C_window=C_window)
    P_s2E, _ = J_k_tensor(P, X_IA_gb2_S2F2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_s20E, _ = J_k_tensor(P, X_IA_gb2_S2fe, static_cfg, k_extrap, k_final, id_pad, l, m,
                            P_window=P_window, C_window=C_window)
    P_d2E, P_d20E, P_s2E, P_s20E = _apply_extrapolation(P_d2E, P_d20E, P_s2E, P_s20E, EK=static_cfg.EK)
    return 2 * P_d2E, 2 * P_d20E, 2 * P_s2E, 2 * P_s20E

def _gI_tt_core(X_IA_gb2_S2he, X_IA_gb2_he, static_cfg, k_extrap, k_final, id_pad, l, m,
                P, P_window=None, C_window=None):
    P_s2E2, _ = J_k_tensor(P, X_IA_gb2_S2he, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_d2E2, _ = J_k_tensor(P, X_IA_gb2_he, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_s2E2, P_d2E2 = _apply_extrapolation(P_s2E2, P_d2E2, EK=static_cfg.EK)
    P_s2E2 = 2 * P_s2E2
    P_d2E2 = 2 * P_d2E2
    return P_s2E2, P_d2E2

def _kPol_core(X_kP1, X_kP2, X_kP3, static_cfg, k_extrap, k_final, id_pad, l, m,
               P, P_window=None, C_window=None):
    P_kP1, _ = J_k_tensor(P, X_kP1, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_kP2, _ = J_k_tensor(P, X_kP2, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_kP3, _ = J_k_tensor(P, X_kP3, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_kP1 = _apply_extrapolation(P_kP1, EK=static_cfg.EK)
    P_kP2 = _apply_extrapolation(P_kP2, EK=static_cfg.EK)
    P_kP3 = _apply_extrapolation(P_kP3, EK=static_cfg.EK)
    return P_kP1 / (80 * jnp.pi ** 2), P_kP2 / (160 * jnp.pi ** 2), P_kP3 / (80 * jnp.pi ** 2)

@partial(jit, static_argnames=["static_cfg"])
def _OV_core(P, X_OV, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    P, _ = J_k_tensor(P, X_OV, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P = _apply_extrapolation(P, EK=static_cfg.EK)
    P_OV = P * (2 * jnp.pi) ** 2
    return P_OV


#### Low level computational functions ####
@partial(jit, static_argnames=["static_cfg"])
def J_k_scalar(P, X, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray, 
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray, 
               P_window=None, C_window=None):
    pf, p, g_m, g_n, two_part_l, h_l = X

    if (static_cfg.low_extrap is not None):
        P = static_cfg.EK.extrap_P_low(P)

    if (static_cfg.high_extrap is not None):
        P = static_cfg.EK.extrap_P_high(P)
    nu = -2
    P_b = P * k_extrap ** (-nu)

    if (static_cfg.n_pad > 0):
        P_b = jnp.pad(P_b, pad_width=(static_cfg.n_pad, static_cfg.n_pad), mode='constant', constant_values=0)

    c_m = fourier_coefficients(P_b, m, static_cfg.N, C_window)

    A_out = jnp.zeros((pf.shape[0], static_cfg.k_size))

    def process_single_row(i):
        C_l = convolution(c_m, c_m, g_m[i], g_n[i], h_l[i], None if two_part_l is None else two_part_l[i])

        l_size = l.shape[0]
        l_midpoint = l_size // 2

        c_plus = C_l[l_midpoint:]
        c_minus = C_l[:l_midpoint]

        C_l_combined = jnp.concatenate([c_plus[:-1], c_minus])

        A_k = ifft(C_l_combined) * C_l_combined.size

        stride = max(1, A_k.shape[0] // static_cfg.k_size)

        return jnp.real(A_k[::stride][:static_cfg.k_size]) * pf[i] * k_final ** (-p[i] - 2)

    rows = jnp.arange(pf.shape[0])
    A_out = jax.vmap(process_single_row)(rows)

    m_midpoint = (m.shape[0] + 1) // 2
    c_m_positive = c_m[m_midpoint-1:]

    P_out = irfft(c_m_positive) * k_final ** nu * static_cfg.N

    if (static_cfg.n_pad > 0):
        P_out = P_out[id_pad]
        A_out = A_out[:, id_pad]

    return P_out, A_out


@partial(jit, static_argnames=["static_cfg"])
def J_k_tensor(P, X, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,
               P_window=None, C_window=None):
    
    pf, p, nu1, nu2, g_m, g_n, h_l = X

    if (static_cfg.low_extrap is not None):
        P = static_cfg.EK.extrap_P_low(P)
    
    if (static_cfg.high_extrap is not None):
        P = static_cfg.EK.extrap_P_high(P)
    
    l_midpoint = l.shape[0] // 2

    def process_single_index(i):
        nu1_i = nu1[i]
        nu2_i = nu2[i]
        pf_i = pf[i]
        p_i = p[i]
        g_m_i = g_m[i]
        g_n_i = g_n[i]
        h_l_i = h_l[i]
        
        P_b1 = P * k_extrap ** (-nu1_i)
        P_b2 = P * k_extrap ** (-nu2_i)
        
        if P_window is not None:
            W = p_window(k_extrap, P_window[0], P_window[1])
            P_b1 = P_b1 * W
            P_b2 = P_b2 * W
            
        if static_cfg.n_pad > 0:
            P_b1 = jnp.pad(P_b1, pad_width=(static_cfg.n_pad, static_cfg.n_pad), mode='constant', constant_values=0)
            P_b2 = jnp.pad(P_b2, pad_width=(static_cfg.n_pad, static_cfg.n_pad), mode='constant', constant_values=0)
            
        c_m = fourier_coefficients(P_b1, m, static_cfg.N, C_window)
        c_n = fourier_coefficients(P_b2, m, static_cfg.N, C_window)
        
        C_l = convolution(c_m, c_n, g_m_i, g_n_i, h_l_i)
        
        c_plus = C_l[l_midpoint:]
        c_minus = C_l[:l_midpoint]
        C_l_combined = jnp.concatenate([c_plus[:-1], c_minus])

        A_k = jnp.fft.ifft(C_l_combined) * C_l_combined.size
        return jnp.real(A_k[::2]) * pf_i * k_final ** p_i
    
    indices = jnp.arange(pf.size)
    A_out = jax.vmap(process_single_index)(indices)
    
    P_fin = jnp.sum(A_out, axis=0)
    
    if static_cfg.n_pad > 0:
        P_fin = P_fin[id_pad]
        A_out = A_out[:, id_pad]
    
    return P_fin, A_out

@jit
def fourier_coefficients(P_b, m, N, C_window=None):
    from jax.numpy.fft import rfft

    c_m_positive = rfft(P_b)
    c_m_positive = c_m_positive.at[-1].set(c_m_positive[-1] / 2.0)
    c_m_negative = jnp.conjugate(c_m_positive[1:])
    c_m = jnp.hstack((c_m_negative[::-1], c_m_positive)) / jnp.float64(N)
    
    if C_window is not None:
        window_size = jnp.array(C_window * N / 2.0, dtype=int)
        c_m = c_m * c_window(m, window_size)
        
    return c_m

@jit
def convolution(c1, c2, g_m, g_n, h_l, two_part_l=None):
    from jax.scipy.signal import fftconvolve

    C_l = fftconvolve(c1 * g_m, c2 * g_n)

    if two_part_l is not None:
        C_l = C_l * h_l * two_part_l
    else:
        C_l = C_l * h_l

    return C_l

from memory_profiler import profile

@profile
def create_jaxpt():
    k = jnp.logspace(-3, 1, 1000)
    return JAXPT(k, low_extrap=-5, high_extrap=5, n_pad=int(0.5*len(k)))

if __name__ == "__main__":
    #jpt = create_jaxpt() # << For memory profiler

    k = jnp.logspace(-3, 1, 1000)
    jpt = JAXPT(k, low_extrap=-5, high_extrap=5, n_pad=int(0.5*len(k)))
    vjp_result = jpt.diff(function='IA_tt', diff_method='vjp')
    jvp_result = jpt.diff(function='IA_tt', diff_method='jvp')
    jacfwd_result = jpt.diff(function='IA_tt', diff_method='jacfwd')
    P = jpt._jax_cosmo_pk_generator(param_value=0.12, diff_param='Omega_c', P_params={})
    result = jpt.IA_tt(P)

    print(np.allclose(vjp_result[1], result))
    print(np.allclose(jvp_result[1], result))

    # Result and diff result are a tuple of two arrays, plot them to visualize
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # plt.plot(jpt.k_original, result[1], label='IA_tt Result', color='blue')
    # plt.plot(jpt.k_original, diff_result[1], label='IA_tt Diff Result', color='red', linestyle='--')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('k (1/Mpc)')
    # plt.ylabel('Power Spectrum')
    # plt.title('IA_tt Power Spectrum and its Derivative')
    # plt.legend()
    # plt.grid()
    # plt.show()
