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


        self.term_config = {
            # Standard compute_term cases
            "P_E": {"type": "standard", "X": "X_IA_E", "operation": lambda x: 2 * x},
            "P_B": {"type": "standard", "X": "X_IA_B", "operation": lambda x: 2 * x},
            
            "P_A": {"type": "standard", "X": "X_IA_A", "operation": lambda x: 2 * x},
            "P_DEE": {"type": "standard", "X": "X_IA_DEE", "operation": lambda x: 2 * x},
            "P_DBB": {"type": "standard", "X": "X_IA_DBB", "operation": lambda x: 2 * x},
            
            "P_deltaE1": {"type": "standard", "X": "X_IA_deltaE1", "operation": lambda x: 2 * x},
            "P_0E0E": {"type": "standard", "X": "X_IA_0E0E"},
            "P_0B0B": {"type": "standard", "X": "X_IA_0B0B"},
            
            "P_s2E2": {"type": "standard", "X": "X_IA_gb2_S2he", "operation": lambda x: 2 * x},
            "P_d2E2": {"type": "standard", "X": "X_IA_gb2_he", "operation": lambda x: 2 * x},
            
            "P_d2E": {"type": "standard", "X": "X_IA_gb2_F2", "operation": lambda x: 2 * x},
            "P_d20E": {"type": "standard", "X": "X_IA_gb2_fe", "operation": lambda x: 2 * x},
            "P_s2E": {"type": "standard", "X": "X_IA_gb2_S2F2", "operation": lambda x: 2 * x},
            "P_s20E": {"type": "standard", "X": "X_IA_gb2_S2fe", "operation": lambda x: 2 * x},
            
            "P_kP1": {"type": "standard", "X": "X_kP1", "operation": lambda x: x / (80 * jnp.pi ** 2)},
            "P_kP2": {"type": "standard", "X": "X_kP2", "operation": lambda x: x / (160 * jnp.pi ** 2)},
            "P_kP3": {"type": "standard", "X": "X_kP3", "operation": lambda x: x / (80 * jnp.pi ** 2)},
            
            # Special cases/unique terms
            "P_Btype2": {"type": "special", "method": "_get_P_Btype2", "X": None},
            "P_deltaE2": {"type": "special", "method": "_get_P_deltaE2", "X": None},
            "P_OV": {"type": "special", "method": "OV", "X": "X_OV"},
            
            "P_0tE": {"type": "special", "method": "_get_P_0tE", "X": ["X_spt", "X_sptG"]},
            "P_0EtE": {"type": "special", "method": "_get_P_0EtE", "X": ["X_IA_tij_feG2", "X_IA_deltaE1"]}, 
            "P_E2tE": {"type": "special", "method": "_get_P_E2tE", "X": ["X_IA_tij_heG2", "X_IA_A"]},
            "P_tEtE": {"type": "special", "method": "_get_P_tEtE", "X": ["X_IA_tij_F2F2", "X_IA_tij_G2G2", "X_IA_tij_F2G2"]},
            
            "P_1loop": {"type": "special", "method": "_get_1loop", "X": "X_spt"},
            
            "Pd1d2": {"type": "special", "method": "_get_Pd1d2", "X": "X_spt"},
            "Pd2d2": {"type": "special", "method": "_get_Pd2d2", "X": "X_spt"},
            "Pd1s2": {"type": "special", "method": "_get_Pd1s2", "X": "X_spt"},
            "Pd2s2": {"type": "special", "method": "_get_Pd2s2", "X": "X_spt"},
            "Ps2s2": {"type": "special", "method": "_get_Ps2s2", "X": "X_spt"},
            
            "sig4": {"type": "special", "method": "_get_sig4", "X": "X_spt"},
            "sig3nl": {"type": "special", "method": "_get_sig3nl", "X": "X_spt"},
            
            "Pb1L": {"type": "special", "method": "_get_Pb1L", "X": "X_lpt"},
            "Pb1L_2": {"type": "special", "method": "_get_Pb1L_2", "X": "X_lpt"},
            "Pb1L_b2L": {"type": "special", "method": "_get_Pb1L_b2L", "X": "X_lpt"},
            "Pb2L": {"type": "special", "method": "_get_Pb2L", "X": "X_lpt"},
            "Pb2L_2": {"type": "special", "method": "_get_Pb2L_2", "X": "X_lpt"},
            
            "P_d2tE": {"type": "special", "method": "_get_P_d2tE", "X": ["X_IA_gb2_F2", "X_IA_gb2_G2"]},
            "P_s2tE": {"type": "special", "method": "_get_P_s2tE", "X": ["X_IA_gb2_S2F2", "X_IA_gb2_S2G2"]},

            "P_OV": {"type": "special", "method": "_get_OV", "X": "X_OV"},
        }

        self.term_groups = {
            "IA_tt": ["P_E", "P_B"],
            "IA_mix": ["P_A", "P_Btype2", "P_DEE", "P_DBB"],
            "IA_ta": ["P_deltaE1", "P_deltaE2", "P_0E0E", "P_0B0B"],
            "IA_ct": ["P_0tE", "P_0EtE", "P_E2tE", "P_tEtE"],
            "gI_ct": ["P_d2tE", "P_s2tE"],
            "gI_ta": ["P_d2E", "P_d20E", "P_s2E", "P_s20E"],
            "gI_tt": ["P_s2E2", "P_d2E2"],
            "kPol": ["P_kP1", "P_kP2", "P_kP3"],
            "one_loop_dd_bias_b3nl": ["P_1loop", "Pd1d2", "Pd2d2", "Pd1s2", "Pd2s2", "Ps2s2", "sig4", "sig3nl"],
            "one_loop_dd_bias_lpt_NL": ["Pb1L", "Pb1L_2", "Pb1L_b2L", "Pb2L", "Pb2L_2", "sig4"]
        }

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
        self.X_spt = process_x_term(self.temp_fpt.X_spt)
        self.X_lpt = process_x_term(self.temp_fpt.X_lpt)
        #TODO may be able to add these back as cached properties, check differentiability

        if warmup: self._warm_up_jit_functions()

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
        
        print("JIT warm-up completed.")



    @jax_cached_property
    def X_sptG(self):
        return self.temp_fpt.X_sptG
    @jax_cached_property
    def X_cleft(self):
        return self.temp_fpt.X_cleft
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
    def X_IA_tij_F2G2reg(self):
        return self.temp_fpt.X_IA_tij_F2G2reg
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
    @jax_cached_property
    def X_RSDA(self):
        return self.temp_fpt.X_RSDA
    @jax_cached_property
    def X_RSDB(self):
        return self.temp_fpt.X_RSDB


        
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
        return tuple(self.get(t, P, P_window=P_window, C_window=C_window) for t in self.term_groups["one_loop_dd_bias_b3nl"])

    def one_loop_dd_bias_lpt_NL(self, P, P_window=None, C_window=None):
        return tuple(self.get(t, P, P_window=P_window, C_window=C_window) for t in self.term_groups["one_loop_dd_bias_lpt_NL"])

    def IA_tt(self, P, P_window=None, C_window=None):
        return tuple(self.get(t, P, P_window=P_window, C_window=C_window) for t in self.term_groups["IA_tt"])

    def IA_mix(self, P, P_window=None, C_window=None):
        return tuple(self.get(t, P, P_window=P_window, C_window=C_window) for t in self.term_groups["IA_mix"])

    def IA_ta(self, P, P_window=None, C_window=None):
        return tuple(self.get(t, P, P_window=P_window, C_window=C_window) for t in self.term_groups["IA_ta"])

    def IA_ct(self, P, P_window=None, C_window=None):
        return tuple(self.get(t, P, P_window=P_window, C_window=C_window) for t in self.term_groups["IA_ct"])

    def gI_ct(self, P, P_window=None, C_window=None):
        return tuple(self.get(t, P, P_window=P_window, C_window=C_window) for t in self.term_groups["gI_ct"])

    def IA_gb2(self, P, P_window=None, C_window=None):
        return tuple(self.get(t, P, P_window=P_window, C_window=C_window) for t in self.term_groups["IA_gb2"])

    def gI_ta(self, P, P_window=None, C_window=None):
        return tuple(self.get(t, P, P_window=P_window, C_window=C_window) for t in self.term_groups["gI_ta"])

    def gI_tt(self, P, P_window=None, C_window=None):
        return tuple(self.get(t, P, P_window=P_window, C_window=C_window) for t in self.term_groups["gI_tt"])

    def OV(self, P, C_window=None):
        return self.get("P_OV", P, C_window=C_window)

    def IA_der(self, P):
        return (self.k_original**2)*P

    def kPol(self, P, P_window=None, C_window=None):
        return tuple(self.get(t, P, P_window=P_window, C_window=C_window) for t in self.term_groups["kPol"])
    
    def get(self, term, P, P_window=None, C_window=None):
        """
        Get computed term(s) using JAX-compatible dispatch
        
        Parameters:
        -----------
        term : str
            Term or group name to compute
        P : array
            Input power spectrum
        P_window, C_window : optional
            Window parameters
        """
        P = jnp.asarray(P, dtype=jnp.float64)
        #Precompute P_window
        if P_window is not None:
            P_window = p_window(self.k_extrap, P_window[0], P_window[1])

        if term in self.term_groups: 
            return tuple(self.get(t, P, P_window=P_window, C_window=C_window) for t in self.term_groups[term])
        
        if term not in self.term_config:
            raise ValueError(f"Unknown term: {term}")
            
        config = self.term_config[term]
        
        if config["type"] == "standard":
            X = getattr(self, config["X"]) #TODO what about when its multiple X's
            operation = config.get("operation")
            return compute_term(P, X, self._static_config, self.k_extrap, self.k_final, 
                                self.id_pad, self.l, self.m,
                                P_window=P_window, C_window=C_window, operation=operation)
        
        elif config["type"] == "special":
            method_name = config["method"]
            method = globals()[method_name]
            X = config["X"]
            if isinstance(X, str):
                X = getattr(self, X)
            elif isinstance(X, list):
                X = [getattr(self, x) for x in X]
            else: #X is None, P_btype2 and P_deltaE2 case
                return method(P, self.k_original)

            if method_name == "_get_P_0tE":
                return method(
                    P, X,
                    static_cfg=self._static_config, k_original=self.k_original,
                    k_extrap=self.k_extrap, k_final=self.k_final, id_pad=self.id_pad,
                    l=self.l, m=self.m, P_window=P_window, C_window=C_window
                )
            
            result = method(
                P, X,
                static_cfg=self._static_config,
                k_extrap=self.k_extrap,
                k_final=self.k_final,
                id_pad=self.id_pad,
                l=self.l,
                m=self.m,
                P_window=P_window,
                C_window=C_window
            )
            return result
                
        raise ValueError(f"Unable to process term: {term}")
    

    def vjp_get(self, term, P, P_window=None, C_window=None, tangent_type='ones', tangent_vector=None, seed=42):
        """
        Compute vector-Jacobian product (gradient) of the requested term with respect to input P.
        
        Parameters:
        -----------
        term : str
            Term or group name to compute
        P : array
            Input power spectrum
        P_window, C_window : optional
            Window parameters
        tangent_type : str, optional
            Type of tangent vector to use for gradient computation:
            - 'ones': Use vector of ones (gradient of sum of outputs)
            - 'random': Use random normal vector (explore random direction in output space)
            - 'custom': Use provided tangent_vector
            - 'unit_i': Use unit vector for single component (i provided in tangent_vector)
            - 'normalized': Ones vector normalized by output size (for scale-invariant gradients)
        tangent_vector : array_like, optional
            Custom tangent vector or index parameter depending on tangent_type
        seed : int, optional
            Random seed when tangent_type='random'
            
        Returns:
        --------
        tuple: (result, gradient)
            - result: The output of the term computation
            - gradient: The gradient of the term with respect to P
        """
        def get_wrt_P(P):
            return self.get(term, P, P_window=P_window, C_window=C_window)
        
        result, vjp_function = vjp(get_wrt_P, P)
        
        if tangent_type == 'ones':
            # BENEFIT: Simple gradient of sum - useful for overall sensitivity analysis
            # and when optimizing the total value across all output elements
            if isinstance(result, tuple):
                tangent = tuple(jnp.ones_like(r) for r in result)
            else:
                tangent = jnp.ones_like(result)
        
        elif tangent_type == 'random':
            # BENEFIT: Explores random directions in gradient space - excellent for
            # detecting numerical instabilities and testing overall differentiability
            key = jax.random.PRNGKey(seed)
            if isinstance(result, tuple):
                keys = jax.random.split(key, len(result))
                tangent = tuple(jax.random.normal(k, r.shape) for k, r in zip(keys, result))
            else:
                tangent = jax.random.normal(key, result.shape)
        
        elif tangent_type == 'custom':
            # BENEFIT: Allows exploring specific physics directions or
            # targeting particular features in spectral space
            if tangent_vector is None:
                raise ValueError("tangent_vector must be provided when tangent_type='custom'")
            tangent = tangent_vector
        
        elif tangent_type == 'unit_i':
            # BENEFIT: Computes gradient for a single output element - useful for
            # understanding how specific spectral regions are influenced
            if tangent_vector is None:
                raise ValueError("tangent_vector (index) must be provided when tangent_type='unit_i'")
            
            if isinstance(result, tuple):
                # For tuple outputs, tangent_vector should be a tuple of (tuple_index, element_index)
                tuple_idx, element_idx = tangent_vector
                tangent = tuple(jnp.zeros_like(r) for r in result)
                # Create a one-hot vector for the specified element
                unit_vec = jnp.zeros_like(result[tuple_idx])
                unit_vec = unit_vec.at[element_idx].set(1.0)
                # Replace the zeros at tuple_idx with the one-hot vector
                tangent = list(tangent)
                tangent[tuple_idx] = unit_vec
                tangent = tuple(tangent)
            else:
                # For single array output, tangent_vector is the index
                unit_vec = jnp.zeros_like(result)
                unit_vec = unit_vec.at[tangent_vector].set(1.0)
                tangent = unit_vec
        
        elif tangent_type == 'normalized':
            # BENEFIT: Scale-invariant gradients - especially important when comparing
            # sensitivity across different terms with varying output sizes
            if isinstance(result, tuple):
                tangent = tuple(jnp.ones_like(r) / jnp.prod(jnp.array(r.shape)) for r in result)
            else:
                tangent = jnp.ones_like(result) / jnp.prod(jnp.array(result.shape))
        
        else:
            raise ValueError(f"Unknown tangent_type: {tangent_type}")
        
        gradient = vjp_function(tangent)
        
        return result, gradient[0]  # [0] because vjp_fn returns a tuple of gradients (in this case there's only one element)

    def jvp_get(self, term, P, tangent_P=None, P_window=None, C_window=None, 
                tangent_type='custom', tangent_params=None, scaling=1.0, seed=42):
        """
        Compute Jacobian-vector product (directional derivative) of the requested term 
        with respect to a perturbation of input P in the direction of tangent_P.
        
        Parameters:
        -----------
        term : str
            Term or group name to compute
        P : array
            Input power spectrum
        tangent_P : array, optional
            Custom tangent vector specifying direction in input space
        P_window, C_window : optional
            Window parameters
        tangent_type : str, optional
            Type of tangent vector to use:
            - 'custom': Use provided tangent_P (default)
            - 'random': Use random normal vector to explore sensitivity across frequencies
            - 'unit_i': Use unit vector for single component to test sensitivity to a specific k bin
            - 'sinusoidal': Use sinusoidal perturbation to test frequency-dependent response
            - 'gaussian': Use gaussian perturbation to test localized response
        tangent_params : dict or int, optional
            Parameters for specialized tangent vectors:
            - For 'unit_i': Index of element to perturb
            - For 'sinusoidal': {'amplitude': float, 'frequency': float, 'phase': float}
            - For 'gaussian': {'center': float, 'width': float, 'amplitude': float}
            - Common: {'normalize': bool} to normalize the tangent vector
        scaling : float, optional
            Scaling factor for the tangent vector
        seed : int, optional
            Random seed when tangent_type='random'
            
        Returns:
        --------
        tuple: (result, jvp_value, tangent_P)
            - result: The output of the term computation
            - jvp_value: The directional derivative in the direction of tangent_P
            - tangent_P: The actual tangent vector used (useful for auto-generated tangents)
        """
        # Prepare the function to differentiate
        def get_wrt_P(P):
            return self.get(term, P, P_window=P_window, C_window=C_window)
        
        # Create the tangent vector based on tangent_type
        if tangent_type == 'custom':
            if tangent_P is None:
                raise ValueError("tangent_P must be provided when tangent_type='custom'")
        elif tangent_type == 'random':
            key = jax.random.PRNGKey(seed)
            tangent_P = jax.random.normal(key, P.shape)
        elif tangent_type == 'unit_i':
            if not isinstance(tangent_params, int):
                raise ValueError("tangent_params must be an integer index when tangent_type='unit_i'")
            tangent_P = jnp.zeros_like(P)
            tangent_P = tangent_P.at[tangent_params].set(1.0)
        elif tangent_type == 'sinusoidal':
            if tangent_params is None:
                tangent_params = {'amplitude': 1.0, 'frequency': 5.0/len(P), 'phase': 0.0}
            k_indices = jnp.arange(len(P))
            amplitude = tangent_params.get('amplitude', 1.0)
            frequency = tangent_params.get('frequency', 5.0/len(P))
            phase = tangent_params.get('phase', 0.0)
            tangent_P = amplitude * jnp.sin(frequency * k_indices + phase)
        elif tangent_type == 'gaussian':
            if tangent_params is None:
                tangent_params = {'center': len(P)//2, 'width': len(P)/10, 'amplitude': 1.0}
            k_indices = jnp.arange(len(P))
            center = tangent_params.get('center', len(P)//2)
            width = tangent_params.get('width', len(P)/10)
            amplitude = tangent_params.get('amplitude', 1.0)
            tangent_P = amplitude * jnp.exp(-((k_indices - center) / width) ** 2)
        else:
            raise ValueError(f"Unknown tangent_type: {tangent_type}")
        
        # Normalize tangent vector if requested
        if isinstance(tangent_params, dict) and tangent_params.get('normalize', False):
            norm = jnp.linalg.norm(tangent_P)
            # Avoid division by zero
            tangent_P = jnp.where(norm > 0, tangent_P / norm, tangent_P)
        
        # Apply scaling
        tangent_P = scaling * tangent_P
        
        # Compute the JVP
        result, jvp_value = jvp(get_wrt_P, (P,), (tangent_P,))
        
        return result, jvp_value, tangent_P
    
    def diff(self, config):
        """
        Main differentiation interface that delegates to appropriate differentiation methods
        based on the provided configuration.
        
        Parameters:
        -----------
        config : DiffConfigDC
            Validated configuration object with all differentiation parameters
            
        Returns:
        --------
        The result of the differentiation operation as specified by the configuration
        """        

        if config.term is not None:
            jpt_func = functools.partial(self.get, term=config.term)
        else:
            jpt_func = getattr(self, config.function)
            
        jpt_params = {"P_window":config.P_window, "C_window":config.C_window}
        
        if config.diff_type == 'vector':
            return self._vector_diff(
                pk_method=config.pk_generation_method,
                P_params=config.pk_params,
                jpt_func=jpt_func,
                jpt_params=jpt_params,
                diff_param=config.pk_diff_param,
                diff_method=config.diff_method,
                tangent=config.tangent,
            )
        else:  # scalar differentiation
            return self._scalar_diff(
                pk_method=config.pk_generation_method,
                P_params=config.pk_params,
                jpt_func=jpt_func,
                jpt_params=jpt_params,
                diff_param=config.pk_diff_param,
                diff_method=config.diff_method,
                reduction_func=config.reduction_func,
            )
        
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
            'Omega_c': P_params['Omega_c'],
            'Omega_b': P_params['Omega_b'],
            'h': P_params['h'],
            'n_s': P_params['n_s'],
            'sigma8': P_params['sigma8'],
            'Omega_k': P_params['Omega_k'],
            'w0': P_params['w0'],
            'wa': P_params['wa'],
        }
        
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
    
    def _vector_diff(self, pk_method, P_params, jpt_func, jpt_params, diff_param, diff_method, tangent=None):
        
        current_value = P_params[diff_param]
        current_value = jnp.array(current_value, dtype=jnp.float64)

        def diff_func(param_value):
            power_spectrum = self._pk_generator(pk_method, param_value, diff_param, P_params)
            return jpt_func(P=power_spectrum, **jpt_params)

        if diff_method == 'jacfwd':
            result = jacfwd(diff_func)(current_value)
            return result
        
        elif diff_method == 'jvp':
            if tangent is None:
                # Default: use a scalar value of 1.0 for cosmological parameter differentiation
                tangent = jnp.array(1.0, dtype=jnp.float64)
            else:
                tangent = jnp.array(tangent, dtype=jnp.float64)
            
            primal_out, jvp_result = jvp(diff_func, (current_value,), (tangent,))
            # Should primals be returned as well? Or removed to keep the ouput consistent?
            return primal_out, jvp_result
        elif diff_method == 'vjp':
            primal_out, vjp_fn = vjp(diff_func, current_value)

            if isinstance(primal_out, tuple):
                # For tuple outputs, create a tuple of scalar ones
                tangent = tuple(jnp.ones_like(r) for r in primal_out)
            else:
                # For scalar output
                tangent = jnp.array(1.0)
            gradient = vjp_fn(tangent)[0]
            return primal_out, gradient

        
    def _scalar_diff(self, pk_method, P_params, jpt_func, jpt_params, diff_param, diff_method, reduction_func):
        if reduction_func is None:
            raise ValueError("reduction_func must be provided for scalar differentiation")
            
        current_value = P_params[diff_param]
        current_value = jnp.array(current_value, dtype=jnp.float64)
        
        def diff_func(param_value):
            power_spectrum = self._pk_generator(pk_method, param_value, diff_param, P_params)
            result = jpt_func(P=power_spectrum, **jpt_params)
            
            if isinstance(result, tuple):
                # If result is a tuple, apply reduction to each element
                return tuple(reduction_func(r) for r in result)
            else:
                # If result is a single array, apply reduction directly
                return reduction_func(result)
        
        if diff_method == 'vjp':
            primal_out, vjp_fn = vjp(diff_func, current_value)
            
            if isinstance(primal_out, tuple):
                # For tuple outputs, create a tuple of scalar ones
                tangent = tuple(jnp.array(1.0) for _ in primal_out)
            else:
                # For scalar output
                tangent = jnp.array(1.0)
            
            # This current approach uses 1 tangent resulting in one gradient, should they be split into one for each term?
            gradient = vjp_fn(tangent)[0]
            return primal_out, gradient
        
        elif diff_method == 'grad':
            # (For a jpt func) Would either need to combine the gradients or return them separately in a new tuple
            grad_fn = grad(diff_func)
            gradient = grad_fn(current_value)
            return gradient 
        
        else:
            raise ValueError(f"Unsupported scalar differentiation method: {diff_method}")
        

@dataclass(frozen=True)
class StaticConfig:
    k_size: int
    n_pad: int
    N: int
    low_extrap: Optional[float] = None
    high_extrap: Optional[float] = None
    EK: Optional[Any] = None

            
    def get(self, name):
        if name not in self.functions:
            raise KeyError(f"Function {name} not registered")
        return self.functions[name]


@partial(jit, static_argnames=["EK"])
def _apply_extrapolation(*args, EK=None):
    if EK is None:
        return args if len(args) > 1 else args[0]
    return [EK.PK_original(var)[1] for var in args] if len(args) > 1 else EK.PK_original(args[0])[1]

@partial(jit, static_argnames=["static_cfg", "operation"])
def compute_term(P, X, static_cfg, k_extrap, k_final, id_pad, l, m,
                 P_window=None, C_window=None, operation=None):        
    result, _ = J_k_tensor(P, X, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    result = _apply_extrapolation(result, EK=static_cfg.EK)

    if operation:
        final_result = operation(result)
        return final_result
    return result

@partial(jit, static_argnames=["static_cfg"])
def _get_1loop(P, X_spt, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    P22_coef = jnp.array([2*1219/1470., 2*671/1029., 2*32/1715., 2*1/3., 2*62/35., 2*8/35., 1/3.])
    Ps, mat = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P22_mat = jnp.multiply(P22_coef, jnp.transpose(mat))
    P22 = jnp.sum(P22_mat, axis=1)
    P13 = P_13_reg(k_extrap, Ps)
    P_1loop = P22 + P13
    P_1loop = _apply_extrapolation(P_1loop, EK=static_cfg.EK)
    return P_1loop

@partial(jit, static_argnames=["static_cfg"])
def _get_sig4(P, X_spt, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    Ps, _ = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    sig4 = jax.scipy.integrate.trapezoid(k_extrap ** 3 * Ps ** 2, x=jnp.log(k_extrap)) / (2. * jnp.pi ** 2)
    return sig4

@partial(jit, static_argnames=["static_cfg"])
def _get_Pd1d2(P, X_spt, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray, 
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray, 
               P_window=None, C_window=None):
    _, mat = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    Pd1d2 = 2. * (17. / 21 * mat[0, :] + mat[4, :] + 4. / 21 * mat[1, :])
    Pd1d2 = _apply_extrapolation(Pd1d2, EK=static_cfg.EK)
    return Pd1d2

@partial(jit, static_argnames=["static_cfg"])
def _get_Pd2d2(P, X_spt, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    _, mat = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    Pd2d2 = 2. * (mat[0, :])
    Pd2d2 = _apply_extrapolation(Pd2d2, EK=static_cfg.EK)
    return Pd2d2

@partial(jit, static_argnames=["static_cfg"])
def _get_Pd1s2(P, X_spt, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    _, mat = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    Pd1s2 = 2. * (8. / 315 * mat[0, :] + 4. / 15 * mat[4, :] + 254. / 441 * mat[1, :] + 2. / 5 * mat[5,:] + 16. / 245 * mat[2,:])
    Pd1s2 = _apply_extrapolation(Pd1s2, EK=static_cfg.EK)
    return Pd1s2

@partial(jit, static_argnames=["static_cfg"])
def _get_Pd2s2(P, X_spt, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    _, mat = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    Pd2s2 = 2. * (2. / 3 * mat[1, :])
    Pd2s2 = _apply_extrapolation(Pd2s2, EK=static_cfg.EK)
    return Pd2s2

@partial(jit, static_argnames=["static_cfg"])
def _get_Ps2s2(P, X_spt, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    _, mat = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    Pd2s2 = 2. * (4. / 45 * mat[0, :] + 8. / 63 * mat[1, :] + 8. / 35 * mat[2, :])
    Pd2s2 = _apply_extrapolation(Pd2s2, EK=static_cfg.EK)
    return Pd2s2

@partial(jit, static_argnames=["static_cfg"])
def _get_P_0EtE(P, X_list, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    X_IA_tij_feG2, X_IA_deltaE1 = X_list
    P_feG2, A = J_k_tensor(P, X_IA_tij_feG2, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_feG2 = _apply_extrapolation(P_feG2, EK=static_cfg.EK)
    P_A00E = compute_term(P, X_IA_deltaE1, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window, operation=lambda x: 2 * x)
    P_0EtE = jnp.subtract(P_feG2,(1/2)*P_A00E)
    P_0EtE = 2*P_0EtE
    return P_0EtE

@partial(jit, static_argnames=["static_cfg"])
def _get_P_E2tE(P, X_list, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    X_IA_tij_heG2, X_IA_A = X_list
    P_heG2, A = J_k_tensor(P, X_IA_tij_heG2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_heG2 = _apply_extrapolation(P_heG2, EK=static_cfg.EK)
    P_A0E2 = compute_term(P, X_IA_A, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window, operation=lambda x: 2 * x)
    P_E2tE = jnp.subtract(P_heG2,(1/2)*P_A0E2)
    P_E2tE = 2*P_E2tE
    return P_E2tE
    
@partial(jit, static_argnames=["static_cfg"])
def _get_P_tEtE(P, X_list, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    X_IA_tij_F2F2, X_IA_tij_G2G2, X_IA_tij_F2G2 = X_list
    P_F2F2, A = J_k_tensor(P,X_IA_tij_F2F2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_G2G2, A = J_k_tensor(P,X_IA_tij_G2G2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_F2G2, A = J_k_tensor(P,X_IA_tij_F2G2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_F2F2, P_G2G2, P_F2G2 = _apply_extrapolation(P_F2F2, P_G2G2, P_F2G2, EK=static_cfg.EK)
    P_tEtE = P_F2F2+P_G2G2-2*P_F2G2
    P_tEtE = 2*P_tEtE
    return P_tEtE

@partial(jit, static_argnames=["static_cfg"])
def _get_Pb1L_b2L(P, X_lpt, static_cfg: StaticConfig,
                  k_extrap: jnp.ndarray, k_final: jnp.ndarray,
                  id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
                  P_window=None, C_window=None):
    _, mat = J_k_scalar(P, X_lpt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    X3 = (50. / 21.) * j000 + 2. * j1n11 - (8. / 21.) * j002
    Pb1L_b2L = X3
    Pb1L_b2L = _apply_extrapolation(Pb1L_b2L, EK=static_cfg.EK)
    return Pb1L_b2L

@partial(jit, static_argnames=["static_cfg"])
def _get_Pb2L(P, X_lpt, static_cfg: StaticConfig,
                  k_extrap: jnp.ndarray, k_final: jnp.ndarray,
                  id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
                  P_window=None, C_window=None):
    _, mat = J_k_scalar(P, X_lpt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    X4 = (34. / 21.) * j000 + 2. * j1n11 + (8. / 21.) * j002
    Pb2L = X4
    Pb2L = _apply_extrapolation(Pb2L, EK=static_cfg.EK)
    return Pb2L

@partial(jit, static_argnames=["static_cfg"])
def _get_Pb2L_2(P, X_lpt, static_cfg: StaticConfig,
                  k_extrap: jnp.ndarray, k_final: jnp.ndarray,
                  id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
                  P_window=None, C_window=None):
    _, mat = J_k_scalar(P, X_lpt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    X5 = j000
    Pb2L_2 = X5
    Pb2L_2 = _apply_extrapolation(Pb2L_2, EK=static_cfg.EK)
    return Pb2L_2

@partial(jit, static_argnames=["static_cfg"])
def _get_P_d2tE(P, X_list, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    X_IA_gb2_F2, X_IA_gb2_G2 = X_list
    P_F2, _ = J_k_tensor(P, X_IA_gb2_F2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_G2, _ = J_k_tensor(P, X_IA_gb2_G2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_F2 = _apply_extrapolation(P_F2, EK=static_cfg.EK)
    P_G2 = _apply_extrapolation(P_G2, EK=static_cfg.EK)
    P_d2tE = 2 * (P_G2 - P_F2)
    return P_d2tE

@partial(jit, static_argnames=["static_cfg"])
def _get_P_s2tE(P, X_list, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    X_IA_gb2_S2F2, X_IA_gb2_S2G2 = X_list
    P_S2F2, _ = J_k_tensor(P, X_IA_gb2_S2F2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_S2G2, _ = J_k_tensor(P, X_IA_gb2_S2G2, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P_S2F2 = _apply_extrapolation(P_S2F2, EK=static_cfg.EK)
    P_S2G2 = _apply_extrapolation(P_S2G2, EK=static_cfg.EK)
    P_s2tE = 2 * (P_S2G2 - P_S2F2)
    return P_s2tE



#Get functions that use jax_utils functions, produce non exact outputs though 
#differences are due to jpt versions of input parameters (parameters pass allclose, output does not)

#Also P13 but that was absorbed in P_1loop
@partial(jit, static_argnames=["static_cfg"])
def _get_Pb1L(P, X_lpt, static_cfg: StaticConfig,
                  k_extrap: jnp.ndarray, k_final: jnp.ndarray,
                  id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
                  P_window=None, C_window=None):
    Ps, mat = J_k_scalar(P, X_lpt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    X1 = ((144. / 245.) * j000 - (176. / 343.) * j002 - (128. / 1715.) * j004 + (16. / 35.) * j1n11 - (
            16. / 35.) * j1n13)
    Y1 = Y1_reg_NL(k_extrap, Ps)
    Pb1L = X1 + Y1
    Pb1L = _apply_extrapolation(Pb1L, EK=static_cfg.EK)
    return Pb1L

@partial(jit, static_argnames=["static_cfg"])
def _get_Pb1L_2(P, X_lpt, static_cfg: StaticConfig,
                  k_extrap: jnp.ndarray, k_final: jnp.ndarray,
                  id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
                  P_window=None, C_window=None):
    Ps, mat = J_k_scalar(P, X_lpt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    X2 = ((16. / 21.) * j000 - (16. / 21.) * j002 + (16. / 35.) * j1n11 - (16. / 35.) * j1n13)
    Y2 = Y2_reg_NL(k_extrap, Ps)
    Pb1L_2 = X2 + Y2
    Pb1L_2 = _apply_extrapolation(Pb1L_2, EK=static_cfg.EK)
    return Pb1L_2

@jit
def _get_P_Btype2(P, k_original):
    P_Btype2 = P_IA_B(k_original, P)
    P_Btype2 = 4 * P_Btype2
    return P_Btype2

@jit
def _get_P_deltaE2(P, k_original):
    P_deltaE2 = P_IA_deltaE2(k_original, P)
    #Add extrap?
    P_deltaE2 = 2 * P_deltaE2
    return P_deltaE2

@partial(jit, static_argnames=["static_cfg"])
def _get_P_0tE(P, X_list, static_cfg: StaticConfig,
                  k_original: jnp.array, k_extrap: jnp.ndarray, k_final: jnp.ndarray,
                  id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
                  P_window=None, C_window=None):
    X_spt, X_sptG = X_list
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
    return P_0tE

@partial(jit, static_argnames=["static_cfg"])
def _get_OV(P, X_OV, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray,
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
               P_window=None, C_window=None):
    P, _ = J_k_tensor(P, X_OV, static_cfg, k_extrap, k_final, id_pad, l, m,
                           P_window=P_window, C_window=C_window)
    P = _apply_extrapolation(P, EK=static_cfg.EK)
    P_OV = P * (2 * jnp.pi) ** 2
    return P_OV

@partial(jit, static_argnames=["static_cfg"])
def _get_sig3nl(P, X_spt, static_cfg: StaticConfig,
                  k_extrap: jnp.ndarray, k_final: jnp.ndarray,
                  id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  
                  P_window=None, C_window=None):
    Ps, _ = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    sig3nl = Y1_reg_NL(k_extrap, Ps)
    sig3nl = _apply_extrapolation(sig3nl, EK=static_cfg.EK)
    return sig3nl

@partial(jit, static_argnames=["static_cfg"])
def J_k_scalar(P, X, static_cfg: StaticConfig,
               k_extrap: jnp.ndarray, k_final: jnp.ndarray, # Dynamic arrays from ComputeConfig
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  # More dynamic arrays
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
               k_extrap: jnp.ndarray, k_final: jnp.ndarray, # Dynamic arrays from ComputeConfig
               id_pad: jnp.ndarray, l: jnp.ndarray, m: jnp.ndarray,  # More dynamic arrays
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
            P_b1 = P_b1 * P_window
            P_b2 = P_b2 * P_window
            
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
    # jpt = create_jaxpt() # << For memory profiler

    k = jnp.logspace(-3, 1, 1000)
    jpt = JAXPT(k, low_extrap=-5, high_extrap=5, n_pad=int(0.5*len(k)))
    from jaxpt.diff_config import DiffConfig
    import matplotlib.pyplot as plt
    import jax.numpy as jnp

    config = DiffConfig()
    config.function = 'gI_ta'
    # config.term = 'P_A'
    config.diff_method = 'vjp'
    config.diff_type = 'vector'
    config.pk_diff_param = 'Omega_c'
    config.pk_generation_method = 'jax-cosmo'
    diff_config = config.build_and_validate()
    dPk = jpt.diff(diff_config)
    direct = jpt.gI_ta(P=jpt._jax_cosmo_pk_generator(0.12, 'Omega_c', diff_config.pk_params))
    
    print(dPk)