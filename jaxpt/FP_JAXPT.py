import jax.numpy as jnp
from jax import jit, vjp, jvp, jacfwd, jacrev
import numpy as np
from jax import config
import jax
from fastpt import FASTPT as FPT
from jaxpt.jax_utils import p_window, c_window, jax_k_extend
from jaxpt.jax_utils import P_13_reg, Y1_reg_NL, Y2_reg_NL, P_IA_B, P_IA_deltaE2, P_IA_13F, P_IA_13G
config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
import os
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

        if warmup: self._simple_warmup()

    def _simple_warmup(self):
        """Streamlined JIT warm-up focused on top-level API functions and internal computation methods."""
        print("Starting JIT warm-up...")
        t0 = time()
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
            "one_loop_dd",
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

        # Pk generation warm-up
        _ = jit_jax_cosmo_pk_generator(0.012, 'Omega_c', {}, self.k_original) 
        # Currently only empty and full dicts are compiled, any params dict that is not full will trigger recompilation
        representative_pk_params_all = {
            'Omega_c': 0.12, 
            'Omega_b': 0.022,  
            'h': 0.69,
            'n_s': 0.96,
            'sigma8': 0.8,
            'Omega_k': 0.0, 
            'w0': -1.0,
            'wa': 0.0,
        }
        _ = jit_jax_cosmo_pk_generator(0.12, 'Omega_c', representative_pk_params_all, self.k_original)
        t1 = time()
        print(f"JIT warm-up completed in {t1-t0:.2f} seconds.")


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
        
    def one_loop_dd(self, P, P_window=None, C_window=None):
        return _one_loop_core(self.X_spt, self._static_config, self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                          P, P_window=P_window, C_window=C_window)
    
    #NOTE: this function has a signature change from FASTPT, it now no longer returns P1loop and Ps
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

    def kPol(self, P, P_window=None, C_window=None):
        return _kPol_core(self.X_kP1, self.X_kP2, self.X_kP3, self._static_config,
                          self.k_extrap, self.k_final, self.id_pad, self.l, self.m,
                          P, P_window=P_window, C_window=C_window)
    
    

    def diff(self, pk_method, pk_params, pk_diff_param, function=None, P_window=None, C_window=None,
              diff_method='jacfwd', tangent=None):    
        
        if function is None:
            raise ValueError("No function provided for differentiation. Please specify a function name.")
        if not isinstance(pk_params, dict):
            raise ValueError("pk_params must be a dictionary of cosmological parameters.")
        if pk_diff_param not in pk_params:
            raise ValueError(f"{pk_diff_param} not found in pk_params")
        
        func = getattr(self, function)
        def diff_func(param_value):
            P = self._pk_generator(pk_method, param_value, pk_diff_param, pk_params)
            P_new = func(P, P_window=P_window, C_window=C_window)
            return P_new

        param_value = pk_params.get(pk_diff_param)
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
            if tangent is None:
                if isinstance(primal_out, tuple):
                    tangent = tuple(jnp.ones_like(r) for r in primal_out)
                else: #OV case
                    tangent = jnp.ones_like(primal_out, dtype=jnp.float64)
            vjp_result = vjp_fn(tangent)[0]
            return vjp_result, primal_out
        else:
            raise ValueError(f"Unsupported differentiation method: {diff_method}. Use 'jacfwd', 'jvp', or 'vjp'.")
        
    def multi_param_diff(self, pk_method, pk_params, pk_diff_params,
                    function=None, P_window=None, C_window=None, 
                    diff_method='jacfwd', output_indices=None):
        if function is None:
            raise ValueError("No function provided for differentiation. Please specify a function name.")
        if not isinstance(pk_diff_params, list):
            raise ValueError("pk_diff_params must be a list of parameter names.")
        if not isinstance(pk_params, dict):
            raise ValueError("pk_params must be a dictionary of cosmological parameters.")
        for param in pk_diff_params:
            if param not in pk_params:
                raise ValueError(f"{param} not found in pk_params")
        
        func = getattr(self, function)
        param_values = jnp.array([pk_params.get(param) for param in pk_diff_params])

        if any(value is None for value in param_values):
            raise ValueError(f"Missing parameter values for: {', '.join([param for param, value in zip(pk_diff_params, param_values) if value is None])}. "
                             "Please provide all required parameters in pk_params.")
        
        def vector_diff_func(param_vector):
            updated_params = pk_params.copy()
            for i, name in enumerate(pk_diff_params):
                updated_params[name] = param_vector[i]
            
            # [0] is used to preserve the pk_generator signature, it is not actually needed in this case as the params are set explicitly above
            P = self._pk_generator(pk_method, param_vector[0], pk_diff_params[0], updated_params)
            
            P_new = func(P, P_window=P_window, C_window=C_window)
        
            # Handle output selection
            if output_indices is not None:
                if isinstance(P_new, tuple):
                    if isinstance(output_indices, int):
                        return P_new[output_indices]
                    else:
                        return tuple(P_new[i] for i in output_indices)
                else:
                    # Single output, ignore output_indices
                    return P_new
            
            return P_new
        
        if diff_method == 'jacfwd':
            jacobian = jacfwd(vector_diff_func)(param_values)
        elif diff_method == 'jacrev':
            jacobian = jacrev(vector_diff_func)(param_values)
        else:
            raise ValueError(f"Unsupported differentiation method: {diff_method}. Use 'jacfwd' or 'jacrev'.")
        
        if isinstance(jacobian, (list, tuple)):
            # Multiple outputs - return nested dictionary
            result = {}
            output_names = self._get_output_names(function)
            if output_indices is not None: output_names = [output_names[i] for i in output_indices]
            
            for i, jac in enumerate(jacobian):
                if i < len(output_names):
                    name = output_names[i]
                else:
                    name = f'output_{i}'
                
                jac = jnp.asarray(jac)
                
                if jac.ndim == 1:
                    # Scalar output - jacobian is 1D array of derivatives
                    result[name] = {
                        param: jac[j]
                        for j, param in enumerate(pk_diff_params)
                    }
                else:
                    # Vector output - jacobian is 2D array
                    result[name] = {
                        param: jac[:, j]
                        for j, param in enumerate(pk_diff_params)
                    }
            return result
        else:
            # Single output - ensure it's an array
            jacobian = jnp.asarray(jacobian)
            
            if jacobian.ndim == 1:
                # Single scalar output
                return {
                    param: jacobian[i]
                    for i, param in enumerate(pk_diff_params)
                }
            else:
                # Single vector output
                return {
                    param: jacobian[:, i]
                    for i, param in enumerate(pk_diff_params)
                }

    def _get_output_names(self, function_name):
        output_map = {
            'one_loop_dd': ['P_1loop', 'Ps'],
            'one_loop_dd_bias_b3nl': ["Pd1d2", "Pd2d2", "Pd1s2", "Pd2s2", "Ps2s2", "sig4", "sig3nl"],
            'one_loop_dd_bias_lpt_NL': ["Ps", "Pb1L", "Pb1L_2", "Pb1L_b2L", "Pb2L", "Pb2L_2", "sig4"],
            'IA_ct': ["P_0tE", "P_0EtE", "P_E2tE", "P_tEtE"],
            'IA_mix': ['P_A', 'P_Btype2', 'P_DEE', 'P_DBB'],
            'IA_ta': ['P_deltaE1', 'P_deltaE2', 'P_0E0E', 'P_0B0B'],
            'IA_tt': ['P_E', 'P_B'],
            'gI_ct': ["P_d2tE", "P_s2tE"],
            'gI_ta': ["P_d2E", "P_d20E", "P_s2E", "P_s20E"],
            'gI_tt': ["P_s2E2", "P_d2E2"],
            'OV': ['P_OV'],
            'kPol': ['P_kP1', 'P_kP2', 'P_kP3'],
        }
        return output_map[function_name]

    def _pk_generator(self, pk_method, param_value, diff_param, P_params):
        if pk_method == 'jax-cosmo':
            return jit_jax_cosmo_pk_generator(param_value, diff_param, P_params, self.k_original)
        elif pk_method == 'discoeb':
            return self._discoeb_pk_generator(param_value, diff_param, P_params)
        else:
            raise ValueError(f"Unsupported power spectrum generation method: {pk_method}")
    
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
        Pkm = jnp.interp(self.k_original, kmodes, Pkm)
        return Pkm

@partial(jit, static_argnames=["diff_param"])
def jit_jax_cosmo_pk_generator(param_value, diff_param, P_params, k_original):
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

    # Explicitly use the differentiation parameter, done before this method on multi param diff
    cosmo_dict[diff_param] = param_value
    new_cosmo = Cosmology(**cosmo_dict)
    return power.linear_matter_power(new_cosmo, k_original)
        

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
def _one_loop_core(X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                   P, P_window=None, C_window=None):
    Ps, mat = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P22_coef = jnp.array([2*1219/1470., 2*671/1029., 2*32/1715., 2*1/3., 2*62/35., 2*8/35., 1/3.])
    P22_mat = jnp.multiply(P22_coef, jnp.transpose(mat))
    P22 = jnp.sum(P22_mat, axis=1)
    P13 = P_13_reg(k_extrap, Ps)
    P_1loop = P22 + P13
    P_1loop, Ps, = _apply_extrapolation(P_1loop, Ps, EK=static_cfg.EK)
    return P_1loop, Ps

@partial(jit, static_argnames=["static_cfg"])
def _b3nl_core(X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
               P, P_window=None, C_window=None):
    Ps, mat = J_k_scalar(P, X_spt, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    Pd1d2 = 2. * (17. / 21 * mat[0, :] + mat[4, :] + 4. / 21 * mat[1, :])
    Pd2d2 = 2. * (mat[0, :])
    Pd1s2 = 2. * (8. / 315 * mat[0, :] + 4. / 15 * mat[4, :] + 254. / 441 * mat[1, :] + 2. / 5 * mat[5,:] + 16. / 245 * mat[2,:])
    Pd2s2 = 2. * (2. / 3 * mat[1, :])
    Ps2s2 = 2. * (4. / 45 * mat[0, :] + 8. / 63 * mat[1, :] + 8. / 35 * mat[2, :])
    sig4 = jax.scipy.integrate.trapezoid(k_extrap ** 3 * Ps ** 2, x=jnp.log(k_extrap)) / (2. * jnp.pi ** 2)
    sig3nl = Y1_reg_NL(k_extrap, Ps)
 
    Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig3nl = _apply_extrapolation(
        Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig3nl, EK=static_cfg.EK)
    
    return Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4, sig3nl

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
def _IA_ta_core(X_IA_deltaE1, X_IA_0E0E, X_IA_0B0B, static_cfg, k_original, k_extrap, k_final, id_pad, l, m,
                P, P_window=None, C_window=None):
    P_deltaE1, _ = J_k_tensor(P, X_IA_deltaE1, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_deltaE2 = P_IA_deltaE2(k_original, P)
    P_0E0E, _ = J_k_tensor(P, X_IA_0E0E, static_cfg, k_extrap, k_final, id_pad, l, m,
                         P_window=P_window, C_window=C_window)
    P_0B0B, _ = J_k_tensor(P, X_IA_0B0B, static_cfg, k_extrap, k_final, id_pad, l, m,
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

@partial(jit, static_argnames=["static_cfg"])
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

@partial(jit, static_argnames=["static_cfg"])
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

@partial(jit, static_argnames=["static_cfg"])
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

@partial(jit, static_argnames=["static_cfg"])
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

@partial(jit, static_argnames=["N"])
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

    def diff_p(param_value):
        """Differentiation function for testing."""
        pk_params = {
            'Omega_c': 0.12,  # Cold dark matter density parameter
            'Omega_b': 0.022,  # Baryon density parameter
            'h': 0.69,         # Dimensionless Hubble constant
            'sigma8': 0.8,    # Amplitude of matter fluctuations
            'n_s': 0.96,      # Scalar spectral index
            'Omega_k': 0.0,   # Curvature density parameter
            'w0': -1.0,       # Dark energy equation of state parameter
            'wa': 0.0,        # Dark energy equation of state parameter
        }
        pk_params['Omega_c'] = param_value
        return jit_jax_cosmo_pk_generator(param_value, 'Omega_c', pk_params, jpt.k_original)
    
    pk_params = {
        'Omega_c': 0.12,  # Cold dark matter density parameter
        'Omega_b': 0.022,  # Baryon density parameter
        'h': 0.69,         # Dimensionless Hubble constant
        'sigma8': 0.8,    # Amplitude of matter fluctuations
        'n_s': 0.96,      # Scalar spectral index
        'Omega_k': 0.0,   # Curvature density parameter
        'w0': -1.0,       # Dark energy equation of state parameter
        'wa': 0.0,        # Dark energy equation of state parameter
    }
    t0 = time()
    # tangent = jnp.ones_like(0.012, dtype=jnp.float64)
    # primal_out, jvp_result = jvp(diff_p, (0.012,), (tangent,))
    jit_jax_cosmo_pk_generator(0.12, 'Omega_c', pk_params, jpt.k_original)
    t1 = time()
    print(f"Pk differentiation time: {t1 - t0:.8f} seconds")


    t0 = time()
    jvp_result = jpt.diff(pk_method='jax-cosmo', pk_params=pk_params, pk_diff_param='Omega_c', function='IA_tt', diff_method='jvp')
    t1 = time()
    print(f"JVP total computation time: {t1 - t0:.4f} seconds")