from fastpt.jax_utils import p_window, c_window, jax_k_extend
import jax.numpy as jnp
from jax import jit
from time import time
import numpy as np
from jax import vjp
from jax import config
import jax
from fastpt import FASTPT as FPT
config.update("jax_enable_x64", True)
import functools
from fastpt.jax_utils import P_13_reg, Y1_reg_NL, Y2_reg_NL, P_IA_B, P_IA_deltaE2, P_IA_13F, P_IA_13G
from jax.numpy.fft import ifft, irfft

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
    def __init__(self, k, P_window=None, C_window=None, low_extrap=None, high_extrap=None, n_pad=None):
        
        if (k is None or len(k) == 0):
            raise ValueError('You must provide an input k array.')        
        if not isinstance(k, jnp.ndarray):
            try:
                k = jnp.asarray(k, dtype=jnp.float64)
            except:
                raise ValueError('Input k array must be a jax numpy array, automatic conversion failed.')
        if not isinstance(P_window, jnp.ndarray) and P_window is not None:
            try:
                P_window = jnp.asarray(P_window, dtype=jnp.float64)
            except:
                raise ValueError('Input P_window array must be a jax numpy array, automatic conversion failed.')
            
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

        if P_window is not None:
            self.p_win = jnp.array(p_window(self.k_extrap, P_window[0], P_window[1]))
        else:
            self.p_win = None
        self.C_window = C_window

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
            "P_Btype2": {"type": "special", "method": "_get_P_Btype2"},
            "P_deltaE2": {"type": "special", "method": "_get_P_deltaE2"},
            "P_der": {"type": "special", "method": "IA_der"},
            "P_OV": {"type": "special", "method": "OV"},
            
            "P_0tE": {"type": "special", "method": "_get_P_0tE"},
            "P_0EtE": {"type": "special", "method": "_get_P_0EtE"}, 
            "P_E2tE": {"type": "special", "method": "_get_P_E2tE"},
            "P_tEtE": {"type": "special", "method": "_get_P_tEtE"},
            
            "P_1loop": {"type": "special", "method": "_get_1loop"},
            
            "Pd1d2": {"type": "special", "method": "_get_Pd1d2"},
            "Pd2d2": {"type": "special", "method": "_get_Pd2d2"},
            "Pd1s2": {"type": "special", "method": "_get_Pd1s2"},
            "Pd2s2": {"type": "special", "method": "_get_Pd2s2"},
            "Ps2s2": {"type": "special", "method": "_get_Ps2s2"},
            
            "sig4": {"type": "special", "method": "_get_sig4"},
            "sig3nl": {"type": "special", "method": "_get_sig3nl"},
            
            "Pb1L": {"type": "special", "method": "_get_Pb1L"},
            "Pb1L_2": {"type": "special", "method": "_get_Pb1L_2"},
            "Pb1L_b2L": {"type": "special", "method": "_get_Pb1L_b2L"},
            "Pb2L": {"type": "special", "method": "_get_Pb2L"},
            "Pb2L_2": {"type": "special", "method": "_get_Pb2L_2"},
            
            "P_d2tE": {"type": "special", "method": "_get_P_d2tE"},
            "P_s2tE": {"type": "special", "method": "_get_P_s2tE"}
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

        #JIT Compile functions
        try:
            self.J_k_scalar = jit(self.J_k_scalar, static_argnames=["n_pad", "k_size", "EK",
                                                                     "N", "l", "id_pad", "k_extrap", "k_final", "low_extrap", "high_extrap"])
        except:
            print("J_k_scalar JIT compilation failed. Using default python implementation.")
        try:
            self._J_k_tensor_core = jit(self.J_k_tensor, static_argnames=["n_pad", "k_size", "EK",
                                                                          "N", "l", "id_pad", "k_extrap", "k_final", "low_extrap", "high_extrap"])
        except:
            print("J_k_tensor JIT compilation failed. Using default python implementation.")
        try:
            self.fourier_coefficients = jit(self.fourier_coefficients, static_argnames=["N"])
        except:
            print("fourier_coefficients JIT compilation failed. Using default python implementation.")
        try:
            self.convolution = jit(self.convolution)
        except:
            print("convolution JIT compilation failed. Using default python implementation.")
        try:
            self.compute_term = jit(self.compute_term, static_argnames=["operation"])
        except:
            print("Compute term JIT compilation failed. Using default python implementation.")
        try:
            self._apply_extrapolation = jit(self._apply_extrapolation)
        except:
            print("Apply extrapolation JIT compilation failed. Using default python implementation.")
        try:
            self.get = jit(self.get, static_argnames=["term"])
        except:
            print("get JIT compilation failed. Using default python implementation.")
        
        #These cannot be cached properties since they would be accessed twice in one function call (the one loop functions)
        #Therefore producing a side affect as the second access is done via cache and breaking differentiability
        self.X_spt = process_x_term(self.temp_fpt.X_spt)
        self.X_lpt = process_x_term(self.temp_fpt.X_lpt)


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
        

    def get(self, term, P, C_window=None, diff_mode=None):
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
        C_window = C_window if C_window is None else C_window
        if term in self.term_groups:
            return tuple(self.get(t, P, C_window) for t in self.term_groups[term])
        
        if term not in self.term_config:
            raise ValueError(f"Unknown term: {term}")
            
        config = self.term_config[term]
        
        if config["type"] == "standard":
            X = getattr(config["X"])
            operation = config.get("operation")
            return compute_term(X, operation=operation, P=P, C_window=C_window)
        
        elif config["type"] == "special":
            method_name = config["method"]
            
            if hasattr(method_name):
                method = getattr(method_name)
                result = method(P, C_window=C_window)
                return result
                
        raise ValueError(f"Unable to process term: {term}")
    
    def one_loop_dd_bias_b3nl_core(self, P, C_window=None):
        return tuple(self.get(t, P, C_window) for t in self.term_groups["one_loop_dd_bias_b3nl"])

    def one_loop_dd_bias_lpt_NL_core(self, P, C_window=None):
        return tuple(self.get(t, P, C_window) for t in self.term_groups["one_loop_dd_bias_lpt_NL"])

    def IA_tt_core(self, P, C_window=None):
        return tuple(self.get(t, P, C_window) for t in self.term_groups["IA_tt"])

    def IA_mix_core(self, P, C_window=None):
        return tuple(self.get(t, P, C_window) for t in self.term_groups["IA_mix"])

    def IA_ta_core(self, P, C_window=None):
        return tuple(self.get(t, P, C_window) for t in self.term_groups["IA_ta"])

    def IA_ct_core(self, P, C_window=None):
        return tuple(self.get(t, P, C_window) for t in self.term_groups["IA_ct"])

    def gI_ct_core(self, P, C_window=None):
        return tuple(self.get(t, P, C_window) for t in self.term_groups["gI_ct"])

    def IA_gb2_core(self, P, C_window=None):
        return tuple(self.get(t, P, C_window) for t in self.term_groups["IA_gb2"])

    def gI_ta_core(self, P, C_window=None):
        return tuple(self.get(t, P, C_window) for t in self.term_groups["gI_ta"])

    def gI_tt_core(self, P, C_window=None):
        return tuple(self.get(t, P, C_window) for t in self.term_groups["gI_tt"])

    def OV_core(self, P, C_window=None):
        P, A = J_k_tensor(P, self.X_OV, self.k_extrap, self.k_final, self.k_size,
                                    self.n_pad, self.id_pad, self.l, self.m, self.N, P_window=self.p_win, C_window=C_window,
                                    low_extrap=self.low_extrap, high_extrap=self.high_extrap, EK=self.EK)
        P = _apply_extrapolation(P)
        P_OV = P * (2 * jnp.pi) ** 2
        return P_OV

    def IA_der_core(self, P, C_window=None):
        return (self.k_original**2)*P

    def kPol_core(self, P, C_window=None):
        return tuple(self.get(t, P, C_window) for t in self.term_groups["kPol"])




def _apply_extrapolation(extrap, EK, *args):
    if not extrap:
        return args if len(args) > 1 else args[0]
    return [EK.PK_original(var)[1] for var in args] if len(args) > 1 else EK.PK_original(args[0])[1]

def compute_term(P, X, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, p_win,
                 operation=None, C_window=None, low_extrap=None, high_extrap=None, EK=None):        
    result, _ = J_k_tensor(P, X, k_extrap, k_final, k_size,
                                n_pad, id_pad, l, m, N, P_window=p_win, C_window=C_window,
                                low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    result = _apply_extrapolation(result)

    if operation:
        final_result = operation(result)
        return final_result
    return result

def _get_1loop(P, C_window=None):
    P22 = _get_P22(P, C_window=C_window)
    P13 = _get_P13(P, C_window=C_window)
    P_1loop = P22 + P13
    P_1loop = _apply_extrapolation(P_1loop)
    return P_1loop

def _get_P22(P, X_spt, m, N, n_pad, id_pad, k_extrap, k_final, k_size, l, 
             C_window=None, low_extrap=None, high_extrap=None, EK=None):
    P22_coef = jnp.array([2*1219/1470., 2*671/1029., 2*32/1715., 2*1/3., 2*62/35., 2*8/35., 1/3.])
    _, mat = J_k_scalar(P, X_spt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    P22_mat = jnp.multiply(P22_coef, jnp.transpose(mat))
    P22 = jnp.sum(P22_mat, axis=1)
    return P22

def _get_sig4(P, X_spt, m, N, n_pad, id_pad, k_extrap, k_final, k_size, l, 
              C_window=None, low_extrap=None, high_extrap=None, EK=None):
    Ps, _ = J_k_scalar(P, X_spt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    sig4 = jax.scipy.integrate.trapezoid(k_extrap ** 3 * Ps ** 2, x=jnp.log(k_extrap)) / (2. * jnp.pi ** 2)
    return sig4

def _get_Pd1d2(P, X_spt, m, N, n_pad, id_pad, k_extrap, k_final, k_size, l, 
               C_window=None, low_extrap=None, high_extrap=None, EK=None):
    _, mat = J_k_scalar(P, X_spt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    Pd1d2 = 2. * (17. / 21 * mat[0, :] + mat[4, :] + 4. / 21 * mat[1, :])
    Pd1d2 = _apply_extrapolation(Pd1d2)
    return Pd1d2

def _get_Pd2d2(P, X_spt, m, N, n_pad, id_pad, k_extrap, k_final, k_size, l, 
               C_window=None, low_extrap=None, high_extrap=None, EK=None):
    _, mat = J_k_scalar(P, X_spt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    Pd2d2 = 2. * (mat[0, :])
    Pd2d2 = _apply_extrapolation(Pd2d2)
    return Pd2d2

def _get_Pd1s2(P, X_spt, m, N, n_pad, id_pad, k_extrap, k_final, k_size, l, 
               C_window=None, low_extrap=None, high_extrap=None, EK=None):
    _, mat = J_k_scalar(P, X_spt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    Pd1s2 = 2. * (8. / 315 * mat[0, :] + 4. / 15 * mat[4, :] + 254. / 441 * mat[1, :] + 2. / 5 * mat[5,:] + 16. / 245 * mat[2,:])
    Pd1s2 = _apply_extrapolation(Pd1s2)
    return Pd1s2

def _get_Pd2s2(P, X_spt, m, N, n_pad, id_pad, k_extrap, k_final, k_size, l, 
               C_window=None, low_extrap=None, high_extrap=None, EK=None):
    _, mat = J_k_scalar(P, X_spt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    Pd2s2 = 2. * (2. / 3 * mat[1, :])
    Pd2s2 = _apply_extrapolation(Pd2s2)
    return Pd2s2

def _get_Ps2s2(P, X_spt, m, N, n_pad, id_pad, k_extrap, k_final, k_size, l, 
               C_window=None, low_extrap=None, high_extrap=None, EK=None):
    _, mat = J_k_scalar(P, X_spt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    Pd2s2 = 2. * (4. / 45 * mat[0, :] + 8. / 63 * mat[1, :] + 8. / 35 * mat[2, :])
    Pd2s2 = _apply_extrapolation(Pd2s2)
    return Pd2s2

def _get_P_0EtE(P, X_IA_tij_feG2, X_IA_deltaE1, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, p_win, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    P_feG2, A = J_k_tensor(P, X_IA_tij_feG2, k_extrap, k_final, k_size,
                                n_pad, id_pad, l, m, N, P_window=p_win, C_window=C_window,
                                low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    P_feG2 = _apply_extrapolation(P_feG2)
    P_A00E = compute_term(X_IA_deltaE1, operation=lambda x: 2 * x, 
                                    P=P, C_window=C_window)
    P_0EtE = jnp.subtract(P_feG2,(1/2)*P_A00E)
    P_0EtE = 2*P_0EtE
    return P_0EtE

def _get_P_E2tE(P, X_IA_tij_heG2, X_IA_A, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, p_win, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    P_heG2, A = J_k_tensor(P,X_IA_tij_heG2, k_extrap, k_final, k_size,
                                n_pad, id_pad, l, m, N, P_window=p_win, C_window=C_window,
                                low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    P_heG2 = _apply_extrapolation(P_heG2)
    P_A0E2 = compute_term(X_IA_A, operation=lambda x: 2 * x, 
                                P=P, C_window=C_window)
    P_E2tE = jnp.subtract(P_heG2,(1/2)*P_A0E2)
    P_E2tE = 2*P_E2tE
    return P_E2tE
    
def _get_P_tEtE(P, X_IA_tij_F2F2, X_IA_tij_G2G2, X_IA_tij_F2G2, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, p_win, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    P_F2F2, A = J_k_tensor(P,X_IA_tij_F2F2, k_extrap, k_final, k_size,
                                n_pad, id_pad, l, m, N, P_window=p_win, C_window=C_window,
                                low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    P_G2G2, A = J_k_tensor(P,X_IA_tij_G2G2, k_extrap, k_final, k_size,
                                n_pad, id_pad, l, m, N, P_window=p_win, C_window=C_window,
                                low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    P_F2G2, A = J_k_tensor(P,X_IA_tij_F2G2, k_extrap, k_final, k_size,
                                n_pad, id_pad, l, m, N, P_window=p_win, C_window=C_window,
                                low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    P_F2F2, P_G2G2, P_F2G2 = _apply_extrapolation(P_F2F2, P_G2G2, P_F2G2)
    P_tEtE = P_F2F2+P_G2G2-2*P_F2G2
    P_tEtE = 2*P_tEtE
    return P_tEtE

def _get_Pb1L_b2L(P, X_lpt, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    _, mat = J_k_scalar(P, X_lpt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    X3 = (50. / 21.) * j000 + 2. * j1n11 - (8. / 21.) * j002
    Pb1L_b2L = X3
    Pb1L_b2L = _apply_extrapolation(Pb1L_b2L)
    return Pb1L_b2L

def _get_Pb2L(P, X_lpt, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    _, mat = J_k_scalar(P, X_lpt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    X4 = (34. / 21.) * j000 + 2. * j1n11 + (8. / 21.) * j002
    Pb2L = X4
    Pb2L = _apply_extrapolation(Pb2L)
    return Pb2L

def _get_Pb2L_2(P, X_lpt, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    _, mat = J_k_scalar(P, X_lpt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    X5 = j000
    Pb2L_2 = X5
    Pb2L_2 = _apply_extrapolation(Pb2L_2)
    return Pb2L_2

def _get_P_d2tE(P, X_IA_gb2_F2, X_IA_gb2_G2, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, p_win, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    P_F2, _ = J_k_tensor(P, X_IA_gb2_F2, k_extrap, k_final, k_size,
                                n_pad, id_pad, l, m, N, P_window=p_win, C_window=C_window,
                                low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    P_G2, _ = J_k_tensor(P, X_IA_gb2_G2, k_extrap, k_final, k_size,
                                n_pad, id_pad, l, m, N, P_window=p_win, C_window=C_window,
                                low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    P_F2 = _apply_extrapolation(P_F2)
    P_G2 = _apply_extrapolation(P_G2)
    P_d2tE = 2 * (P_G2 - P_F2)
    return P_d2tE

def _get_P_s2tE(P, X_IA_gb2_S2F2, X_IA_gb2_S2G2, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, p_win, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    P_S2F2, _ = J_k_tensor(P, X_IA_gb2_S2F2, k_extrap, k_final, k_size,
                                n_pad, id_pad, l, m, N, P_window=p_win, C_window=C_window,
                                low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    P_S2G2, _ = J_k_tensor(P, X_IA_gb2_S2G2, k_extrap, k_final, k_size,
                                n_pad, id_pad, l, m, N, P_window=p_win, C_window=C_window,
                                low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    P_S2F2 = _apply_extrapolation(P_S2F2)
    P_S2G2 = _apply_extrapolation(P_S2G2)
    P_s2tE = 2 * (P_S2G2 - P_S2F2)
    return P_s2tE



#Get functions that use jax_utils functions, produce non exact outputs though 
#differences are due to jpt versions of input parameters (parameters pass allclose, output does not)
def _get_P13(P, X_spt, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    Ps, _ = J_k_scalar(P, X_spt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    P13 = P_13_reg(k_extrap, Ps)
    return P13

def _get_Pb1L(P, X_lpt, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    Ps, mat = J_k_scalar(P, X_lpt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    X1 = ((144. / 245.) * j000 - (176. / 343.) * j002 - (128. / 1715.) * j004 + (16. / 35.) * j1n11 - (
            16. / 35.) * j1n13)
    Y1 = Y1_reg_NL(k_extrap, Ps)
    Pb1L = X1 + Y1
    Pb1L = _apply_extrapolation(Pb1L)
    return Pb1L

def _get_Pb1L_2(P, X_lpt, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    Ps, mat = J_k_scalar(P, X_lpt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    X2 = ((16. / 21.) * j000 - (16. / 21.) * j002 + (16. / 35.) * j1n11 - (16. / 35.) * j1n13)
    Y2 = Y2_reg_NL(k_extrap, Ps)
    Pb1L_2 = X2 + Y2
    Pb1L_2 = _apply_extrapolation(Pb1L_2)
    return Pb1L_2

def _get_P_Btype2(P, k_original, C_window=None):
    P_Btype2 = P_IA_B(k_original, P)
    P_Btype2 = 4 * P_Btype2
    return P_Btype2

def _get_P_deltaE2(P, k_original, C_window=None):
    P_deltaE2 = P_IA_deltaE2(k_original, P)
    #Add extrap?
    P_deltaE2 = 2 * P_deltaE2
    return P_deltaE2

def _get_P_0tE(P, X_spt, X_sptG, k_original, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    Ps, mat = J_k_scalar(P, X_spt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    one_loop_coef = jnp.array(
        [2 * 1219 / 1470., 2 * 671 / 1029., 2 * 32 / 1715., 2 * 1 / 3., 2 * 62 / 35., 2 * 8 / 35., 1 / 3.])
    P22_mat = jnp.multiply(one_loop_coef, jnp.transpose(mat))
    P_22F = jnp.sum(P22_mat, 1)

    one_loop_coefG= jnp.array(
        [2*1003/1470, 2*803/1029, 2*64/1715, 2*1/3, 2*58/35, 2*12/35, 1/3])
    PsG, matG = J_k_scalar(P, X_sptG, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    P22G_mat = jnp.multiply(one_loop_coefG, jnp.transpose(matG))
    P_22G = jnp.sum(P22G_mat, 1)
    P_22F, P_22G = _apply_extrapolation(P_22F, P_22G)
    P_13G = P_IA_13G(k_original,P,)
    P_13F = P_IA_13F(k_original, P)
    P_0tE = P_22G-P_22F+P_13G-P_13F
    P_0tE = 2*P_0tE
    return P_0tE

def _get_sig3nl(P, X_spt, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    Ps, _ = J_k_scalar(P, X_spt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    sig3nl = Y1_reg_NL(k_extrap, Ps)
    sig3nl = _apply_extrapolation(sig3nl)
    return sig3nl

def _get_Pb1L(P, X_lpt, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    Ps, mat = J_k_scalar(P, X_lpt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    X1 = ((144. / 245.) * j000 - (176. / 343.) * j002 - (128. / 1715.) * j004 + (16. / 35.) * j1n11 - (
            16. / 35.) * j1n13)
    Y1 = Y1_reg_NL(k_extrap, Ps)
    Pb1L = X1 + Y1
    Pb1L = _apply_extrapolation(Pb1L)
    return Pb1L

def _get_Pb1L_2(P, X_lpt, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, 
                C_window=None, low_extrap=None, high_extrap=None, EK=None):
    Ps, mat = J_k_scalar(P, X_lpt, -2, m, N, n_pad, id_pad, 
                                k_extrap, k_final, k_size, l,
                                C_window=C_window, low_extrap=low_extrap, high_extrap=high_extrap, EK=EK)
    [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                        mat[5, :], mat[6, :]]
    X2 = ((16. / 21.) * j000 - (16. / 21.) * j002 + (16. / 35.) * j1n11 - (16. / 35.) * j1n13)
    Y2 = Y2_reg_NL(k_extrap, Ps)
    Pb1L_2 = X2 + Y2
    Pb1L_2 = _apply_extrapolation(Pb1L_2)
    return Pb1L_2


def J_k_scalar(P, X, nu, m, N, n_pad, id_pad, k_extrap, k_final, k_size, l, C_window=None, low_extrap=None, high_extrap=None, EK=None):
    
    pf, p, g_m, g_n, two_part_l, h_l = X

    if (low_extrap is not None):
        P = EK.extrap_P_low(P)
    
    if (high_extrap is not None):
        P = EK.extrap_P_high(P)
    
    P_b = P * k_extrap ** (-nu)
    
    if (n_pad > 0):
        P_b = jnp.pad(P_b, pad_width=(n_pad, n_pad), mode='constant', constant_values=0)
    
    c_m = fourier_coefficients(P_b, m, N, C_window)
    
    A_out = jnp.zeros((pf.shape[0], k_size))
    
    def process_single_row(i):
        C_l = convolution(c_m, c_m, g_m[i], g_n[i], h_l[i], None if two_part_l is None else two_part_l[i])
        
        l_size = l.shape[0]
        l_midpoint = l_size // 2  # Assuming l is centered around 0
        
        c_plus = C_l[l_midpoint:]  # Positive part (including 0)
        c_minus = C_l[:l_midpoint]  # Negative part
        
        # Combine them, dropping the last element of c_plus
        C_l_combined = jnp.concatenate([c_plus[:-1], c_minus])
        
        A_k = ifft(C_l_combined) * C_l_combined.size
        
        stride = max(1, A_k.shape[0] // k_size)
        
        return jnp.real(A_k[::stride][:k_size]) * pf[i] * k_final ** (-p[i] - 2)
    
    rows = jnp.arange(pf.shape[0])
    A_out = jax.vmap(process_single_row)(rows)
    
    m_midpoint = (m.shape[0] + 1) // 2  # Position of 0 in m
    c_m_positive = c_m[m_midpoint-1:]  # Select m >= 0
    
    P_out = irfft(c_m_positive) * k_final ** nu * N
    
    if (n_pad > 0):
        P_out = P_out[id_pad]
        A_out = A_out[:, id_pad]
    
    return P_out, A_out


def J_k_tensor(P, X, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, C_window=None, P_window=None, low_extrap=None, high_extrap=None, EK=None):
    
    pf, p, nu1, nu2, g_m, g_n, h_l = X

    if (low_extrap is not None):
        P = EK.extrap_P_low(P)
    
    if (high_extrap is not None):
        P = EK.extrap_P_high(P)
    
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
            
        if n_pad > 0:
            P_b1 = jnp.pad(P_b1, pad_width=(n_pad, n_pad), mode='constant', constant_values=0)
            P_b2 = jnp.pad(P_b2, pad_width=(n_pad, n_pad), mode='constant', constant_values=0)
            
        c_m = fourier_coefficients(P_b1, m, N, C_window)
        c_n = fourier_coefficients(P_b2, m, N, C_window)
        
        C_l = convolution(c_m, c_n, g_m_i, g_n_i, h_l_i)
        
        c_plus = C_l[l_midpoint:]
        c_minus = C_l[:l_midpoint]
        C_l_combined = jnp.concatenate([c_plus[:-1], c_minus])

        A_k = jnp.fft.ifft(C_l_combined) * C_l_combined.size
        return jnp.real(A_k[::2]) * pf_i * k_final ** p_i
    
    indices = jnp.arange(pf.size)
    A_out = jax.vmap(process_single_index)(indices)
    
    P_fin = jnp.sum(A_out, axis=0)
    
    if n_pad > 0:
        P_fin = P_fin[id_pad]
        A_out = A_out[:, id_pad]
    
    return P_fin, A_out


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

def convolution(c1, c2, g_m, g_n, h_l, two_part_l=None):
    from jax.scipy.signal import fftconvolve

    C_l = fftconvolve(c1 * g_m, c2 * g_n)

    if two_part_l is not None:
        C_l = C_l * h_l * two_part_l
    else:
        C_l = C_l * h_l

    return C_l


if __name__ == "__main__":
    from jax import vjp, jvp, grad, jit, vmap
    from fastpt.IA_ct import P_IA_13G as oldG, P_IA_13F as oldF
    from fastpt import FASTPT
    import os
    data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'benchmarking', 'Pk_test.dat')
    d = np.loadtxt(data_path)
    P = d[:, 1]
    k = d[:, 0]
    jpt = JAXPT(k, P_window=jnp.array([0.2, 0.2]), C_window=0.75, low_extrap=-5, high_extrap=3)
    fpt = FASTPT(k, low_extrap=-5, high_extrap=3)
    j = jpt.one_loop_dd_bias_b3nl(P)