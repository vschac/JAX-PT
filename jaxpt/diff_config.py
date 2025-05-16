import jax.numpy as jnp
from dataclasses import dataclass

class DiffConfig:
    
    def __init__(self):        
        #Power spectrum generation config
        self.pk_generation_method = 'jax-cosmo'
        self.pk_params = {'Omega_c': 0.12,
                            'Omega_b': 0.022,
                            'h': 0.69,
                            'n_s': 0.96,
                            'sigma8': 0.8,
                            'Omega_k': 0.0,
                            'w0': -1.0,
                            'wa': 0.0
                            }
        self.pk_diff_param = 'h'

        # Jax-PT terms/functions config
        self.term = None
        self.function = None
        self.P_window = None
        self.C_window = None

        # Jax Differentiation config
        self.diff_type = 'vector'
        self.diff_method = 'jacfwd'
        self.reduction_func = None
        self.tangent = None


    # Default reduction functions
       

    def build_and_validate(self):
        # Pk generation validation
        if self.pk_generation_method not in ['jax-cosmo', 'discoeb']:
            raise ValueError("Invalid pk_generation_method. Supported methods: ['jax-cosmo']")
        # Check is pk params is empty
        if not self.pk_params:
            raise ValueError("pk_params cannot be empty. Please provide a dictionary of parameters.")
        if self.pk_generation_method == 'jax-cosmo':
            jax_comso_params = ('Omega_c', 'Omega_b', 'h', 'n_s', 'sigma8', 'Omega_k', 'w0', 'wa')
            if not any(param in jax_comso_params for param in self.pk_params.keys()):
                raise ValueError(f"Invalid pk_params. Supported params: {jax_comso_params}")
            if self.pk_diff_param not in jax_comso_params:
                raise ValueError(f"Invalid pk_diff_param. Supported params: {jax_comso_params}")
        elif self.pk_generation_method == 'discoeb':
            discoeb_params = ('Omegam', 'Omegab', 'w_DE_0', 'w_DE_a', 'cs2_DE', 'Omegak', 'A_s', 'n_s', 'H0', 'Tcmb', 'YHe', 'Neff', 'Nmnu', 'mnu', 'k_p')
            if not any(param in discoeb_params for param in self.pk_params.keys()):
                raise ValueError(f"Invalid pk_params. Supported params: {discoeb_params}")
            if self.pk_diff_param not in discoeb_params:
                raise ValueError(f"Invalid pk_diff_param. Supported params: {discoeb_params}")
        

        # Jax-PT terms/functions validation
        valid_terms = [
            "P_E", "P_B", "P_A", "P_DEE", "P_DBB", "P_deltaE1", "P_0E0E", "P_0B0B",
            "P_s2E2", "P_d2E2", "P_d2E", "P_d20E", "P_s2E", "P_s20E", "P_kP1", "P_kP2", 
            "P_kP3", "P_Btype2", "P_deltaE2", "P_OV", "P_0tE", "P_0EtE", "P_E2tE", 
            "P_tEtE", "P_1loop", "Pd1d2", "Pd2d2", "Pd1s2", "Pd2s2", "Ps2s2", "sig4", 
            "sig3nl", "Pb1L", "Pb1L_2", "Pb1L_b2L", "Pb2L", "Pb2L_2", "P_d2tE", "P_s2tE", 
        ]
        
        valid_functions = [
            "one_loop_dd_bias_b3nl", "one_loop_dd_bias_lpt_NL", "IA_tt", "IA_mix",
            "IA_ta", "IA_ct", "gI_ct", "gI_ta", "gI_tt", "OV", "IA_der", "kPol"
        ]
        
        if self.term is not None and self.function is not None:
            raise ValueError("Both term and function cannot be provided simultaneously. Choose one.")
        
        if self.term is None and self.function is None:
            raise ValueError("Either term or function must be provided.")
        
        if self.term is not None:
            if self.term not in valid_terms:
                raise ValueError(f"Invalid term. Must be one of: {valid_terms}")
        
        if self.function is not None:
            if isinstance(self.function, str):
                if self.function not in valid_functions:
                    raise ValueError(f"Invalid function name. Must be one of: {valid_functions}")
            elif not callable(self.function):
                raise ValueError("Function must be either a valid function name string or a callable.")
        
        if self.P_window is not None:
            if not (isinstance(self.P_window, tuple) and len(self.P_window) == 2 and 
                    all(isinstance(val, (int, float)) for val in self.P_window)):
                raise ValueError("P_window must be None or a tuple of two numbers (p_low, p_high).")
        
        if self.C_window is not None and not isinstance(self.C_window, (int, float)):
            raise ValueError("C_window must be None or a number.")

        # Jax differentiation validation
        if self.diff_type not in ['scalar', 'vector']:
            raise ValueError("diff_type must be either 'scalar' or 'vector'")
        if self.diff_type == 'scalar':
            if self.diff_method not in ['grad', 'vjp']:
                raise ValueError("diff_method must be either 'grad' or 'vjp' for scalar differentiation")
            if self.reduction_func is None:
                print("Warning: reduction_func is None for scalar differentiation. Defaulting to sum function.")
                self.reduction_func = jnp.sum
        if self.diff_type == 'vector' and self.diff_method not in ['jacfwd', 'jvp', 'vjp']:
            raise ValueError("diff_method must be either 'jacfwd', 'jacrev', or 'jvp' for vector differentiation")
        if self.diff_method not in ['jacfwd', 'jacrev', 'vjp', 'jvp', 'grad']:
            raise ValueError("diff_method must be one of jacfwd, jacrev, vjp, jvp, or grad")
        if self.reduction_func is not None and not callable(self.reduction_func):
            raise ValueError("reduction_func must be a callable function")
        if self.tangent is not None and not isinstance(self.tangent, jnp.ndarray):
            raise ValueError("tangent must be a jnp.ndarray")
        
        return DiffConfigDC(
            pk_generation_method=self.pk_generation_method,
            pk_params=self.pk_params,
            pk_diff_param=self.pk_diff_param,
            term=self.term,
            function=self.function,
            P_window=self.P_window,
            C_window=self.C_window,
            diff_type=self.diff_type,
            diff_method=self.diff_method,
            reduction_func=self.reduction_func,
            tangent=self.tangent
        )
    
    def __repr__(self):
        return f"DiffConfig(pk_generation_method={self.pk_generation_method},\npk_params={self.pk_params},\npk_diff_param={self.pk_diff_param},\nterm={self.term},\nfunction={self.function},\nP_window={self.P_window},\nC_window={self.C_window},\ndiff_type={self.diff_type},\ndiff_method={self.diff_method},\nreduction_func={self.reduction_func},\ntangent={self.tangent})"
    def __str__(self):
        return self.__repr__()
    

@dataclass(frozen=True)
class DiffConfigDC():
    pk_generation_method: str
    pk_params: dict
    pk_diff_param: str
    term: str
    function: callable
    P_window: jnp.ndarray
    C_window: float
    diff_type: str
    diff_method: str
    reduction_func: callable
    tangent: jnp.ndarray

    def __post_init__(self):
        if self.pk_params is None:
            self.pk_params = {
                'Omega_c': 0.12,
                'Omega_b': 0.022,
                'h': 0.69,
                'n_s': 0.96,
                'sigma8': 0.8,
                'Omega_k': 0.0,
                'w0': -1.0,
                'wa': 0.0
            }


if __name__ == "__main__":
    # Example usage
    config = DiffConfig()
    config.pk_generation_method = 'jax-cosmo'
    config.pk_params = {'Omega_c': 0.12,
                        'Omega_b': 0.022,
                        'h': 0.69,
                        'n_s': 0.96,
                        'sigma8': 0.8,
                        'Omega_k': 0.0,
                        'w0': -1.0,
                        'wa': 0.0
                        }
    config.pk_diff_param = 'h'
    config.term = 'P_1loop'
    config.function = None
    config.P_window = None
    config.C_window = None
    config.diff_type = 'vector'
    config.diff_method = 'jacfwd'
    config.reduction_func = None
    config.tangent = None

    print(config)
    # validated_config = config.build_and_validate()
    # print(validated_config)