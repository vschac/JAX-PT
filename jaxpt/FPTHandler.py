import numpy as np
import inspect
from fastpt import FASTPT
from numpy import pi, log
import os

class FPTHandler:
    """
    Handler class for FAST-PT that simplifies function calls and result management.
    
    This class provides a simplified interface for working with FAST-PT functions,
    with features including power spectra generation, saving/loading outputs, 
    direct access to specific power spectrum terms, and more.
    
    Parameters
    ----------
    fastpt_instance : FASTPT
        An initialized FASTPT instance.
    do_cache : bool, optional
        Whether to cache function results for repeated calls. Default is False.
    save_all : str, optional
        File format to save all results ('txt', 'csv', or 'json'). If one is provided, all outputs of the run function will be saved. Default is None.
    save_dir : str, optional
        Directory to save results. Default is 'outputs' directory in package location.
    max_cache_entries : int, optional
        Maximum number of results to keep in cache. Default is 500.
    **params : dict
        Default parameters to use for all function calls.
        
    Examples
    --------
    >>> from fastpt import FASTPT, FPTHandler
    >>> import numpy as np
    >>> k = np.logspace(-3, 1, 200)
    >>> fpt = FASTPT(k)
    >>> handler = FPTHandler(fpt, C_window=0.75)
    >>> P = handler.generate_power_spectra()
    >>> handler.update_default_params(P=P)
    >>> P_1loop = handler.get('P_E')  # Direct access to the P_E term of fpt.IA_tt
    """
    def __init__(self, fastpt_instance: FASTPT, do_cache=False, save_all=None, save_dir=None, max_cache_entries=500, **params):
        if fastpt_instance is None:
            raise ValueError("You must provide a valid FASTPT instance.")
        self.__fastpt = fastpt_instance
        self.cache = {}
        self.do_cache = do_cache
        self.max_cache_entries = max_cache_entries
        self.save_all = save_all
        
        # Set default output directory if none specified
        if save_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        else:
            self.output_dir = save_dir
            
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.default_params = {}
        if params:
            try:
                self.default_params = self.fastpt._validate_params(**params)
            except ValueError as e:
                if "You must provide an input power spectrum array" in str(e):
                    print("No power spectrum provided. You'll need to provide 'P' in each function call.")
                    self.default_params = params
                else:
                    raise e

        #Commented out terms have not been implemented yet
        self.term_sources = {
            "P_1loop": ("one_loop_dd", None),
            "Ps": ("one_loop_dd", None),
            "Pd1d2": ("one_loop_dd_bias", None),  
            "Pd2d2": ("one_loop_dd_bias", None),
            "Pd1s2": ("one_loop_dd_bias", None),
            "Pd2s2": ("one_loop_dd_bias", None),
            "Ps2s2": ("one_loop_dd_bias", None),
            "sig4": ("one_loop_dd_bias", None),
        
            "sig3nl": ("one_loop_dd_bias_b3nl", None),
        
            "Pb1L": ("one_loop_dd_bias_lpt_NL", None),
            "Pb1L_2": ("one_loop_dd_bias_lpt_NL", None),
            "Pb1L_b2L": ("one_loop_dd_bias_lpt_NL", None),
            "Pb2L": ("one_loop_dd_bias_lpt_NL", None),
            "Pb2L_2": ("one_loop_dd_bias_lpt_NL", None),
        
            "P_E": ("IA_tt", "X_IA_E", lambda x: 2 * x),
            "P_B": ("IA_tt", "X_IA_B", lambda x: 2 * x),
        
            "P_A": ("IA_mix", "X_IA_A", lambda x: 2 * x),
            "P_Btype2": ("IA_mix", None),
            "P_DEE": ("IA_mix", "X_IA_DEE", lambda x: 2 * x),
            "P_DBB": ("IA_mix", "X_IA_DBB", lambda x: 2 * x),
        
            "P_deltaE1": ("IA_ta", "X_IA_deltaE1", lambda x: 2 * x),
            "P_deltaE2": ("IA_ta", None),
            "P_0E0E": ("IA_ta", "X_IA_0E0E", None),
            "P_0B0B": ("IA_ta", "X_IA_0B0B", None),
        
            # "P_gb2sij": ("IA_gb2", "X_IA_gb2_F2", lambda x: 2 * x),
            # "P_gb2dsij": ("IA_gb2", "X_IA_gb2_fe", lambda x: 2 * x),
            # "P_gb2sij2": ("IA_gb2", "X_IA_gb2_he", lambda x: 2 * x),

            "P_der": ("IA_der", None),

            "P_0tE": ("IA_ct", None),
            "P_0EtE": ("IA_ct", None),
            "P_E2tE": ("IA_ct", None),
            "P_tEtE": ("IA_ct", None),
        
            "P_d2tE": ("gI_ct", None),
            "P_s2tE": ("gI_ct", None),
        
            "P_s2E2": ("gI_tt", "X_IA_gb2_S2he", lambda x: 2 * x),
            "P_d2E2": ("gI_tt", "X_IA_gb2_he", lambda x: 2 * x),
        
            "P_d2E": ("gI_ta", "X_IA_gb2_F2", lambda x: 2 * x),
            "P_d20E": ("gI_ta", "X_IA_gb2_fe", lambda x: 2 * x),
            "P_s2E": ("gI_ta", "X_IA_gb2_S2F2", lambda x: 2 * x),
            "P_s20E": ("gI_ta", "X_IA_gb2_S2fe", lambda x: 2 * x),
        
            "P_OV": ("OV", None),
        
            "P_kP1": ("kPol", "X_kP1", lambda x: x / (80 * pi ** 2)),
            "P_kP2": ("kPol", "X_kP2", lambda x: x / (160 * pi ** 2)),
            "P_kP3": ("kPol", "X_kP3",lambda x: x / (80 * pi ** 2)),
        
            # "A1": ("RSD_components", 0),
            # "A3": ("RSD_components", 1),
            # "A5": ("RSD_components", 2),
            # "B0": ("RSD_components", 3),
            # "B2": ("RSD_components", 4),
            # "B4": ("RSD_components", 5),
            # "B6": ("RSD_components", 6),
            # "P_Ap1": ("RSD_components", 7),
            # "P_Ap3": ("RSD_components", 8),
            # "P_Ap5": ("RSD_components", 9),
        
            # "ABsum_mu2": ("RSD_ABsum_components", 0),
            # "ABsum_mu4": ("RSD_ABsum_components", 1),
            # "ABsum_mu6": ("RSD_ABsum_components", 2),
            # "ABsum_mu8": ("RSD_ABsum_components", 3),
        
            # "ABsum": ("RSD_ABsum_components", 0),

            # "P_IRres": ("IRres", 0),
        }
 
    @property
    def fastpt(self):
        """
        Get the underlying FASTPT instance.
        
        Returns
        -------
        FASTPT
            The FASTPT instance used by this handler.
        """
        return self.__fastpt


    def _get_function_params(self, func):
        """ Returns both required and optional parameter names for a given FASTPT function. """
        signature = inspect.signature(func)
        required_params = []
        optional_params = []
    
        for param_name, param in signature.parameters.items():
            # Skip self parameter and *args/**kwargs
            if (param_name == 'self' or 
                param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)):
                continue
            
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)
            else:
                optional_params.append(param_name)
    
        return {
            'required': required_params,
            'optional': optional_params,
            'all': required_params + optional_params
        }


    def _cache_result(self, function_name, params, result):
        """ Stores results uniquely by function name and its specific parameters. """
        if len(self.cache) >= self.max_cache_entries:
            print("Max cache size reached. Removing oldest entry.")
            self.cache.pop(next(iter(self.cache)))
        hashable_params = self._convert_to_hashable(params)
        self.cache[(function_name, hashable_params)] = result
    

    def _convert_to_hashable(self, params):
        """Convert parameters to hashable format, handling numpy arrays specially"""
        hashable_params = []
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                hashable_params.append((k, hash(v.tobytes())))
            else:
                hashable_params.append((k, v))
        return tuple(sorted(hashable_params))
    
    def _prepare_function_params(self, func, override_kwargs):
        """Prepares and validates parameters for a FASTPT function."""
        if override_kwargs: 
            self.fastpt._validate_params(**override_kwargs)
    
        merged_params = {**self.default_params, **override_kwargs}

        params_info = self._get_function_params(func)
        missing_params = [p for p in params_info['required'] if p not in merged_params]
    
        if missing_params:
            raise ValueError(f"Missing required parameters for '{func.__name__}': {missing_params}. "
                        f"Please recall with the missing parameters.")

        # Return only the params the function actually needs
        passing_params = {k: v for k, v in merged_params.items() if k in params_info['all']}
        return passing_params, params_info


    def run(self, function_name, save_type=None, save_dir=None, **override_kwargs):
        """
        Run a FAST-PT function with validated parameters.
        
        This method calls the specified FAST-PT function, handles parameter validation,
        caches results if enabled, and optionally saves the output to a file.
        
        Parameters
        ----------
        function_name : str
            Name of the FAST-PT function to run
        save_type : str, optional
            File format to save results ('txt', 'csv', or 'json')
        save_dir : str, optional
            Directory to save results. Defaults to the class's output_dir.
        **override_kwargs : dict
            Additional parameters to pass to the FAST-PT function
            
        Returns
        -------
        result : object
            Result from the FAST-PT function call, typically a tuple of numpy arrays
            
        Raises
        ------
        ValueError
            If the function is not found or parameter validation fails
            
        Examples
        --------
        >>> handler = FPTHandler(fpt, P=P, C_window=0.75)
        >>> P_1loop_result = handler.run('one_loop_dd')
        >>> ia_result = handler.run('IA_tt', save_type='csv')
        """
        if not hasattr(self.fastpt, function_name):
            raise ValueError(f"Function '{function_name}' not found in FASTPT.")

        if 'save_type' in override_kwargs:
            save_param = override_kwargs.pop('save_type')
            if save_type is None:
                save_type = save_param
                
        if 'save_dir' in override_kwargs:
            save_dir = override_kwargs.pop('save_dir')

        output_directory = save_dir if save_dir is not None else self.output_dir

        func = getattr(self.fastpt, function_name)
        passing_params, _ = self._prepare_function_params(func, override_kwargs)
        
        if self.do_cache:
            cache_key = (function_name, self._convert_to_hashable(passing_params))
            if cache_key in self.cache:
                print(f"Using cached result for {function_name}")
                return self.cache[cache_key]

        result = func(**passing_params)
        if self.do_cache: 
            self._cache_result(function_name, passing_params, result)
        if save_type is not None: 
            self.save_output(result, function_name, type=save_type, output_dir=output_directory)
        elif self.save_all is not None: 
            self.save_output(result, function_name, type=self.save_all, output_dir=output_directory)
        return result
    
    def bulk_run(self, func_names, power_spectra, flip=False, verbose=False, **override_kwargs):
        """
        Run multiple functions with multiple power spectra.
        
        This method provides a convenient way to run multiple FAST-PT functions
        with different power spectra and collect all the results.
        
        Parameters
        ----------
        func_names : list of str
            List of FAST-PT function names to call
        power_spectra : list of array_like
            List of power spectra to use for each function call
        flip : bool, optional
            Whether to run each function at every power spectrum or every power spectrum at each function
            (Default is False, which runs each function at every power spectrum)
        verbose : bool, optional
            Whether to print progress messages
        **override_kwargs : dict
            Additional parameters to pass to all function calls
        
        Returns
        -------
        dict
            Results keyed by (function_name, power_spectrum_index)
            
        Examples
        --------
        >>> k = np.logspace(-3, 1, 200)
        >>> P1 = k**(-1.5)  # Example power spectrum 1
        >>> P2 = k**(-1.0)  # Example power spectrum 2
        >>> handler = FPTHandler(fpt, C_window=0.75)
        >>> results = handler.bulk_run(['one_loop_dd', 'IA_tt'], [P1, P2])
        >>> one_loop_P1 = results[('one_loop_dd', 0)]
        """
        results = {}
        if flip:
            for i, P in enumerate(power_spectra):
                for func_name in func_names:
                    params = {**self.default_params, **override_kwargs, 'P': P}
                    if verbose: print(f"Running {func_name} with power spectrum {i}")
                    results[(func_name, i)] = self.run(func_name, **params)
        else:
            for func_name in func_names:
                for i, P in enumerate(power_spectra):
                    params = {**self.default_params, **override_kwargs, 'P': P}
                    if verbose: print(f"Running {func_name} with power spectrum {i}")
                    results[(func_name, i)] = self.run(func_name, **params)
        return results
       
    def get(self, *terms, **override_kwargs):
        """
        Get specific power spectrum terms directly.
        
        This method provides direct access to specific power spectrum components
        without needing to know which FAST-PT function calculates them.
        
        Parameters
        ----------
        *terms : str
            Names of power spectrum terms to retrieve
        **override_kwargs : dict
            Parameters for the underlying FAST-PT calculations
            
        Returns
        -------
        term or dict
            If a single term is requested, returns that term directly.
            If multiple terms are requested, returns a dictionary mapping
            term names to their values.
            
        Raises
        ------
        ValueError
            If a requested term is not found or parameters are invalid
            
        Examples
        --------
        >>> handler = FPTHandler(fpt, P=P_linear, C_window=0.75)
        >>> P_1loop = handler.get('P_1loop')
        >>> ia_terms = handler.get('P_E', 'P_B')
        """
        if not terms:
            raise ValueError("At least one term must be provided.")
        output = {}
        unique_funcs = {
                    "P_Btype2": "_get_P_Btype2",
                    "P_deltaE2": "_get_P_deltaE2",
                    "P_der": "IA_der",
                    "P_OV": "OV",
                    "P_0tE": "_get_P_0tE",
                    "P_0EtE": "_get_P_0EtE",
                    "P_E2tE": "_get_P_E2tE",
                    "P_tEtE": "_get_P_tEtE",
                    "P_1loop": "_get_P_1loop",
                    "Ps": "_get_Ps",
                    "Pd1d2": "_get_Pd1d2",
                    "Pd2d2": "_get_Pd2d2",
                    "Pd1s2": "_get_Pd1s2",
                    "Pd2s2": "_get_Pd2s2",
                    "Ps2s2": "_get_Ps2s2",
                    "sig4": "_get_sig4",
                    "sig3nl": "_get_sig3nl",
                    "Pb1L": "_get_Pb1L",
                    "Pb1L_2": "_get_Pb1L_2",
                    "Pb1L_b2L": "_get_Pb1L_b2L",
                    "Pb2L": "_get_Pb2L",
                    "Pb2L_2": "_get_Pb2L_2",
                    "P_d2tE": "_get_P_d2tE",
                    "P_s2tE": "_get_P_s2tE"
                }
        for term in terms:
            if term not in self.term_sources:
                raise ValueError(f"Term '{term}' not found in FASTPT.")
            if term in unique_funcs.keys(): #Terms that have their own unique functions
                func_name = unique_funcs[term]
                func = getattr(self.fastpt, func_name)
            
                passing_params, _ = self._prepare_function_params(func, override_kwargs)
                result = func(**passing_params)

            else:
                func_name = self.term_sources[term][0]
                func = getattr(self.fastpt, func_name)
                passing_params, params_info = self._prepare_function_params(func, override_kwargs)

                compute_func = getattr(self.fastpt, "compute_term")

                X_source = self.term_sources[term][1]
                operation = self.term_sources[term][2]

                # Handle case where we need multiple X terms (like for ctbias)
                if isinstance(X_source, tuple):
                    X_names = X_source
                    X_terms = []
                    for name in X_names:
                        if name in dir(self.fastpt):
                            X_terms.append(getattr(self.fastpt, name))
                        else:
                            raise AttributeError(f"'{name}' not found in FASTPT")
                    result = compute_func(term, tuple(X_terms), operation=operation, **passing_params)
                else:
                    # Standard case with a single X tracer
                    X_term = getattr(self.fastpt, X_source)
                    result = compute_func(term, X_term, operation=operation, **passing_params)                
                
            output[term] = result

        # If only one term was requested, return just that value
        if len(output) == 1 and len(terms) == 1:
            return output[list(output.keys())[0]]
        return output
    
    def get_tracer(self, tracer_name, **override_kwargs):
        """
        Get Fast-PT terms relevant to a specific tracer.
        
        Parameters
        ----------
        tracer_name : str
            Name of tracer terms to retrieve
        **override_kwargs : dict
            Parameters for the underlying FAST-PT calculations
            
        Returns
        -------
        dict
            Returns a dictionary of key-value pairs where the key is the term name
            and the value is the calculated term from Fast-PT.
            
        Raises
        ------
        ValueError
            If a requested tracer is not found or parameters are invalid

        Notes
        -----
            Some terms have different names in FAST-PT than in CCL, see the "aliases" dictionary.
            
        Examples
        --------
        >>> handler = FPTHandler(fpt, P=P, C_window=0.75)
        >>> P_1loop = handler.get('pgm')
        >>> #returns Pd1d2, Pd1s2, Pd1p3 (sig3nl)

        """
        aliases = {
            #Names used in CCL for FASTPT terms

            "Pd1p3" : "sig3nl",

            "a00e": "P_deltaE1", "c00e": "P_deltaE2", "a0e2": "P_A",
            "b0e2": "P_Btype2", "tijsij" : "P_0tE", "gb2tij": "P_d2tE", 
            "s2tij": "P_s2tE", "gb2sij" : "P_d2E", "gb2dsij" : "P_d20E", "gb2sij2" : "P_d2E2",
            "s2sij" : "P_s2E", "s2dsij" : "P_s20E", "s2sij2" : "P_s2E2",

            "a0e0e": "P_0E0E", "a0b0b": "P_0B0B", "ae2e2": "P_E", "ab2b2": "P_B",
            "a0e2": "P_A", "b0e2": "P_Btype2", "d0ee2": "P_DEE", "d0bb2": "P_dDBB",
            "tijdsij" : "P_0EtE", "tij2sij" : "P_E2tE", "tijtij" : "P_tEtE", "Pak2" : "P_der", #<< Sometimes der sometimes non fpt term?

        }
        tracer_map = {
            "pgg": ("Pd1d2", "Pd2d2", "Pd1s2", "Pd2s2", "Ps2s2","Pd1p3"), 
            #^^ also needs Pd1d1 and Pd1k2
            "pgi": ("a00e", "c00e", "a0e2", "b0e2", "tijsij", "gb2tij", "s2tij",
                    "gb2sij", "gb2dsij", "gb2sij2", "s2sij", "s2dsij", "s2sij2",
                    "sig3nl"),
            #^^ also needs Pd1d1
            "pgm": ("Pd1d2", "Pd1s2", "Pd1p3"),
            #^^ also needs Pd1d1 and Pd1k2
            "pii": ("a00e", "c00e", "a0e0e", "a0b0b", "ae2e2", "ab2b2", "a0e2", 
                    "b0e2", "d0ee2", "d0bb2", "tijsij", "tijdsij", "tij2sij", "tijtij", "Pak2"),
            #^^ also needs Pd1d1, Pak2 has a weird if chek_h
            "pim": ("a00e", "c00e", "a0e2", "b0e2", "tijsij", "Pak2"),
            #^^ also needs Pd1d1, Pak2 has a weird if chek_h
            "pmm": ("P_1loop")
        }

        if tracer_name not in tracer_map.keys():
            raise ValueError(f"Tracer '{tracer_name}' not currently supported. Available tracers are: {list(tracer_map.keys())}")
        
        result = {}
        for term in tracer_map[tracer_name]:
            try:
                calc_term = self.get(term, **override_kwargs)
            except ValueError as e:
                error_msg = str(e)
                if "not found in FASTPT" in error_msg:
                    if term in aliases:
                        calc_term = self.get(aliases[term], **override_kwargs)
                    else:
                        raise e
                else:
                    raise e
            result[term] = calc_term
        return result


    def clear_cache(self, function_name=None):
        """
        Clear cached function results.
        
        Parameters
        ----------
        function_name : str, optional
            If provided, only clears cache for the specified function.
            If None, clears the entire cache.
            
        Examples
        --------
        >>> handler = FPTHandler(fpt, do_cache=True)
        >>> handler.run('one_loop_dd', P=P_linear)
        >>> handler.clear_cache('one_loop_dd')  # Clear just one_loop_dd results
        >>> handler.clear_cache()  # Clear all cached results
        """
        if function_name:
            self.cache = {key: value for key, value in self.cache.items() if key[0] != function_name}
            print(f"Cache cleared for '{function_name}'.")
        else:
            self.cache.clear()
            print("Cache cleared for all functions.")

    def show_cache_info(self):
        """
        Display information about the current cache state.
        
        Shows the number of entries in the cache, the maximum allowed entries,
        and the current usage percentage.
        
        Examples
        --------
        >>> handler = FPTHandler(fpt, do_cache=True, max_cache_entries=100)
        >>> handler.run('one_loop_dd', P=P_linear)
        >>> handler.show_cache_info()
        """
        num_entries = len(self.cache)
        print({
            "num_entries": num_entries,
            "max_entries": self.max_cache_entries,
            "usage_percent": (num_entries / self.max_cache_entries) * 100 if self.max_cache_entries > 0 else 0
        })


    def list_available_functions(self):
        """
        List all callable FAST-PT functions.
        
        Prints a list of all public functions available in the FAST-PT instance.
        
        Examples
        --------
        >>> handler = FPTHandler(fpt)
        >>> handler.list_available_functions()
        ['OV', 'IA_ct', 'gI_ct', 'gI_tt', 'IA_der', ...]
        """
        print([f for f in dir(self.fastpt) if callable(getattr(self.fastpt, f)) and not f.startswith("_")])

    def list_available_terms(self):
        """
        List all available power spectrum terms that can be requested via get().
        
        Prints available terms organized by the FAST-PT function that calculates them,
        along with any special parameter requirements.
        
        Examples
        --------
        >>> handler = FPTHandler(fpt)
        >>> handler.list_available_terms()
        """
        # Organize by function
        organized = {}
        for term, (func, _) in self.term_sources.items():
            if func not in organized:
                organized[func] = []
            organized[func].append(term)
        
        # Print in a nice format
        print("Available terms by function:")
        for func, terms in organized.items():
            print(f"\n{func}:")
            terms_str = ", ".join(sorted(terms))
            print(f"  {terms_str}")
        
        # Special parameter requirements
        special_params = {
            "RSD_components": ["f"],
            "RSD_ABsum_components": ["f"],
            "RSD_ABsum_mu": ["f", "mu_n"],
            "IRres": ["L", "h", "rsdrag"]
        }
    
        print("\nSpecial parameter requirements:")
        for func, params in special_params.items():
            print(f"{func}: requires {', '.join(params)}")
        
        return organized
    
    def clear_default_params(self):
        """
        Clear all default parameters.
        
        Removes all stored default parameters so they must be specified
        in subsequent function calls.
        
        Examples
        --------
        >>> handler = FPTHandler(fpt, P=P_linear, C_window=0.75)
        >>> handler.clear_default_params()
        >>> # Now P must be provided in each function call
        """
        self.default_params = {}
        print("Default parameters cleared.")

    def update_default_params(self, **params):
        """
        Update the default parameters used for all function calls.
        
        Parameters
        ----------
        **params : dict
            Parameters to set as defaults
            
        Raises
        ------
        ValueError
            If any parameters are invalid or inconsistent
            
        Examples
        --------
        >>> handler = FPTHandler(fpt)
        >>> handler.update_default_params(P=P_linear, C_window=0.75)
        >>> # Now these parameters will be used by default
        """
        self.default_params = {**self.default_params, **self.fastpt._validate_params(**params)}
        print("Default parameters updated.")

    def update_fastpt_instance(self, fastpt_instance: FASTPT):
        """
        Update the FAST-PT instance used by this handler.
        
        This method replaces the current FASTPT instance and clears the cache.
        
        Parameters
        ----------
        fastpt_instance : FASTPT
            New FASTPT instance to use
            
        Examples
        --------
        >>> k_new = np.logspace(-4, 2, 300)  # Different k range
        >>> fpt_new = FASTPT(k_new)
        >>> handler.update_fastpt_instance(fpt_new)
        """
        self.__fastpt = fastpt_instance
        self.clear_cache()
        print("FASTPT instance updated. Cache cleared.")

    def save_output(self, result, func_name, type="txt", output_dir=None):
        """
        Save calculation results to a file.
        
        Parameters
        ----------
        result : array_like or tuple of array_like
            Result to save
        func_name : str
            Name of the function that produced the result
        type : str, optional
            File format ('txt', 'csv', or 'json'). Default is 'txt'.
        output_dir : str, optional
            Directory to save the file. Default is the handler's output_dir.
            
        Raises
        ------
        ValueError
            If an invalid file type is specified
            
        Examples
        --------
        >>> handler = FPTHandler(fpt)
        >>> result = handler.run('one_loop_dd', P=P_linear)
        >>> handler.save_output(result, 'one_loop_dd', type='csv')
        """
        if type not in ("txt", "csv", "json"): 
            raise ValueError("Invalid file type. Must be 'txt', 'csv', or 'json'")
        
        import os
        save_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(save_dir, exist_ok=True)

        if func_name in ("one_loop_dd_bias_lpt_NL", "one_loop_dd_bias_b3nl", "one_loop_dd_bias"):
            for i, element in enumerate(result):
                if isinstance(element, float): # sig4 is of type float, converting it to np array
                    new_array = np.zeros(len(result[i-1]))
                    new_array[0] = element
                    result = list(result)
                    result[i] = new_array

        base_name = f"{func_name}_output.{type}"
        file_path = os.path.join(save_dir, base_name)
        
        counter = 1
        while os.path.exists(file_path):
            new_name = f"{func_name}_{counter}_output.{type}"
            file_path = os.path.join(save_dir, new_name)
            counter += 1
        
        try:
            if type == "txt":
                np.savetxt(file_path, np.transpose(result), header=f'{func_name}')
            elif type == "csv":
                import csv
                data_for_csv = []
                
                if isinstance(result, np.ndarray) and result.ndim == 1:
                    data_for_csv = [[x] for x in result]
                else:
                    # Try to handle as collection of arrays or values
                    transposed = np.transpose(result)
                    data_for_csv = transposed.tolist() if hasattr(transposed, 'tolist') else transposed
                
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if isinstance(result, tuple) or isinstance(result, list):
                        header = [f'{func_name}_{i}' for i in range(len(result))]
                    else:
                        header = [func_name]
                    writer.writerow(header)
                    writer.writerows(data_for_csv)
            elif type == "json":
                import json
                
                # Prepare data for JSON serialization (numpy arrays aren't directly JSON serializable)
                def numpy_to_python(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (tuple, list)):
                        return [numpy_to_python(item) for item in obj]
                    elif isinstance(obj, np.number):
                        return obj.item()
                    return obj
                
                json_data = {func_name: numpy_to_python(result)}
                with open(file_path, 'w') as jsonfile:
                    json.dump(json_data, jsonfile, indent=2)
            
            print(f"Output saved to {file_path}")
        except Exception as e:
            print(f"Error saving {func_name} output: {str(e)}")

    def load(self, file_path, load_dir=None):
        """ 
        Load a saved output file and return it in the same format as FASTPT outputs (tuple of numpy arrays)
        
        Args:
            file_path (str): Name or path of the file to load
            load_dir (str, optional): Directory to load file from. If None, uses default output directory
                                     If file_path already includes a directory, load_dir is ignored.
        
        Returns:
            tuple: A tuple of numpy arrays matching the original FASTPT function output format
        
        Examples
        --------
        >>> handler = FPTHandler(fpt)
        >>> result = handler.run('one_loop_dd', save_type='txt')
        >>> loaded_data = handler.load('one_loop_dd_output.txt')
        """
        import os
        import numpy as np
        import re
        
        # If file_path is an absolute path or already contains directory info, use it as is
        if os.path.isabs(file_path) or os.path.dirname(file_path):
            full_path = file_path
        else:
            # Otherwise, build path from load_dir or default output directory
            directory = load_dir if load_dir is not None else self.output_dir
            full_path = os.path.join(directory, file_path)
        
        # Get file extension
        _, ext = os.path.splitext(full_path)
        ext = ext.lower()
    
        # Check for valid extension before checking if file exists
        if ext not in (".txt", ".csv", ".json"):
            raise FileNotFoundError(f"Unsupported file extension: {ext}. Must be '.txt', '.csv', or '.json'")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File '{full_path}' not found.")

        
        # Extract function name from filename
        func_name = re.match(r'(.+?)(?:_\d+)?_output\.', os.path.basename(full_path))
        func_name = func_name.group(1) if func_name else None
        
        try:
            arrays = []
            
            if ext == ".txt":
                # Load and transpose to match original format
                loaded_data = np.loadtxt(full_path)
                # Split columns into separate arrays
                for i in range(loaded_data.shape[1]):
                    arrays.append(loaded_data[:, i])
                
            elif ext == ".csv":
                import csv
                with open(full_path, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    # Skip header row
                    next(reader)
                    # Read all rows
                    data_rows = list(reader)
                    
                    # Convert to numeric values
                    numeric_data = []
                    for row in data_rows:
                        numeric_data.append([float(val) for val in row])
                    
                    # Convert to numpy array
                    all_data = np.array(numeric_data)
                    
                    # Split columns into separate arrays (transposing to match original format)
                    for i in range(all_data.shape[1]):
                        arrays.append(all_data[:, i])
                    
            elif ext == ".json":
                import json
                with open(full_path, 'r') as jsonfile:
                    data = json.load(jsonfile)
                    # Get the function name from the data if available
                    if not func_name and len(data) == 1:
                        func_name = next(iter(data))
                    
                    # Get the data under the function name key
                    result_data = data[next(iter(data))]
                    
                    # Handle different possible structures in JSON
                    if isinstance(result_data, list):
                        # If result_data is a list of lists (multiple arrays)
                        if result_data and isinstance(result_data[0], list):
                            for arr in result_data:
                                arrays.append(np.array(arr))
                        else:
                            # Single array
                            arrays.append(np.array(result_data))
                    else:
                        # Single value or other structure
                        arrays.append(np.array([result_data]))
            else:
                raise ValueError(f"Unsupported file extension: {ext}. Must be '.txt', '.csv', or '.json'")
            
            # Handle special case for sig4 in bias functions - convert back to float
            # In one_loop_dd_bias and one_loop_dd_bias_b3nl, sig4 is at index 7
            # In one_loop_dd_bias_lpt_NL, sig4 is at index 6
            if func_name in ["one_loop_dd_bias", "one_loop_dd_bias_b3nl"] and len(arrays) > 7:
                # Check if the array is mostly zeros with one value
                if arrays[7].size > 1 and np.count_nonzero(arrays[7]) <= 1:
                    # Get the first non-zero value or the first value if all zeros
                    if np.any(arrays[7]):
                        sig4_value = arrays[7][np.nonzero(arrays[7])[0][0]]
                    else:
                        sig4_value = arrays[7][0]
                    arrays[7] = sig4_value
                    
            elif func_name == "one_loop_dd_bias_lpt_NL" and len(arrays) > 6:
                # Similar check for lpt_NL case
                if arrays[6].size > 1 and np.count_nonzero(arrays[6]) <= 1:
                    if np.any(arrays[6]):
                        sig4_value = arrays[6][np.nonzero(arrays[6])[0][0]]
                    else:
                        sig4_value = arrays[6][0]
                    arrays[6] = sig4_value
                    
            print(f"Output loaded from {full_path}")
            
            # Convert list of arrays to tuple to match FASTPT output format
            return tuple(arrays)
        
        except Exception as e:
            print(f"Error loading output from {full_path}: {str(e)}")
            return None
        

    def save_params(self, filename, output_dir=None, **params):
        """
        Save parameters to a compressed numpy .npz file.
        
        This method saves both numpy arrays and other Python objects (strings, floats, ints, etc.)
        in a single file. Arrays are stored directly, while non-array values are collected
        in a metadata dictionary and stored as a special array.
        
        Parameters
        ----------
        filename : str
            Path where parameters will be saved. If the path doesn't end with '.npz',
            the extension will be added automatically.
        output_dir : str, optional
            Directory to save parameters. If None, uses default output directory.
            If filename already contains a directory path, output_dir is ignored.
        **params : dict
            Parameters to save. If no parameters are provided, the handler's default
            parameters will be saved instead.
            
        Notes
        -----
        Non-array values are stored in a special array with key '__metadata__'.
        Arrays and scalars can be freely mixed in the parameters.
        
        Examples
        --------
        >>> handler = FPTHandler(fpt, P=P_linear, C_window=0.75)
        >>> # Save specific parameters
        >>> handler.save_params('my_params', P=P_linear, C_window=0.75, bias=1.5)
        >>> # Save default parameters
        >>> handler.save_params('default_params')
        >>> # Save to specific directory
        >>> handler.save_params('custom_params', output_dir='/path/to/save')
        """
        metadata = {}
        arrays = {}
        if not params and not self.default_params:
            raise ValueError("No parameters stored or provided to save.")
        if not params:
            print("Saving default params...")
            for param in self.default_params.keys():
                value = self.default_params[param]
                if isinstance(value, np.ndarray):
                    arrays[param] = value
                else:
                    metadata[param] = value
        else:
            for key, value in params.items():
                if isinstance(value, np.ndarray):
                    arrays[key] = value
                else:
                    metadata[key] = value
        
        if metadata:
            arrays['__metadata__'] = np.array([metadata], dtype=object)
        
        if os.path.isabs(filename):
            # If filename is absolute, use it directly
            full_path = filename
        else:
            # Otherwise, determine the base directory
            directory = output_dir if output_dir is not None else self.output_dir
            full_path = os.path.join(directory, filename)
        
        os.makedirs(os.path.dirname(os.path.abspath(full_path)), exist_ok=True)
        
        if not full_path.endswith('.npz'):
            full_path += '.npz'
        np.savez_compressed(full_path, **arrays)

    def load_params(self, filename, load_dir=None):
        """
        Load parameters from a saved .npz file.
        
        Loads both array and non-array parameters from a file created with save_params().
        Arrays are loaded directly, while scalar values are extracted from the metadata.
        
        Parameters
        ----------
        filename : str
            Path to the parameter file. If the path doesn't end with '.npz',
            the extension will be added automatically.
        load_dir : str, optional
            Directory to load parameters from. If None, uses default output directory.
            If filename already contains a directory path, load_dir is ignored.
            
        Returns
        -------
        dict
            Dictionary containing all loaded parameters.
            
        Notes
        -----
        This method automatically handles the separation between array parameters and
        scalar metadata that was created during saving.
        
        Examples
        --------
        >>> handler = FPTHandler(fpt)
        >>> params = handler.load_params('my_params.npz')
        >>> print(params.keys())
        >>> # Use loaded parameters in a calculation
        >>> result = handler.run('one_loop_dd', **params)
        >>> # Load from specific directory
        >>> params = handler.load_params('custom_params', load_dir='/path/to/load')
        """
        if os.path.isabs(filename) or os.path.dirname(filename):
            full_path = filename
        else:
            directory = load_dir if load_dir is not None else self.output_dir
            full_path = os.path.join(directory, filename)
        
        if not full_path.endswith('.npz'):
            full_path += '.npz'
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Parameter file '{full_path}' not found.")
        
        data = np.load(full_path, allow_pickle=True)
        params = {}
        for key in data.files:
            if key != '__metadata__':
                params[key] = data[key]
        if '__metadata__' in data.files:
            metadata = data['__metadata__'][0]
            params.update(metadata)
        
        return params

    def plot(self, data=None, terms=None, k=None, ax=None, title=None, 
             log_scale=True, legend_loc='best', grid=True, style=None,
             colors=None, linewidth=1.5, label_map=None, fig_size=(10, 7), 
             save_path=None, dpi=300, xlim=None, ylim=None, scale_factors=None,
             return_fig=False, show=True, **override_kwargs):
        """
        Create a plot of power spectrum terms.
        
        Parameters
        ----------
        data : dict or array-like, optional
            Data to plot. Can be:
            - Dictionary mapping labels to arrays (direct plotting)
            - Result from FAST-PT function call (will be plotted directly)
            - None (requires 'terms' parameter to get data)
        terms : str or list, optional
            Term name(s) to plot if data is not provided
        k : array-like, optional
            k values for x-axis. If None, uses handler's FASTPT k values
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure
        title : str, optional
            Plot title
        log_scale : bool or tuple, optional
            Whether to use log scale. If True, both axes are log.
            Can provide (x_log, y_log) tuple to specify each axis
        legend_loc : str, optional
            Location of legend. Default is 'best'
        grid : bool, optional
            Whether to display grid lines
        style : dict or list of dict, optional
            Plotting style(s) to apply to lines. Each dict can contain matplotlib
            line properties (color, linestyle, marker, alpha, etc.)
        colors : str or list, optional
            Color(s) for plots. If a list, cycles through colors
        linewidth : float, optional
            Line width for plots
        label_map : dict, optional
            Mapping from term names to display labels
        fig_size : tuple, optional
            Figure size (width, height) in inches
        save_path : str, optional
            Path to save figure. If None, figure is not saved
        dpi : int, optional
            DPI for saved figure
        xlim : tuple, optional
            (min, max) for x-axis
        ylim : tuple, optional
            (min, max) for y-axis
        scale_factors : dict or float, optional
            Scaling factor(s) for plotted quantities
        return_fig : bool, optional
            Whether to return the figure object
        show : bool, optional
            Whether to show the plot immediately
        **override_kwargs : dict
            Additional parameters for FAST-PT function calls
            
        Returns
        -------
        fig : matplotlib.figure.Figure, optional
            Figure object if return_fig is True
            
        Examples
        --------
        >>> handler = FPTHandler(fpt, P=P_linear, C_window=0.75)
        >>> # Basic plot from terms
        >>> handler.plot(terms=['P_1loop', 'P_E', 'P_B'])
        >>> # Plot with custom styling
        >>> handler.plot(terms='P_1loop', colors='red', title='1-loop SPT')
        >>> # Plot with style dictionaries for fine control
        >>> handler.plot(terms=['P_1loop', 'P_E'], style=[
        ...     {'color': 'red', 'linestyle': '--', 'marker': 'o', 'alpha': 0.8},
        ...     {'color': 'blue', 'linestyle': '-', 'linewidth': 2}
        ... ])
        """
        import matplotlib.pyplot as plt
        from itertools import cycle
        
        # Set up figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            fig = ax.figure
            
        # Set plot title if provided
        if title is not None:
            ax.set_title(title, fontsize=14)
            
        # Set up x-axis values (k)
        if k is None:
            k = self.fastpt.k_original
            
        # Default color cycle
        if colors is None:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        if isinstance(colors, str):
            colors = [colors]
        color_cycle = cycle(colors)
        
        # Set up label mapping
        if label_map is None:
            label_map = {}
            
        # Set up scale factors
        if scale_factors is None:
            scale_factors = {}
        if isinstance(scale_factors, (int, float)):
            default_scale = scale_factors
            scale_factors = {}
        else:
            default_scale = 1.0
        
        # Prepare style dictionaries
        if style is None:
            style = [{}]  # Empty dict = use defaults
        elif isinstance(style, dict):
            style = [style]  # Single style dict
        style_cycle = cycle(style)
        
        # Get data to plot
        to_plot = {}
        
        if data is not None:
            # Case 1: User provided data directly
            if isinstance(data, dict):
                to_plot = data
            elif isinstance(data, (list, tuple, np.ndarray)):
                if isinstance(data, (list, tuple)) and len(data) > 1:
                    for i, arr in enumerate(data):
                        label = label_map.get(i, f"Component {i}")
                        to_plot[label] = arr
                else:
                    label = label_map.get(0, "Data")
                    to_plot[label] = data[0] if isinstance(data, (list, tuple)) else data
            else:
                raise ValueError("Unsupported data format. Provide a dict, list, tuple, or numpy array.")
        elif terms is not None:
            # Case 2: User provided term names to plot
            if isinstance(terms, str):
                terms = [terms]
                
            # Get data for each term
            term_data = self.get(*terms, **override_kwargs)
            
            if len(terms) == 1:
                term_name = terms[0]
                display_label = label_map.get(term_name, term_name)
                to_plot[display_label] = term_data
            else:
                for term_name in terms:
                    display_label = label_map.get(term_name, term_name)
                    to_plot[display_label] = term_data[term_name]
        else:
            raise ValueError("Either 'data' or 'terms' must be provided.")
            
        # Plot each data series
        for label, data_array in to_plot.items():
            # Get current style and color
            curr_style = next(style_cycle).copy()  # Copy to avoid modifying the original
            
            # Add default color if not in style
            if 'color' not in curr_style:
                curr_style['color'] = next(color_cycle)
                
            # Set default linewidth if not in style
            if 'linewidth' not in curr_style:
                curr_style['linewidth'] = linewidth
                
            # Apply scale factor if provided for this label
            scale = scale_factors.get(label, default_scale)
            
            # Check if data contains negative values
            if isinstance(data_array, np.ndarray) and np.any(data_array < 0):
                # Plot positive values
                mask_pos = data_array >= 0
                if np.any(mask_pos):
                    ax.plot(k[mask_pos], scale * data_array[mask_pos], label=label, **curr_style)
                
                # Plot negative values with dashed lines for visibility
                mask_neg = data_array < 0
                if np.any(mask_neg):
                    neg_style = curr_style.copy()
                    neg_style['linestyle'] = neg_style.get('linestyle', '--')  # Default to dashed
                    ax.plot(k[mask_neg], scale * np.abs(data_array[mask_neg]), **neg_style)
            else:
                # Regular plotting for non-negative data
                ax.plot(k, scale * data_array, label=label, **curr_style)
                
        # Set scales
        if log_scale:
            if isinstance(log_scale, (list, tuple)):
                x_log, y_log = log_scale
            else:
                x_log = y_log = True
                
            if x_log:
                ax.set_xscale('log')
            if y_log:
                ax.set_yscale('log')
        
        # Set axis labels
        ax.set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]', fontsize=12)
        ax.set_ylabel(r'$P(k)$ [$h^{-3}$ Mpc$^3$]', fontsize=12)
        
        # Set limits if provided
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        # Add grid if requested
        if grid:
            ax.grid(True, alpha=0.3)
            
        # Add legend if there are labeled lines
        if to_plot:
            ax.legend(loc=legend_loc)
            
        # Apply tight layout
        fig.tight_layout()
            
        # Save figure if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
            
        # Show the plot if requested
        if show:
            plt.show()
            
        # Return figure if requested
        if return_fig:
            return fig
            
    def plot_comparison(self, results_dict, k=None, ratio=False, ratio_baseline=None,
                    fig_size=(12, 8), **plot_kwargs):
        """
        Create comparison plots for multiple results, optionally with ratio panel.
    
        Parameters
        ----------
        results_dict : dict
            Dictionary mapping labels to data arrays
        k : array-like, optional
            k values for x-axis. If None, uses handler's FASTPT k values
        ratio : bool, optional
            Whether to include a ratio panel
        ratio_baseline : str, optional
            Key in results_dict to use as baseline for ratio.
            If None, uses the first key in results_dict
        fig_size : tuple, optional
            Figure size (width, height) in inches
        **plot_kwargs : dict
            Additional keyword arguments passed to the plot method
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure

        Examples
        --------
        >>> handler = FPTHandler(fpt, P=P_linear, C_window=0.75)
        >>> P_1loop_result = handler.run('one_loop_dd')
        >>> ia_result = handler.run('IA_tt')
        >>> handler.plot_comparison({'1-loop': P_1loop_result, 'IA': ia_result})
        """
        import matplotlib.pyplot as plt
    
        if k is None:
            k = self.fastpt.k_original
        
        if not results_dict:
            raise ValueError("results_dict cannot be empty")
        
        if ratio_baseline is None:
            # Use first key as baseline if not specified
            ratio_baseline = list(results_dict.keys())[0]
        
        if ratio and ratio_baseline not in results_dict:
            raise ValueError(f"Ratio baseline '{ratio_baseline}' not found in results")
        
        # Create figure and axes
        if ratio:
            fig = plt.figure(figsize=fig_size)
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=(3, 1), hspace=0)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
        else:
            fig, ax1 = plt.subplots(figsize=fig_size)
            ax2 = None
        
        # Plot main data
        # Remove parameters we're handling explicitly to avoid conflicts
        plot_kwargs_filtered = {k: v for k, v in plot_kwargs.items() if k not in ('show', 'save_path', 'k', 'ax')}
        self.plot(data=results_dict, k=k, ax=ax1, show=False, return_fig=False, **plot_kwargs_filtered)
    
        # Plot ratio panel if requested
        if ratio:
            baseline_data = results_dict[ratio_baseline]
            ratio_data = {}
        
            for label, data in results_dict.items():
                if np.array_equal(data, baseline_data):
                    continue
                
                # Calculate ratio, handling potential zeros or NaNs
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_array = data / baseline_data
                    # Replace inf/NaN with zeros for plotting
                    ratio_array = np.nan_to_num(ratio_array, nan=0, posinf=0, neginf=0)
                
                ratio_data[f"{label}/{ratio_baseline}"] = ratio_array
            
            # Plot the ratios
            self.plot(data=ratio_data, k=k, ax=ax2, log_scale=(True, False), 
                    show=False, return_fig=False)
        
            # Add horizontal line at y=1.0
            ax2.axhline(y=1.0, color='k', linestyle='-', alpha=0.3)
        
            # Set y-label for ratio panel
            ax2.set_ylabel('Ratio', fontsize=12)
        
            # Remove x-label from top plot to avoid overlap
            ax1.set_xlabel('')
        
            # Only show x tick labels on bottom panel
            plt.setp(ax1.get_xticklabels(), visible=False)
        
        # Save figure if path provided
        if 'save_path' in plot_kwargs:
            fig.savefig(plot_kwargs['save_path'], dpi=plot_kwargs.get('dpi', 300), bbox_inches='tight')
            print(f"Figure saved to: {plot_kwargs['save_path']}")
        
        # Show plot
        if plot_kwargs.get('show', True):
            plt.show()
        
        return fig
    
    def generate_power_spectra(self, method='classy', mode='single', **kwargs):
        """
        Generate power spectra using the specified mode and method.
        
        Parameters
        ----------
        method : str
            Either 'classy' or 'camb'
        mode : str
            'single', 'bulk', or 'diff'

            - single: generate one power spectra with the given params, which should be passed as floats or bools
            - bulk: generate multiple power spectra with the given params, which should be passed as lists or np arrays. If any of the params lists are shorter than the longest, they will be padded with the last value.
            - diff: generate multiple power spectra with the given params, which should be passed as lists or np arrays. The length of the params lists must be 1 or 3, and a power spectrum will be generated varrying each of the parameters while holding the central column of values constant.
        **kwargs
            Cosmological parameters to pass to the appropriate method
            
            - For CLASSY: omega_b, omega_cdm, h, z
            - For CAMB: omega_b, omega_cdm, h, z, As, ns, halofit_version, k_hunit, nonlinear, H0, kmax, hubble_units, extrap_kmax, k_per_logint
        """
        method = method.lower()
        if method not in ('classy', 'camb'):
            raise ValueError("Invalid method. Choose either 'classy' or 'camb'.")
            
        if mode not in ('single', 'bulk', 'diff'):
            raise ValueError("Invalid mode. Choose 'single', 'bulk', or 'diff'.")

        if mode not in ('single', 'bulk', 'diff'):
            raise ValueError("Invalid mode. Choose 'single', 'bulk', or 'diff'.")

        if method == 'classy':
            camb_specific_params = {'As', 'ns', 'k_hunit', 'nonlinear', 'H0', 'kmax', 
                            'hubble_units', 'extrap_kmax', 'k_per_logint', 'halofit_version'}
            class_params = {'omega_cdm', 'h', 'omega_b', 'z'}
            camb_params_used = [param for param in camb_specific_params if param in kwargs]
            if camb_params_used:
                import warnings
                warnings.warn(f"CAMB-specific parameters will be ignored when using CLASS: {camb_params_used}")
                kwargs = {k: v for k, v in kwargs.items() if k in class_params}
        
        if mode == 'diff':
            return self._diff_power_spectra(method, **kwargs)
        elif mode == 'bulk':
            return self._bulk_power_spectra(method, **kwargs)
        else: 
            for key, val in kwargs.items():
                if isinstance(val, (list, np.ndarray)):
                    raise ValueError(f"Parameter '{key}' must be a single value for single mode.")
                    
            if method == 'classy':
                return self._class_power_spectra(**kwargs)
            else: 
                return self._camb_power_spectra(**kwargs)
    
    def _bulk_power_spectra(self, method, **params):
        max_len = 1
        for param_name, value in params.items():
            if isinstance(value, (list, np.ndarray)):
                max_len = max(max_len, len(value))
        
        param_arrays = {}
        for param_name, value in params.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) < max_len:
                    param_arrays[param_name] = np.pad(value, (0, max_len - len(value)), mode='edge')
                else:
                    param_arrays[param_name] = np.array(value)
            else:
                param_arrays[param_name] = np.full(max_len, value)
            
        if method.lower() == 'classy':
            output = []
            for i in range(max_len):
                output.append(self._class_power_spectra(
                    omega_b=param_arrays['omega_b'][i],
                    omega_cdm=param_arrays['omega_cdm'][i],
                    h=param_arrays['h'][i],
                    z=param_arrays['z'][i]
                ))
            
            return output[0] if len(output) == 1 else output
            
        elif method.lower() == 'camb':
            output = []
            for i in range(max_len):
                output.append(self._camb_power_spectra(
                    omega_b=param_arrays['omega_b'][i],
                    omega_cdm=param_arrays['omega_cdm'][i],
                    h=param_arrays['h'][i],
                    z=param_arrays['z'][i],
                    As=param_arrays['As'][i],
                    ns=param_arrays['ns'][i],
                    halofit_version=param_arrays['halofit_version'][i],
                    k_hunit=param_arrays['k_hunit'][i],
                    nonlinear=param_arrays['nonlinear'][i],
                    H0=param_arrays['H0'][i],
                    kmax=param_arrays['kmax'][i],
                    hubble_units=param_arrays['hubble_units'][i],
                    extrap_kmax=param_arrays['extrap_kmax'][i],
                    k_per_logint=param_arrays['k_per_logint'][i],
                ))
            
            return output[0] if len(output) == 1 else output
        else:
            raise ValueError("Invalid method. Choose either 'classy' or 'camb'.")

    def _diff_power_spectra(self, method, **kwargs):
        if method not in ('classy', 'camb'):
            raise ValueError("Invalid method. Choose either 'classy' or 'camb'.")
        
        diff_params = {
            'omega_cdm': kwargs.get('omega_cdm', [0.12]),
            'h': kwargs.get('h', [0.67]),
            'omega_b': kwargs.get('omega_b', [0.022]),
            'z': kwargs.get('z', [0.0]),
        }
        
        has_param_with_length_3 = any(
            isinstance(value, (list, np.ndarray)) and len(value) == 3
            for value in diff_params.values()
        )
        if not has_param_with_length_3:
            raise ValueError("At least one parameter must have length 3 to use diff mode.")
        
        camb_params = {
            'As': kwargs.get('As', 2.1e-9),
            'ns': kwargs.get('ns', 0.96),
            'k_hunit': kwargs.get('k_hunit', True),
            'nonlinear': kwargs.get('nonlinear', False),
            'H0': kwargs.get('H0', None),
            'kmax': kwargs.get('kmax', None),
            'hubble_units': kwargs.get('hubble_units', True),
            'extrap_kmax': kwargs.get('extrap_kmax', None),
            'k_per_logint': kwargs.get('k_per_logint', None),
            'halofit_version': kwargs.get('halofit_version', 'mead')
        }
        
        for key, value in diff_params.items():
            if key == 'z':
                diff_params['z'] = [value] if isinstance(value, (int, float)) else value
                continue
                
            if isinstance(value, (int, float)):
                diff_params[key] = [value]            
            if not isinstance(diff_params[key], (list, np.ndarray)):
                raise ValueError(f"Parameter '{key}' must be a list or numpy array.")                
            if len(diff_params[key]) not in (1, 3):
                raise ValueError(f"Parameter '{key}' must have length 1 or 3.")                
            if len(diff_params[key]) == 1:
                diff_params[key] = [diff_params[key][0]] * 3
        
        result = {}
        
        compute_func = self._class_power_spectra if method == 'classy' else self._camb_power_spectra
        
        for z in diff_params['z']:
            param_combinations = []
            
            center_values = [diff_params[p][1] for p in ['omega_cdm', 'h', 'omega_b']]
            param_combinations.append(center_values)
            
            for param_idx, param in enumerate(['omega_cdm', 'h', 'omega_b']):
                param_values = diff_params[param]
                
                if param_values[0] == param_values[1] == param_values[2]:
                    continue
                    
                for val_idx in [0, 2]:
                    values = center_values.copy()
                    values[param_idx] = param_values[val_idx]
                    param_combinations.append(values)

            for combo in param_combinations:
                omega_cdm, h, omega_b = combo
                key = (omega_cdm, h, omega_b, z)
                
                if method == 'classy':
                    result[key] = compute_func(
                        omega_cdm=omega_cdm,
                        h=h,
                        omega_b=omega_b,
                        z=z
                    )
                else: 
                    result[key] = compute_func(
                        omega_cdm=omega_cdm,
                        h=h,
                        omega_b=omega_b,
                        z=z,
                        **camb_params
                    )
        
        return result

    def _class_power_spectra(self, omega_cdm=0.12, h=0.67, omega_b=0.022, z=0.0):
        try:
            from classy import Class
        except ImportError as e:
            raise ImportError("Classy is not installed. Please install it to use this function.") from e
        k = self.fastpt.k_original
        k_max = max(k)
        params = {
            'output': 'mPk',
            'P_k_max_1/Mpc': k_max * 1.1,
            'z_max_pk': z,
            'h': h,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            # 'ln10^{10}A_s':  np.log10(2.1e-9*1e10),
            # 'n_s':           0.965, #Update these two to be passed as kwargs if needed
        }
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
        output = np.array([cosmo.pk(k, z) for k in k])
        # output = np.array([cosmo.pk(ki, 0.0) for ki in k]) * h**3
        cosmo.struct_cleanup()
        cosmo.empty()
        return output

    def _camb_power_spectra(self,
                                z: float = 0.0,
                                nonlinear: bool = True,
                                h: float = 0.67,
                                H0: float = None,
                                omega_b: float = 0.022,
                                omega_cdm: float = 0.122,
                                As: float = 2.1e-9,
                                ns: float = 0.965,
                                halofit_version: str = 'mead',
                                kmax: float = None,
                                hubble_units: bool = True,
                                k_hunit: bool = True,
                                extrap_kmax: float = None,
                                k_per_logint: int = None
                               ):
        try:
            import camb
        except ImportError as e:
            raise ImportError("CAMB is not installed. Please install it to use this function.") from e
        k = self.fastpt.k_original

        if H0 is None: H0 = h * 100
         # 1) Set up CAMB parameters
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=omega_b, omch2=omega_cdm)                 # standard cosmology
        pars.InitPower.set_params(As=As, ns=ns)                            # primordial spectrum

        # 2) Matter power settings
        kmax = kmax or float(np.max(k))
        pars.set_matter_power(redshifts=[z], kmax=kmax, k_per_logint=k_per_logint)

        # 3) Choose HALOFIT version (no 'nonlinear' flag here)
        pars.NonLinearModel.set_params(halofit_version=halofit_version)

        # 4) Build interpolator, passing the nonlinear flag here
        PK = camb.get_matter_power_interpolator(pars,
                                                zmin=z, zmax=z, nz_step=1, zs=[z],
                                                kmax=kmax,
                                                nonlinear=nonlinear,
                                                hubble_units=hubble_units,
                                                k_hunit=k_hunit,
                                                extrap_kmax=extrap_kmax,
                                                k_per_logint=k_per_logint)

        # 5) Evaluate at stored k
        return PK.P(z, k)
    
if __name__ == '__main__':

    k = np.logspace(1e-4, 1, 1000)
    fpt = FASTPT(k)
    handler = FPTHandler(fpt)
    handler.generate_power_spectra(method='classy', mode='single', omega_cdm=0.12, h=0.67, omega_b=0.022, z=0.0,
                                   As=2.1e-9, ns=0.965, halofit_version='mead', hubble_units=True,
                                   k_hunit=True, extrap_kmax=None, k_per_logint=None)