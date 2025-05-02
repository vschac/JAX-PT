import numpy as np
import inspect
from pprint import pprint

class FPT:
    def __init__(self, k):
        self.k = k
    """Regular init code here"""

    def one_loop_dd(self, P, P_window=None, C_window=None):
        """Method code here"""
        return P, P_window, C_window
    
    def IA_ct(self, P, P_window=None, C_window=None):
        """Method code here"""
        return P, P_window, C_window
    
    def RSD_components(self, P, f, P_window=None, C_window=None):
        """Method code here"""
        return P, P_window, C_window
    




class FunctionHandler:
    def __init__(self, fastpt_instance, **params):
        self.fastpt = fastpt_instance
        self.default_params = self._validate_params(params) if params else {}
        self.cache = {}


    def _validate_params(self, params):
        """" Same function as before """
        #Would need to add checks for every possible parameter (f, nu, X, etc)
        #Unless we want to make it so only P, P_window, C_window passable at init
        return params


    def _get_function_params(self, func):
        """ Returns the required parameter names for a given FASTPT function. """
        signature = inspect.signature(func)
        return [param for param in signature.parameters if signature.parameters[param].default == inspect.Parameter.empty]


    def _cache_result(self, function_name, params, result):
        """ Stores results uniquely by function name and its specific parameters. """
        hashable_params = self._convert_to_hashable(params)
        self.cache[(function_name, hashable_params)] = result
    

    def _convert_to_hashable(self, params):
        hashable_params = []
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                hashable_params.append((k, tuple(v.flat)))
            else:
                hashable_params.append((k, v))
        return tuple(sorted(hashable_params))


    def run(self, function_name, **override_kwargs):
        """ Runs the selected function from FASTPT with validated parameters. """
        if not hasattr(self.fastpt, function_name):
            raise ValueError(f"Function '{function_name}' not found in FASTPT.")

        func = getattr(self.fastpt, function_name)
        required_params = self._get_function_params(func)

        if (override_kwargs): self._validate_params(override_kwargs)
        merged_params = {**self.default_params, **override_kwargs}
        print("merged: ", merged_params)
        missing_params = [p for p in required_params if p not in merged_params]
        if missing_params:
            raise ValueError(f"Missing required parameters for '{function_name}': {missing_params}. "
                             f"Please recall with the missing parameters.")

        # Check cache first
        param_tuple = self._convert_to_hashable(merged_params)
        if (function_name, param_tuple) in self.cache:
            print(f"Using cached result for {function_name} with parameters {merged_params}.")
            return self.cache[(function_name, param_tuple)]

        result = func(**merged_params)
        self._cache_result(function_name, merged_params, result)
        return result
    

    def clear_cache(self, function_name=None):
        """ Clears specific or all cached results. """
        if function_name:
            self.cache = {key: value for key, value in self.cache.items() if key[0] != function_name}
            print(f"Cache cleared for '{function_name}'.")
        else:
            self.cache.clear()
            print("Cache cleared for all functions.")

    def show_cache(self):
        pprint(self.cache)


    def list_available_functions(self):
        """ Returns a list of valid FASTPT functions. """
        print([f for f in dir(self.fastpt) if callable(getattr(self.fastpt, f)) and not f.startswith("__")])

    def get_k(self):
        return self.fastpt.k
    

if __name__ == "__main__":
    """
    The main purpose of the FunctionHandler class is to provide a cache (or multiple methods for caching)
    and ease the user experience through a clear syntax and storing redundant parameters to be used later.
        - Can pass in parameters at FuncionHandler definition to be pre validated
        - If new parameters are passed they need to be revalidated
        - Function result with those specific parameters is cached
        - Passing any parameters on the run call will override any stored parameters,
            not passing any required parameters will default to what is stored
        - TODO: Add a verbose flag and check how it would merge/conflict with FPT verbose
        - ?TODO?: Add functionality to update handler's instance of FAST-PT, currently it is pointing to an instance 
                so it would update automatically without needing a new object created. 
                (Should cache be deleted since there's a new k or other params)
        - ?TODO?: Should I implement a max limit (suggested 1000 entries) for the cache?
                Is there a potential case when that many computations would be needed?
    """

    k = np.logspace(-3, 1, 200)
    fpt = FPT(k)

    handler = FunctionHandler(fpt, P=np.array([1.0, 2.0, 3.0]), P_window=(0.1, 0.2), C_window=0.75)
    result = handler.run("one_loop_dd")
    print(result)
    r2 = handler.run("one_loop_dd")
    print(r2)

    #rIA = handler.run("IA_ct")
    #rIA2 = handler.run("IA_ct", P=np.array([4, 5, 6])) #Different P value so non cached output is used
    
    #Still backwards compatible
    #print(fpt.IA_ct(P=np.array([4, 5, 6])))

    # try:
    #     r4 = handler.run("RSD_components", P=(1, 3, 6))
    # except ValueError as e:
    #     print(e)