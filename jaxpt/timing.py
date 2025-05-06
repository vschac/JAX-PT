from jaxpt.JAXPT import JAXPT
from jaxpt.FP_JAXPT import JAXPT as FP_JAXPT
import jax.numpy as jnp
from fastpt import FASTPT, FPTHandler
from time import time
import numpy as np
from jax import jit
import jaxpt
from jax import vjp


if __name__ == "__main__":
    data_path = 'tests/benchmarking/Pk_test.dat'
    d = np.loadtxt(data_path)
    P = d[:, 1]
    k = d[:, 0]
    P_window = jnp.array([0.2, 0.2])
    C_window = 0.75
    n_pad = int(0.5 * len(k))
    
    jpt = JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)
    jpt2 = FP_JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)
    fpt = FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)
    
    # # Define a function that returns the value you want to differentiate
    # def compute_ia_tt(P):
    #     # Get the result from your JAXPT model
    #     result = jpt2.IA_tt(P, C_window=C_window)
    #     # If the result is a tuple and you want to work with a specific component:
    #     return result[0]  # Adjust based on which component you need

    # # Evaluate the function and get the VJP function
    # result_value, vjp_fn = vjp(compute_ia_tt, P)

    # # Create a vector v matching the output shape
    # v = jnp.ones_like(result_value)

    # # Compute the gradient with respect to P
    # gradient = vjp_fn(v)[0]  # [0] because vjp_fn returns a tuple

    # print("Gradient shape:", gradient.shape)
    # print("First few gradient values:", gradient[:5])
    


    # Timing the JAXPT implementation
    t0 = time()
    jpt.IA_tt(P, C_window=C_window)
    t1 = time()
    
    # Timing the FASTPT implementation
    t2 = time()
    jpt2.IA_tt(P)
    t3 = time()
    
    print(f"JAXPT Time: {t1 - t0} seconds")
    print(f"FP_JAXPT Time: {t3 - t2} seconds")