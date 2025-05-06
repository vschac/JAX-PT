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
    
    def compute_ia_tt_full(P):
        return jpt2.IA_tt(P, C_window=C_window)

    # Get function value and VJP function
    results, vjp_fn = vjp(compute_ia_tt_full, P)

    # Create tangent vectors and convert to tuple to match the output structure
    tangent_vectors = tuple(jnp.ones_like(result) for result in results)

    # Compute gradient with respect to P
    gradient = vjp_fn(tangent_vectors)[0]
    


    # # Timing the JAXPT implementation
    # t0 = time()
    # jpt.IA_tt(P, C_window=C_window)
    # t1 = time()
    
    # # Timing the FASTPT implementation
    # t2 = time()
    # jpt2.IA_tt(P, C_window=C_window)
    # t3 = time()
    
    # print(f"JAXPT Time: {t1 - t0} seconds")
    # print(f"FP_JAXPT Time: {t3 - t2} seconds")