import pytest
import numpy as np
import jax.numpy as jnp
from jaxpt import FP_JAXPT as JAXPT
import os
from fastpt import FASTPT, FPTHandler
from jax import vjp
import jax 

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
d = np.loadtxt(data_path)
P = d[:, 1]
k = d[:, 0]
P_window = jnp.array([0.2, 0.2])
C_window = 0.75

if __name__ == "__main__":
    jpt = JAXPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5 * len(k)), warmup=True)
    fpt = FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5 * len(k)))
    
    # jres = jpt.one_loop_dd_bias_b3nl(P, P_window=P_window, C_window=C_window)
    # fres = fpt._get_sig3nl(P, P_window=np.array([0.2,0.2]), C_window=C_window)

    # if len(jres) != len(fres):
    #     raise ValueError("The lengths of the results from JAXPT and FASTPT do not match.")
    # if isinstance(jres, tuple):
    #     for i in range(len(jres)):
    #         print(f"Component {i}:")
    #         print(f"Close enough? {np.allclose(jres[i], fres[i])}")
    #         print(f"Max difference: {np.max(np.abs(jres[i] - fres[i]))}")
    #         print(f"Relative difference: {np.max(np.abs((jres[i] - fres[i]) / jres[i]))}")
    # else:
    #     print(f"Close enough? {np.allclose(jres, fres)}")
    #     print(f"Max difference: {np.max(np.abs(jres - fres))}")
    #     print(f"Relative difference: {np.max(np.abs((jres - fres) / jres))}")


    # from jaxpt.jax_utils import P_IA_deltaE2 as jP_IA_deltaE2
    # from fastpt.IA.IA_ta import P_IA_deltaE2 as fP_IA_deltaE2
    # # jP = jpt.one_loop_dd_bias_b3nl(P, P_window=P_window, C_window=C_window)
    # # fP, _ = fpt.J_k_scalar(P, fpt.X_spt, -2, P_window=np.array([0.2,0.2]), C_window=C_window)
    # # print(np.allclose(jP, fP))
    # # print(np.allclose(jpt.k_extrap, fpt.k_extrap))
    # jres = jP_IA_deltaE2(k, P)
    # fres = fP_IA_deltaE2(k, P)
    # print(f"Close enough? {np.allclose(jres, fres)}")
    # print(f"Max difference: {np.max(np.abs(jres - fres))}")
    # print(f"Relative difference: {np.max(np.abs((jres - fres) / jres))}")

    # import matplotlib.pyplot as plt
    # # Data has no positve values so plot the negative values
    # plt.plot(k, jres, label='JAXPT')
    # plt.plot(k, fres, label='FASTPT', linestyle='--')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('k')
    # plt.ylabel('P(k)')
    # plt.legend()
    # plt.show()



    # from time import time

    # def compute_ia_mix(P):
    #     return jpt.IA_mix(P, P_window=P_window, C_window=C_window)
    # t0 = time()
    # result_value, vjp_fn = vjp(compute_ia_mix, P)
    # tangent_vectors = tuple(jnp.ones_like(result) for result in result_value)
    # gradient = vjp_fn(tangent_vectors)[0]  # [0] because vjp_fn returns a tuple
    # t1= time()
    # print(f"JAXPT vjp time: {t1 - t0:.4f} seconds")