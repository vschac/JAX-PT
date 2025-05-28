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


    from jaxpt.jax_utils import P_IA_B as jP_IA_B
    from jaxpt.jax_utils import Y1_reg_NL as jY1
    from fastpt.IA.IA_ABD import P_IA_B as fP_IA_B
    from fastpt.utils.matter_power_spt import Y1_reg_NL as fY1
    from jaxpt.FP_JAXPT import J_k_scalar
    jPs, _ = J_k_scalar(P, jpt.X_spt, jpt._static_config, jpt.k_extrap, jpt.k_final, jpt.id_pad, jpt.l, jpt.m,
                         P_window=P_window, C_window=C_window)
    fPs, _ = fpt.J_k_scalar(P, fpt.X_spt, -2, P_window=np.array([0.2,0.2]), C_window=C_window)
    jres = jP_IA_B(jpt.k_extrap, jPs)
    fres = fP_IA_B(fpt.k_extrap, fPs)
    print(f"Close enough? {np.allclose(jres, fres)}")
    print(f"Max difference: {np.max(np.abs(jres - fres))}")
    print(f"Relative difference: {np.max(np.abs((jres - fres) / jres))}")

    import matplotlib.pyplot as plt
    plt.plot(jpt.k_extrap, jres, label='JAXPT')
    plt.plot(fpt.k_extrap, fres, label='FASTPT', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.legend()
    plt.show()

