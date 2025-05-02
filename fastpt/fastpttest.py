from fastpt import FASTPT
from time import time
import matplotlib.pyplot as plt
import numpy as np
import os

    # Version check

    # load the data file
data_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'Pk_test.dat')
d = np.loadtxt(data_path)
    # declare k and the power spectrum
k = d[:, 0]
P = d[:, 1]

    # set the parameters for the power spectrum window and
    # Fourier coefficient window
    # P_window=np.array([.2,.2])
C_window = .75
    # document this better in the user manual

    # padding length
n_pad = int(0.5 * len(k))
    #	to_do=['one_loop_dd','IA_tt']
to_do = ['one_loop_dd']
    #	to_do=['dd_bias','IA_all']
    # to_do=['all']

    # initialize the FASTPT class
    # including extrapolation to higher and lower k
t1 = time()
fpt = FASTPT(k, to_do=to_do, low_extrap=-5, high_extrap=3, n_pad=n_pad)

t2 = time()
    # calculate 1loop SPT (and time the operation)
    # P_spt=fastpt.one_loop_dd(P,C_window=C_window)
P_lpt = fpt.one_loop_dd_bias_lpt_NL(P, C_window=C_window)
P_der = fpt.IA_der(P,C_window=C_window)

    # for M = 10**14 M_sun/h
b1L = 1.02817
b2L = -0.0426292
b3L = -2.55751
b1E = 1 + b1L

    # for M = 10**14 M_sun/h
    # b1_lag = 1.1631
    # b2_lag = 0.1162

    # [Ps, Pnb, Pb1L, Pb1L_2, Pb1L_b2L, Pb2L, Pb2L_2, Pb3L, Pb1L_b3L] = [P_lpt[0],P_lpt[1],P_lpt[2],P_lpt[3],P_lpt[4],P_lpt[5],P_lpt[6],P_lpt[7],P_lpt[8]]
[Ps, Pnb, Pb1L, Pb1L_2, Pb1L_b2L, Pb2L, Pb2L_2] = [P_lpt[0], P_lpt[1], P_lpt[2], P_lpt[3], P_lpt[4], P_lpt[5],
                                                       P_lpt[6]]

    # Pgg_lpt = (b1E**2)*P + Pnb + (b1L)*(Pb1L) + (b1L**2)*Pb1L_2 + (b1L*b2L)*Pb1L_b2L + (b2L)*(Pb2L) + (b2L**2)*Pb2L_2 + (b3L)*(Pb3L) + (b1L*b3L)*Pb1L_b3L
Pgg_lpt = (b1E ** 2) * P + Pnb + (b1L) * (Pb1L) + (b1L ** 2) * Pb1L_2 + (b1L * b2L) * Pb1L_b2L + (b2L) * (Pb2L) + (
            b2L ** 2) * Pb2L_2

    # print([pnb,pb1L,pb1L_2,pb2L,Pb1L_b2L])

t3 = time()
print('initialization time for', to_do, "%10.3f" % (t2 - t1), 's')
print('one_loop_dd recurring time', "%10.3f" % (t3 - t2), 's')

    # calculate tidal torque EE and BB P(k)
    # P_IA_tt=fastpt.IA_tt(P,C_window=C_window)
    # P_IA_ta=fastpt.IA_ta(P,C_window=C_window)
    # P_IA_mix=fastpt.IA_mix(P,C_window=C_window)
    # P_RSD=fastpt.RSD_components(P,1.0,C_window=C_window)
    # P_kPol=fastpt.kPol(P,C_window=C_window)
    # P_OV=fastpt.OV(P,C_window=C_window)
    # P_IRres=fastpt.IRres(P,C_window=C_window)
    # make a plot of 1loop SPT results


ax = plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'$P(k)$', size=30)
ax.set_xlabel(r'$k$', size=30)

ax.plot(k, P, label='linear')
# ax.plot(k,P_spt[0], label=r'$P_{22}(k) + P_{13}(k)$' )
ax.plot(k, Pgg_lpt, label='P_lpt')
ax.plot(k,P_der, label='P_der')

plt.legend(loc=3)
plt.grid()
plt.show()