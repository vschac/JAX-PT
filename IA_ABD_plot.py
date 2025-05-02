from __future__ import division
import numpy as np 
from matter_power_spt import one_loop
import FASTPT
from time import time 

# load the input power spectrum data 
d=np.genfromtxt('inputs/P_IA.dat',skip_header=1)
## note: for non-trimmed data, genfromtxt is required to deal with NaN values.

k=d[:,0]
P=d[:,1]

d_extend=np.genfromtxt('inputs/P_lin.dat',skip_header=1)
k=d_extend[:-1,0]
P=d_extend[:-1,1]

# use if you want to interpolate data 
#from scipy.interpolate import interp1d 
#power=interp1d(k,P)
#k=np.logspace(np.log10(k[0]),np.log10(k[-1]),3000)
#P=power(k)
#print d[:,0]-k


P_window=np.array([.2,.2])  
C_window=.65	
n_pad=1000
# initialize the FASTPT class		
fastpt=FASTPT.FASTPT(k,to_do=['IA_mix'],low_extrap=-6,high_extrap=4,n_pad=n_pad) 
	
	
t1=time()	
IA_A, IA_Btype2, IA_DEE, IA_DBB=fastpt.IA_mix(P,C_window=C_window) 
t2=time()
# print('execution time to make IA data'), t2-t1 


print('To make a one-loop power spectrum for IA', k.size, ' grid points, using FAST-PT takes ', t2-t1, 'seconds.')

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)

IA_A=IA_A[98:599]
IA_DEE=IA_DEE[98:599]
IA_Btype2=IA_Btype2[98:599]
IA_DBB=IA_DBB[98:599]
k=k[98:599]

fig=plt.figure(figsize=(16,10))

x1=10**(-2.5)
x2=10
ax1=fig.add_subplot(111)
ax1.set_ylim(1e-2,1e3)
ax1.set_xlim(x1,x2)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel(r'$P_{\rm IA,quad}(k)$ [Mpc/$h$]$^3$', size=25)
ax1.tick_params(axis='both', which='major', labelsize=25)
ax1.tick_params(axis='both', width=2, length=10)
ax1.tick_params(axis='both', which='minor', width=1, length=5)
ax1.xaxis.set_major_formatter(FormatStrFormatter('%2.2f'))
ax1.xaxis.labelpad = 20
ax1.set_xticklabels([])


ax1.plot(k,abs(IA_A), lw=4, color='black', label = r'$A_{\rm IA,quad}(k)$')
ax1.plot(k,abs(IA_Btype2), lw=4, color='red', label = r'$B_{\rm IA,quad}(k)$')
ax1.plot(k,IA_DEE, '--', lw=4, color='blue', label = r'$D_{\rm IA,quad}^{EE}(k)$')
ax1.plot(k,IA_DBB, '--', lw=4, color='green', label = r'$D_{\rm IA,quad}^{BB}(k)$')
plt.legend(loc=3,fontsize=25)
plt.grid()

# ax2=fig.add_subplot(212)
# ax2.set_xscale('log')
# ax2.set_xlabel(r'$k$ [$h$/Mpc]', size=25)

# ax2.set_ylim(-3e-5,3e-5)
# ax2.set_xlim(x1,x2)

# labels = [item.get_text() for item in ax2.get_yticklabels()]
# labels[1] = r'$-3\times 10^{-5}$'
# labels[2] = r'$-2\times 10^{-5}$'
# labels[3] = r'$-1\times 10^{-5}$'
# labels[4] = '0'
# labels[5] = r'$1\times 10^{-5}$'
# labels[6] = r'$2\times 10^{-5}$'
# labels[7] = r'$3\times 10^{-5}$'
# ax2.set_yticklabels(labels)

# ax2.tick_params(axis='both', which='major', labelsize=25)
# ax2.tick_params(axis='both', width=2, length=10)
# ax2.tick_params(axis='both', which='minor', width=1, length=5)
# ax2.xaxis.set_major_formatter(FormatStrFormatter('%2.2f'))


# ax2.xaxis.labelpad = 20


# ax2.plot(k,IA_E/(d[:,2]/2.)-1,lw=2, color='black')
# ax2.plot(k,IA_B/(d[:,4]/2.)-1,'--',lw=2, color='black')
# ax2.text(0.02, 0.07, 'fractional difference',transform=ax2.transAxes,verticalalignment='bottom', horizontalalignment='left', fontsize=25, bbox=dict(facecolor='white',edgecolor='black', pad=8.0))

# # plt.legend(loc=3,fontsize=30)
# plt.grid()

#plt.tight_layout()
plt.show()
# fig.savefig('IA_abd_plot.pdf')
