from jax import numpy as jnp
from jax.numpy import pi, sin, log10, log, exp
from jax.scipy.signal import fftconvolve
from jax import config, jit
config.update("jax_enable_x64", True)

def P_13_reg(k, P):

    N = k.size
    n = jnp.arange(-N+1, N)
    dL = log(k[1]) - log(k[0])
    s = n * dL

    cut = 7
    high_mask = s > cut
    low_mask = s < -cut
    mid_high_mask = (s <= cut) & (s > 0)
    mid_low_mask = (s >= -cut) & (s < 0)
    zero_mask = (s == 0)

    Z = lambda r: (12./r**2 + 10. + 100.*r**2 - 42.*r**4 \
        + 3./r**3 * (r**2-1.)**3 * (7*r**2+2.) * log((r+1.)/jnp.absolute(r-1.))) * r
    Z_low = lambda r: (352./5. + 96./5./r**2 - 160./21./r**4 - 1376./1155./r**6 - 1952./5005./r**8) * r
    Z_high = lambda r: (928./5.*r**2 - 4512./35.*r**4 + 416./21.*r**6 + 2656./1155.*r**8) * r

    f = jnp.zeros_like(s)
    
    safe_exp_neg_s = jnp.where(jnp.abs(s) > 1e-10, exp(-s), 1e10)  # Avoid division by zero
    
    # Fill each region using masks
    f = jnp.where(low_mask, Z_low(safe_exp_neg_s), f)
    f = jnp.where(mid_low_mask, Z(safe_exp_neg_s), f)
    f = jnp.where(zero_mask, 80.0, f)
    f = jnp.where(mid_high_mask, Z(safe_exp_neg_s), f)
    f = jnp.where(high_mask, Z_high(safe_exp_neg_s), f)

    g = fftconvolve(P, f) * dL
    g_k = g[N-1:2*N-1]
    P_bar = 1./252. * k**3/(2*pi)**2 * P * g_k

    return P_bar

def Y1_reg_NL(k, P):

    N = k.size
    n = jnp.arange(-N+1, N)
    dL = log(k[1]) - log(k[0])
    s = n * dL

    cut = 7
    high_mask = s > cut
    low_mask = s < -cut
    mid_high_mask = (s <= cut) & (s > 0)
    mid_low_mask = (s >= -cut) & (s < 0)
    zero_mask = (s == 0)

    Z = lambda r: (1./126.)*(-6./r**2 + 22. + 22.*r**2 - 6.*r**4 \
        + 3./r**3 * (r**2-1.)**4 * log((r+1.)/jnp.absolute(r-1.))) * r
    Z_low = lambda r: (1./126.)*(256./5. - 768./35./r**2 + 256./105./r**4 + 256./1155./r**6 + 256./5005./r**8) * r
    Z_high = lambda r: (1./126.)*(256./5.*r**2 - 768./35.*r**4 + 256./105.*r**6 + 256./1155.*r**8) * r

    safe_exp_neg_s = jnp.where(jnp.abs(s) > 1e-10, exp(-s), 1e10)
    
    f = jnp.zeros_like(s)
    f = jnp.where(low_mask, Z_low(safe_exp_neg_s), f)
    f = jnp.where(mid_low_mask, Z(safe_exp_neg_s), f)
    f = jnp.where(zero_mask, 32./126., f)  # Value at s=0
    f = jnp.where(mid_high_mask, Z(safe_exp_neg_s), f)
    f = jnp.where(high_mask, Z_high(safe_exp_neg_s), f)

    g = fftconvolve(P, f) * dL
    g_k = g[N-1:2*N-1]
    P_bar = k**3/(2*pi)**2 * P * g_k

    return P_bar

def Y2_reg_NL(k, P):

    N = k.size
    n = jnp.arange(-N+1, N)
    dL = log(k[1]) - log(k[0])
    s = n * dL

    cut = 7
    high_mask = s > cut
    low_mask = s < -cut
    mid_high_mask = (s <= cut) & (s > 0)
    mid_low_mask = (s >= -cut) & (s < 0)
    zero_mask = (s == 0)

    Z = lambda r: (1./126.)*(-6./r**2 + 22. + 22.*r**2 - 6.*r**4 \
        + 3./r**3 * (r**2-1.)**4 * log((r+1.)/jnp.absolute(r-1.))) * r
    Z_low = lambda r: (1./126.)*(256./5. - 768./35./r**2 + 256./105./r**4 + 256./1155./r**6 + 256./5005./r**8) * r
    Z_high = lambda r: (1./126.)*(256./5.*r**2 - 768./35.*r**4 + 256./105.*r**6 + 256./1155.*r**8) * r

    safe_exp_neg_s = jnp.where(jnp.abs(s) > 1e-10, exp(-s), 1e10)
    
    f = jnp.zeros_like(s)
    f = jnp.where(low_mask, Z_low(safe_exp_neg_s), f)
    f = jnp.where(mid_low_mask, Z(safe_exp_neg_s), f)
    f = jnp.where(zero_mask, 32./126., f)  # Value at s=0
    f = jnp.where(mid_high_mask, Z(safe_exp_neg_s), f)
    f = jnp.where(high_mask, Z_high(safe_exp_neg_s), f)
    
    g = fftconvolve(P, f) * dL
    g_k = g[N-1:2*N-1]
    P_bar = k**3/(2*pi)**2 * P * g_k

    return P_bar

def P_IA_B(k, P):
    
    N = k.size
    n = jnp.arange(-N+1, N)
    dL = log(k[1]) - log(k[0])
    s = n * dL
    
    cut = 3
    high_mask = s > cut
    low_mask = s < -cut
    mid_high_mask = (s <= cut) & (s > 0)
    mid_low_mask = (s >= -cut) & (s < 0)
    zero_mask = (s == 0)
    
    Z1 = lambda r: ((2.* r * (225.- 600.* r**2 + 1198.* r**4 - 600.* r**6 + 225.* r**8) + \
                    225.* (r**2 - 1.)**4 * (r**2 + 1.) * log(jnp.absolute(r-1)/(r+1)) )/(20160.* r**3) - 29./315*r**2 )/2.
    Z1_high = lambda r: ((-16*r**4)/147. + (32*r**6)/441. - (16*r**8)/1617. - (64*r**10)/63063. - 16*r**12/63063. - (32*r**14)/357357. - (16*r**16)/415701. )/2.
    Z1_low = lambda r: (-16./147 - 16/(415701.*r**12) - 32/(357357.*r**10) - 16/(63063.*r**8) - 64/(63063.*r**6) - 16/(1617.*r**4) + 32/(441.*r**2) )/2.
    
    safe_exp_neg_s = jnp.where(jnp.abs(s) > 1e-10, exp(-s), 1e10)
    
    f = jnp.zeros_like(s)
    f = jnp.where(low_mask, Z1_low(safe_exp_neg_s) * safe_exp_neg_s, f)
    f = jnp.where(mid_low_mask, Z1(safe_exp_neg_s) * safe_exp_neg_s, f)
    f = jnp.where(zero_mask, -1./42., f)  # Value at s=0
    f = jnp.where(mid_high_mask, Z1(safe_exp_neg_s) * safe_exp_neg_s, f)
    f = jnp.where(high_mask, Z1_high(safe_exp_neg_s) * safe_exp_neg_s, f)
    
    g = fftconvolve(P, f) * dL
    g_k = g[N-1:2*N-1]
    IA_B = k**3/(2.*pi**2) * P * g_k
    
    return IA_B

def P_IA_deltaE2(k, P):
    
    N = k.size
    n = jnp.arange(-N+1, N)
    dL = log(k[1]) - log(k[0])
    s = n * dL
    
    cut = 3
    high_mask = s > cut
    low_mask = s < -cut
    mid_high_mask = (s <= cut) & (s > 0)
    mid_low_mask = (s >= -cut) & (s < 0)
    zero_mask = (s == 0)
    
    Z1 = lambda r: 30. + 146*r**2 - 110*r**4 + 30*r**6 + log(jnp.absolute(r-1.)/(r+1.))*(15./r - 60.*r + 90*r**3 - 60*r**5 + 15*r**7)
    Z1_high = lambda r: 256*r**2 - 256*r**4 + (768*r**6)/7. - (256*r**8)/21. - (256*r**10)/231. - (256*r**12)/1001. - (256*r**14)/3003.
    Z1_low = lambda r: 768./7 - 256/(7293.*r**10) - 256/(3003.*r**8) - 256/(1001.*r**6) - 256/(231.*r**4) - 256/(21.*r**2)
    
    safe_exp_neg_s = jnp.where(jnp.abs(s) > 1e-10, exp(-s), 1e10)
    
    f = jnp.zeros_like(s)
    f = jnp.where(low_mask, Z1_low(safe_exp_neg_s) * safe_exp_neg_s, f)
    f = jnp.where(mid_low_mask, Z1(safe_exp_neg_s) * safe_exp_neg_s, f)
    f = jnp.where(zero_mask, 96., f)  # Value at s=0 is 96
    f = jnp.where(mid_high_mask, Z1(safe_exp_neg_s) * safe_exp_neg_s, f)
    f = jnp.where(high_mask, Z1_high(safe_exp_neg_s) * safe_exp_neg_s, f)
    
    g = fftconvolve(P, f) * dL
    g_k = g[N-1:2*N-1]
    deltaE2 = k**3/(896.*pi**2) * P * g_k
    
    return deltaE2

def P_IA_13G(k, P):
    
    N = k.size
    n = jnp.arange(-N+1, N)
    dL = log(k[1]) - log(k[0])
    s = n * dL
    
    cut = 7
    high_mask = s > cut
    low_mask = s < -cut
    mid_high_mask = (s <= cut) & (s > 0)
    mid_low_mask = (s >= -cut) & (s < 0)
    zero_mask = (s == 0)
    
    Z1 = lambda r: (12/r**2 - 26 + 4*r**2 - 6*r**4 + (3/r**3)*(r**2-1)**3*(r**2+2)*log((r+1.)/jnp.absolute(r-1))) * r
    Z1_low = lambda r: (-224/5 + 1248/(35*r**2) - 608/(105*r**4) - 26/(21*r**6) - 4/(35*r**8) + 34/(21*r**10) - 4/(3*r**12)) * r
    Z1_high = lambda r: ((-32*r**2)/5 - (96*r**4)/7 + (352*r**6)/105 + (164*r**8)/105 - (58*r**10)/35 + (4*r**12)/21 + (2*r**14)/3) * r
    
    safe_exp_neg_s = jnp.where(jnp.abs(s) > 1e-10, exp(-s), 1e10)
    
    f = jnp.zeros_like(s)
    f = jnp.where(low_mask, Z1_low(safe_exp_neg_s), f)
    f = jnp.where(mid_low_mask, Z1(safe_exp_neg_s), f)
    f = jnp.where(zero_mask, -16.0, f)  # Value at s=0
    f = jnp.where(mid_high_mask, Z1(safe_exp_neg_s), f)
    f = jnp.where(high_mask, Z1_high(safe_exp_neg_s), f)
    
    g = fftconvolve(P, f) * dL
    g_k = g[N-1:2*N-1]
    deltaE2 = 1/84 * k**3/(2*pi)**2 * P * g_k
    
    return deltaE2

def P_IA_13F(k, P):
    
    N = k.size
    n = jnp.arange(-N+1, N)
    dL = log(k[1]) - log(k[0])
    s = n * dL
    
    cut = 7
    high_mask = s > cut
    low_mask = s < -cut
    mid_high_mask = (s <= cut) & (s > 0)
    mid_low_mask = (s >= -cut) & (s < 0)
    zero_mask = (s == 0)
    
    Z = lambda r: (12./r**2 + 10. + 100.*r**2 - 42.*r**4 \
        + 3./r**3 * (r**2-1.)**3 * (7*r**2+2.) * log((r+1.)/jnp.absolute(r-1.))) * r
    Z_low = lambda r: (352./5. + 96./0.5/r**2 - 160./21./r**4 - 526./105./r**6 + 236./35./r**8 - 50./21./r**10 - 4./3./r**12) * r
    Z_high = lambda r: (928./5.*r**2 - 4512./35.*r**4 + 416./21.*r**6 + 356./105.*r**8 + 74./35.*r**10 - 20./3.*r**12 + 14./3.*r**14) * r
    
    safe_exp_neg_s = jnp.where(jnp.abs(s) > 1e-10, exp(-s), 1e10)
    
    f = jnp.zeros_like(s)
    f = jnp.where(low_mask, Z_low(safe_exp_neg_s), f)
    f = jnp.where(mid_low_mask, Z(safe_exp_neg_s), f)
    f = jnp.where(zero_mask, 80.0, f)  # Value at s=0
    f = jnp.where(mid_high_mask, Z(safe_exp_neg_s), f)
    f = jnp.where(high_mask, Z_high(safe_exp_neg_s), f)
    
    g = fftconvolve(P, f) * dL
    g_k = g[N-1:2*N-1]
    P_bar = 1./252. * k**3/(2*pi)**2 * P * g_k
    
    return P_bar




def c_window(n, n_cut):
    n_right = n[-1] - n_cut
    n_left = n[0] + n_cut
    
    W = jnp.ones_like(n)
    
    right_mask = n > n_right
    theta_right = (n[-1] - n) / jnp.array(n[-1] - n_right - 1, dtype=float)
    right_window = theta_right - 1/(2*pi)*sin(2*pi*theta_right)
    
    left_mask = n < n_left
    theta_left = (n - n[0]) / jnp.array(n_left - n[0] - 1, dtype=float)
    left_window = theta_left - 1/(2*pi)*sin(2*pi*theta_left)
    
    W = jnp.where(right_mask, right_window, W)
    W = jnp.where(left_mask, left_window, W)
    
    return W

# def p_window(k, log_k_left, log_k_right):
#     log_k = jnp.log10(k)
    
#     max_log_k = jnp.max(log_k)
#     min_log_k = jnp.min(log_k)
    
#     log_k_left = min_log_k + log_k_left
#     log_k_right = max_log_k - log_k_right
    
#     mask_left = log_k <= log_k_left
#     mask_right = log_k >= log_k_right
    
#     # Extract elements that satisfy the masks
#     left = log_k[mask_left]
#     right = log_k[mask_right]

#     if left.size > 0:
#         x_left = (min_log_k - left) / (min_log_k - left[-1])
#         W_left = x_left - (1 / (2 * pi)) * sin(2 * pi * x_left)
#     else:
#         W_left = jnp.array([])  # Avoid shape mismatch

#     if right.size > 0:
#         x_right = (right - right[-1]) / (right[0] - max_log_k)
#         W_right = x_right - (1 / (2 * pi)) * sin(2 * pi * x_right)
#     else:
#         W_right = jnp.array([])  # Avoid shape mismatch

#     W = jnp.ones_like(k)

#     if W_left.size > 0:
#         W = W.at[mask_left].set(W_left)
#     if W_right.size > 0:
#         W = W.at[mask_right].set(W_right)

#     return W

def p_window(k, log_k_left, log_k_right):
    log_k = jnp.log10(k)
    
    max_log_k = jnp.max(log_k)
    min_log_k = jnp.min(log_k)
    
    log_k_left_val = min_log_k + log_k_left
    log_k_right_val = max_log_k - log_k_right
    
    W = jnp.ones_like(k)
    
    mask_left = log_k <= log_k_left_val
    mask_right = log_k >= log_k_right_val
    
    # LEFT WINDOW
    # Normalize from 0 at min_log_k to 1 at boundary
    x_left = jnp.where(
        mask_left,
        (min_log_k - log_k) / (min_log_k - log_k_left_val),
        0.0
    )
    
    # Apply window function 
    W_left = x_left - (1 / (2 * pi)) * sin(2 * pi * x_left)
    
    # RIGHT WINDOW
    # Normalize from 0 at max_log_k to 1 at boundary
    x_right = jnp.where(
        mask_right,
        (log_k - max_log_k) / (log_k_right_val - max_log_k),
        0.0
    )
    
    W_right = x_right - (1 / (2 * pi)) * sin(2 * pi * x_right)
    
    # Apply the windows to the appropriate regions
    W = jnp.where(mask_left, W_left, W)
    W = jnp.where(mask_right, W_right, W)
    
    return W


class jax_k_extend: 

    def __init__(self,k,low=None,high=None):
        # Initialize with original k 
        self.k = k.copy()  # Store original k explicitly
        self.DL = log(k[1])-log(k[0]) 
        
        if low is not None:
            if (low > log10(k[0])):
                low=log10(k[0])
                print('Warning, you selected a extrap_low that is greater than k_min. Therefore no extrapolation will be done.')
        
            low=10**low
            low=log(low)
            N=jnp.absolute(int((log(k[0])-low)/self.DL))
           
            if (N % 2 != 0 ):
                N=N+1 
            s=log(k[0]) -(jnp.arange(0,N)+1)*self.DL 
            s=s[::-1]
            self.k_min=k[0]
            self.k_low=exp(s) 
           
            self.k=jnp.append(self.k_low,k)
            self.id_extrap=jnp.where(self.k >=self.k_min)[0] 
        else:
            self.k_min = k[0]
            self.id_extrap = jnp.arange(len(k))  # Set default id_extrap
            

        if high is not None:
            if (high < log10(k[-1])):
                high=log10(k[-1])
                print('Warning, you selected a extrap_high that is less than k_max. Therefore no extrapolation will be done.')
                #raise ValueError('Error in P_extend.py. You can not request an extension to high k that is less than your input k_max.')
            
            high=10**high
            high=log(high)
            N=jnp.absolute(int((log(k[-1])-high)/self.DL))
            
            if (N % 2 != 0 ):
                N=N+1 
            s=log(k[-1]) + (jnp.arange(0,N)+1)*self.DL 
            self.k_max=k[-1]
            self.k_high=exp(s)
            self.k=jnp.append(self.k,self.k_high)
            self.id_extrap=jnp.where(self.k <= self.k_max)[0] 
        else:
            self.k_max = k[-1]
            # id_extrap is already set if neither high nor low is specified
            

        if (high is not None) & (low is not None):
            self.id_extrap=jnp.where((self.k <= self.k_max) & (self.k >=self.k_min))[0]
            
            
    def extrap_k(self):
        return self.k 
        
    def extrap_P_low(self,P):
        # If no low extension, return input
        if not hasattr(self, 'k_low') or self.k_low is None:
            return P
      
        ns=(log(P[1])-log(P[0]))/self.DL
        Amp=P[0]/self.k_min**ns
        P_low=self.k_low**ns*Amp
        return jnp.append(P_low,P) 

    def extrap_P_high(self,P):
        # If no high extension, return input
        if not hasattr(self, 'k_high') or self.k_high is None:
            return P
       
        ns=(log(P[-1])-log(P[-2]))/self.DL
        Amp=P[-1]/self.k_max**ns
        P_high=self.k_high**ns*Amp
        return jnp.append(P,P_high) 
    
    def PK_original(self,P): 
        return self.k[self.id_extrap], P[self.id_extrap]