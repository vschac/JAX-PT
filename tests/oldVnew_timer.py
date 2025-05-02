import numpy as np
from classy import Class
from time import time
from fastpt import FASTPT, FPTHandler
import random
import multiprocessing as mp
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

def generate_power_spectrum(omega_cdm=0.12, h=0.67, omega_b=0.022, z=0.0, k_min=1e-4, k_max=1.0, n_points=2000):
    params = {
        'output': 'mPk',
        'P_k_max_1/Mpc': k_max * 1.1,
        'z_max_pk': z,
        'h': h,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm
    }

    # Wavenumbers at which to compute P(k)
    k_values = np.logspace(np.log10(k_min), np.log10(k_max), n_points)

    # Initialize CLASS
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    # Compute the power spectrum
    pk_values = np.array([cosmo.pk(k, z) for k in k_values])

    # Clean up CLASS instance
    cosmo.struct_cleanup()
    cosmo.empty()
    
    return k_values, pk_values


def random_cosmology():
    """Generate random cosmological parameters within reasonable ranges"""
    return {
        'omega_cdm': random.uniform(0.1, 0.14),     # Vary CDM density
        'h': random.uniform(0.65, 0.75),            # Vary Hubble parameter
        'omega_b': random.uniform(0.02, 0.025),     # Vary baryon density
        'z': random.uniform(0.0, 1.0)               # Vary redshift
    }


if __name__ == "__main__":
    k_min, k_max, n_points = 1e-4, 1.0, 3000
    P_window = np.array([0.2, 0.2])
    C_window = 0.75
    
    # Initialize k array once (assuming we want the same k range for all spectra)
    k = np.logspace(np.log10(k_min), np.log10(k_max), n_points)
    
    # Initialize FASTPT once
    fpt = FASTPT(k, to_do=['all'], n_pad=int(0.5*len(k)))
    funcs = ["one_loop_dd", "one_loop_dd_bias", "one_loop_dd_bias_b3nl", "one_loop_dd_bias_lpt_NL", 
            "IA_tt", "IA_mix", "IA_ta", "OV", "kPol"]

    # cum_time = 0
    # for i in range(1000):
    #     for func_name in funcs:
    #         # Generate a new power spectrum with random cosmological parameters
    #         cosmo_params = random_cosmology()
    #         k, pk = generate_power_spectrum(
    #             omega_cdm=cosmo_params['omega_cdm'],
    #             h=cosmo_params['h'],
    #             omega_b=cosmo_params['omega_b'],
    #             z=cosmo_params['z'],
    #             k_min=k_min, 
    #             k_max=k_max, 
    #             n_points=n_points
    #         )

            
    #         func = getattr(fpt, func_name)
            
    #         start = time()
    #         func(pk, P_window, C_window)
    #         stop = time()
    #         cum_time += stop - start

    #         if i % 100 == 0:
    #             print(f"Memory usage: {get_memory_usage():.2f} MB")

    # print(f"Total cumulative time for direct run: {cum_time:.2f} seconds")

    fpt2 = FASTPT(k, to_do=['all'], n_pad=int(0.5*len(k))) #reinit to clear cash
    handler = FPTHandler(fpt2, P_window=P_window, C_window=C_window)
    spectra = []
    for i in range(100):
        cosmo_params = random_cosmology()
        k, pk = generate_power_spectrum(
            omega_cdm=cosmo_params['omega_cdm'],
            h=cosmo_params['h'],
            omega_b=cosmo_params['omega_b'],
            z=cosmo_params['z'],
            k_min=k_min, 
            k_max=k_max, 
            n_points=n_points
        )
        spectra.append(pk)
    print(f"Memory usage: {get_memory_usage():.2f} MB")
    start = time()
    handler.bulk_run(funcs, spectra, flip=True, verbose=True, n_jobs=mp.cpu_count())
    stop = time()
    print(f"Memory usage: {get_memory_usage():.2f} MB")
    print(f"Total cumulative time for bulk run: {stop - start:.2f} seconds")

