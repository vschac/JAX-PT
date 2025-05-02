.. _quickstart:

Quick Start Guide
===============

Using FAST-PT is straightforward. Here's a simple example to get started:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from fastpt import FASTPT, FPTHandler

   #Define a k range
   k = np.logspace(1e-4, 1, 1000)

   # Initialize FASTPT
   fpt = FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5*len(k)))
   handler = FPTHandler(fpt)

   # Use the handler to generate a power spectrum
   P = handler.get_power_spectrum()
   
   # Calculate one-loop corrections
   P_1loop, P_components = fpt.one_loop_dd(P, C_window=0.75)

   # Plot the results
   plt.figure(figsize=(10, 7))
   plt.loglog(k, P, label='Linear P(k)')
   plt.loglog(k, P_1loop, label='One-loop P(k)')
   plt.xlabel('k [h/Mpc]')
   plt.ylabel('P(k) [(Mpc/h)³]')
   plt.legend()
   plt.show()