.. _changelog:

Changelog / Migration Guide
===========================

This document details the changes between the original FAST-PT implementation and the current version.

Major Changes
-------------

* Caching: FAST-PT now caches individual terms and intermediate calculations to speed up computation. 
* FPTHandler: The handler class has been introduced to improve the the user's ability to manage FAST-PT and provide many new convenience features that compliment the FAST-PT class. NOTE: the FPTHandler class is not a replacement for the FAST-PT class, but rather a wrapper that provides additional functionality. It is not necessary for computation.
* To_do list: The to_do list is no longer needed to initialize FAST-PT. The terms will now be calculated as needed and stored as a property of the FAST-PT class.


Minor Changes
-------------

* Simple flag: A new "simple" kwarg has been added to FAST-PT which will instead initialize an instance of FAST-PT simple.
* Private k: The input k is now "private" after initialization via Python's name mangling. This means that the user cannot change the value of k after initialization but can still access the value of k.
* Gamma functions cache: A seperate (and simpler) caching system has been implemented to cache gamma functions and save time on the calculation of the X terms, previously stored in the to_do list.
* Parameter validtion: The parameters P, P_window, and C_window are now validated at every function call to ensure that they have the proper traits needed for the calculation. This is done to prevent errors from propagating through the code and causing issues later on.
* N_pad default: If no n_pad is provided during initialization, the default value is now set to 0.5 * len(k). This is done to prevent errors from propagating through the code and causing issues later on.
* Nu deprecation: The nu parameter is now deprecated as it is no longer needed for initialization. It will always be set to -2.


Performance Improvements
------------------------

The improvement in performance of FAST-PT is going to varry largely with your use case. However, about half of the calculation done for most terms was redundant granting a two times speedup do to the new caching system.
FAST-PT also now calculates terms in a modular format. This means that the user can now choose to calculate only the terms they need, rather than all of the terms grouped into one FAST-PT function. 
This is done by using the FPTHandler class and the get method, or by calling compute_term with the necessary parameters for each term. 
This will greatly improve the performance of each FAST-PT function if your use case only requires a select few terms.

Scientific Updates
------------------

* Description of any changes to the underlying science/algorithms