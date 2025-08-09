"""
JAX-PT: JAX-accelerated FAST-PT for computing perturbation theory power spectra
"""

from .JAXPT import JAXPT
from . import device_utils

__version__ = "1.0.0"
__author__ = "Vincent Schacknies"
__email__ = "vincent.schacknies@icloud.com"

__all__ = ["JAXPT", "device_utils"]