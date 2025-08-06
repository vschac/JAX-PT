from .FP_JAXPT import JAXPT, configure_jax_for_platform, DEVICE_TYPE
from .device_utils import device_info, get_optimal_device_recommendations

__all__ = ['JAXPT', 'device_info', 'get_optimal_device_recommendations']
__version__ = '1.0.0'