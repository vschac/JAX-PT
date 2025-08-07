import jax
from typing import Dict, Any

def device_info():
    """Show current JAX device configuration"""
    import jax
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.device_kind.lower() == 'gpu']
    
    print(f"Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        marker = " (default)" if i == 0 else ""
        print(f"  {i}: {device}{marker}")
    
    if gpu_devices:
        print(f"\nGPU acceleration: Available ({len(gpu_devices)} GPU(s))")
    else:
        print(f"\nGPU acceleration: Not available")
        
    return {
        'devices': devices,
        'gpu_available': len(gpu_devices) > 0,
        'default_device': devices[0]
    }

