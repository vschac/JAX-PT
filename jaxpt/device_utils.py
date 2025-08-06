import jax
from typing import Dict, Any

def get_optimal_device_recommendations() -> Dict[str, Any]:
    """Get recommendations for optimal device usage"""
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.device_kind.lower() == 'gpu']
    
    recommendations = {
        'available_devices': [str(d) for d in devices],
        'gpu_available': len(gpu_devices) > 0,
        'recommended_device': str(gpu_devices[0]) if gpu_devices else str(devices[0]),
        'memory_considerations': {}
    }
    
    if gpu_devices:
        recommendations['gpu_memory_advice'] = [
            "Use warmup='minimal' for memory-constrained GPUs",
            "Consider smaller k arrays (< 2000 elements) for 8GB GPUs", 
            "Enable jax_gpu_memory_fraction < 1.0 if sharing GPU",
            "Use jax.clear_caches() between large computations"
        ]
    
    return recommendations

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

