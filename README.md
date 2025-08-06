# JAX-PT

**Convert to a JAX REAdme**

FAST-PT is a code to calculate quantities in cosmological perturbation theory
at 1-loop (including, e.g., corrections to the matter power spectrum). The code
utilizes Fourier methods combined with analytic expressions to reduce the
computation time to scale as N log N, where N is the number of grid points in
the input linear power spectrum.

NOTE: v3.1.0 and earlier require numpy version < 1.24. This is fixed in v3.1.1 and later, which is available on pip and conda.


## Installation

### CPU-only Installation (Default)
```bash
pip install jax-pt
```

### GPU Installation (Recommended for large computations)

#### NVIDIA GPU with CUDA 12:
```bash
pip install jax-pt[gpu]
```

#### NVIDIA GPU with CUDA 11:
```bash  
pip install jax-pt[gpu-cuda11]
```

#### Full Installation (GPU + Cosmology):
```bash
pip install jax-pt[all]
```

### Apple Silicon (M1/M2/M3):
```bash
pip install jax-pt
# Note: GPU acceleration not yet supported on Apple Silicon
```

## GPU Usage

JAX-PT automatically detects and uses GPU when available:

```python
import jax
from jaxpt import JAXPT

# Check available devices
print("Available devices:", jax.devices())

# Create JAXPT instance (auto-detects GPU)
k = jnp.logspace(-3, 1, 1000)
jpt = JAXPT(k, warmup="moderate")

# Check device info
print(jpt.get_device_info())
```

### Manual Device Selection

```python
# Force CPU usage
jpt = JAXPT(k, device='cpu')

# Force GPU usage (if available)
jpt = JAXPT(k, device='gpu')

# Use specific device
device = jax.devices('gpu')[0]
jpt = JAXPT(k, device=device)
```

### Memory Management

For large computations on GPU:

```python
# Monitor memory usage
info = jpt.get_device_info()
if info['memory_info']:
    print(f"GPU Memory: {info['memory_info']['used_mb']}/{info['memory_info']['total_mb']} MB")

# Clear caches if needed
jax.clear_caches()
```