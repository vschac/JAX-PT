# JAX-PT

JAX-PT is a rewrite of the [FAST-PT](https://github.com/jablazek/FAST-PT/) codebase to be compatible with JAX's autodifferentiation and JIT compilation tools. This tool can be integrated into full JAX data computation pipelines or used on its own. When compiled, the main Jax-PT (same as Fast-PT) functions can see a 5-100x speed increase on Fast-PT 4.0. (depending on the function) For more in depth examples on the features of functionality of Jax-PT, please see [examples](https://github.com/vschac/JAX-PT/tree/main/examples/jpt_example.py).


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