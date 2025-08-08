# JAX-PT

JAX-PT is a rewrite of the [FAST-PT](https://github.com/jablazek/FAST-PT/) codebase to be compatible with JAX's autodifferentiation and JIT compilation tools. This tool can be integrated into full JAX data computation pipelines or used on its own. When compiled, the main JAX-PT (same as Fast-PT) functions can see a 5-100x speed increase on Fast-PT 4.0. (depending on the function) For more in depth examples on the features of functionality of Jax-PT, please see [examples](https://github.com/vschac/JAX-PT/tree/main/examples/jpt_example.py).

### FAST-PT
FAST-PT is a code to calculate quantities in cosmological perturbation theory
at 1-loop (including, e.g., corrections to the matter power spectrum). The code
utilizes Fourier methods combined with analytic expressions to reduce the
computation time to scale as N log N, where N is the number of grid points in
the input linear power spectrum.

[![arXiv:1603.04826](https://img.shields.io/badge/arXiv-1603.04826-b31b1b.svg)](https://arxiv.org/abs/1603.04826)
[![arXiv:1609.05978](https://img.shields.io/badge/arXiv-1609.05978-b31b1b.svg)](https://arxiv.org/abs/1609.05978)
[![arXiv:1708.09247](https://img.shields.io/badge/arXiv-1708.09247-b31b1b.svg)](https://arxiv.org/abs/1708.09247)



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

#### Full Installation (GPU + Dev):
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