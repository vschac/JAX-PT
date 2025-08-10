# JAX-PT

JAX-PT is a rewrite of the [FAST-PT](https://github.com/jablazek/FAST-PT/) codebase to be compatible with JAX's autodifferentiation and JIT compilation tools. This code can be integrated into full JAX data computation pipelines or used on its own. When compiled, the main JAX-PT functions (same as Fast-PT) can see a 5-100x speed increase on Fast-PT 4.0. (depending on the function) For more in depth examples on the features of functionality of Jax-PT, please see [examples](https://github.com/vschac/JAX-PT/tree/main/examples/jpt_example.py).

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

### Default Installation
```bash
pip install jax-pt
```

#### Dev Installation:
```bash
pip install jax-pt[dev]
```

## GPU Usage

JAX-PT allows for you to specify a device to run your computations on. During init pass 'cpu', 'gpu', or any other jax.Device to the device kwarg:

```python
import jax
import jax.numpy as jnp
from jaxpt import JAXPT

# Check available devices
print("Available devices:", jax.devices())

k = jnp.logspace(-3, 1, 1000)

# Create JAXPT instance (defaults to CPU)
jpt = JAXPT(k, warmup="moderate")

# Specify to use GPU
jpt = JAXPT(k, warmup="moderate", device="gpu")

# Add a different jax Device
devices = jax.devices()
jpt = JAXPT(k, warmup="moderate", device=devices[0]) # or any index from devices list
```

Please remember to install the correct jax CUDA libraries for your CUDA version.
For example:

```bash
pip install jax[cuda12]
```
