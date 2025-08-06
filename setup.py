from setuptools import setup, find_packages
import os

def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

install_requires = [
    'numpy>=1.21.0',
    'scipy>=1.7.0',
    'fastpt>=3.0.0',
    'jax>=0.4.0',
    'jaxlib>=0.4.0',
]

# GPU-specific requirements
gpu_requires = [
    'jax[cuda12_pip]>=0.4.0',  # For CUDA 12
    'jaxlib[cuda12_pip]>=0.4.0',
]

# Optional dependencies for full functionality
extras_require = {
    'gpu': gpu_requires,
    'gpu-cuda11': ['jax[cuda11_pip]>=0.4.0', 'jaxlib[cuda11_pip]>=0.4.0'],
    'cosmo': ['jax-cosmo>=0.6.0'],
    'dev': ['pytest', 'memory_profiler', 'pytest-cov'],
    'all': gpu_requires + ['jax-cosmo>=0.6.0', 'pytest', 'memory_profiler'],
}

setup(
    name='jax-pt',
    version='1.0.0',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires='>=3.8',
)