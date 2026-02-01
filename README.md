# gpssi

[![CI](https://github.com/alainjungo/gpssi/actions/workflows/ci.yml/badge.svg)](https://github.com/alainjungo/gpssi/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/gpssi.svg)](https://badge.fury.io/py/gpssi)
[![Python Versions](https://img.shields.io/pypi/pyversions/gpssi.svg)](https://pypi.org/project/gpssi/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Gaussian Process Sampling of Segmentation Images - A Python library for sampling image segmentations with uncertainty quantification.

This repository provides code for the paper:

> Lê, Matthieu, et al. "Sampling image segmentations for uncertainty quantification." Medical image analysis 34 (2016): 42-51. doi:10.1016/j.media.2016.04.005

**Note:** This is not an official implementation by the authors and was motivated by the lack of publicly available code. Be aware that there might be differences to the original implementation.

## Features

- **Pure Python/NumPy/SciPy implementation** - No external dependencies like GeodisTK required
- **Multiple geodesic distance methods** - Euclidean, raster scan, and fast marching approximations
- **Memory-efficient Kronecker representation** - Handle large images without running out of memory
- **Type hints and modern Python** - Full type annotations for better IDE support
- **Comprehensive test suite** - Including performance benchmarks

## Installation

### From PyPI

```bash
pip install gpssi
```

### From Source

```bash
git clone https://github.com/alainjungo/gpssi.git
cd gpssi
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

Or using the Makefile:

```bash
make install-dev
```

## Quick Start

```python
import numpy as np
import gpssi

# Load your image and segmentation
img = np.random.rand(100, 100).astype(np.float32)  # Your image
seg = np.zeros((100, 100), dtype=np.uint8)
seg[30:70, 30:70] = 1  # Your segmentation

# Compute geodesic distance map
geo_map = gpssi.get_geodesic_map(img, seg, lmbda=0.9)

# Setup kernel and covariance
w1 = np.sqrt((seg > 0).sum() / np.pi)  # Length scale based on segmentation size
kernel = gpssi.RbfKernel(w0=5, w1=w1, eps=1e-8)
cov = gpssi.get_covariance(img.shape, kernel, cov_repr='kron')

# Generate samples
samples = []
for i in range(10):
    sample = gpssi.get_sample(geo_map, cov)
    samples.append(sample)
```

## Geodesic Distance Methods

The package provides three methods for computing geodesic distances:

```python
from gpssi import GeodesicMethod

# Pure Euclidean distance (fastest, ignores image content)
geo_map = gpssi.get_geodesic_map(img, seg, method=GeodesicMethod.EUCLIDEAN)

# Raster scan approximation (considers image gradients)
geo_map = gpssi.get_geodesic_map(img, seg, lmbda=0.9, method=GeodesicMethod.RASTER_SCAN)

# Fast marching approximation
geo_map = gpssi.get_geodesic_map(img, seg, lmbda=0.9, method=GeodesicMethod.FAST_MARCHING)
```

The `lmbda` parameter controls the weight between spatial distance (0) and image gradient (1).

## Implementation Details

### Geodesic Map

The geodesic map computation is now implemented in pure Python/NumPy/SciPy, replacing the previous dependency on GeodisTK. Three methods are available:

- **Euclidean**: Fast but ignores image content. Uses `scipy.ndimage.distance_transform_edt`.
- **Raster Scan**: Approximates geodesic distance by iterative forward/backward passes considering image gradients.
- **Fast Marching**: Hybrid approach using Euclidean distance weighted by gradient magnitude.

### Kronecker Factorization

For memory efficiency, the covariance matrix can be represented as a Kronecker product of smaller matrices. This is crucial for large images where the full covariance matrix would be prohibitively large.

Based on:
- Saatçi, Yunus. "Scalable inference for structured Gaussian process models." Diss. University of Cambridge, 2012.
- Gilboa, Elad, Yunus Saatçi, and John P. Cunningham. "Scaling multidimensional inference for structured Gaussian processes." IEEE TPAMI 37.2 (2013): 424-436.

## Development

### Using Make

```bash
# Create virtual environment and install dependencies
make install-dev

# Run linter
make lint

# Format code
make format

# Run type checker
make typecheck

# Run tests
make test

# Run tests with coverage
make test-cov

# Run performance benchmarks
make test-benchmark

# Clean temporary files
make clean

# Clean everything including venv
make clean-all
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run tests excluding slow tests
pytest tests/ -v -m "not slow"

# Run benchmarks
pytest tests/test_benchmarks.py --benchmark-only
```

## API Reference

### Main Functions

- `get_geodesic_map(img, seg, lmbda, ...)`: Compute signed geodesic distance map
- `get_covariance(shape, kernel, cov_repr)`: Create covariance representation
- `get_sample(geo_map, cov, noise_vec)`: Generate segmentation sample

### Classes

- `RbfKernel(w0, w1, eps)`: RBF (squared exponential) kernel
- `KroneckerCovariance`: Memory-efficient Kronecker representation
- `FullCovariance`: Full covariance matrix (for small images only)

### Enums

- `GeodesicMethod`: `EUCLIDEAN`, `RASTER_SCAN`, `FAST_MARCHING`

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{le2016sampling,
  title={Sampling image segmentations for uncertainty quantification},
  author={L{\^e}, Matthieu and Unkelbach, Jan and Ayber, Nicholas and others},
  journal={Medical image analysis},
  volume={34},
  pages={42--51},
  year={2016},
  publisher={Elsevier}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
