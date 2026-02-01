"""Test configuration and fixtures for pytest."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Create a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_2d_image():
    """Create a simple 2D test image."""
    return np.random.rand(30, 30).astype(np.float32)


@pytest.fixture
def simple_2d_segmentation():
    """Create a simple 2D test segmentation (square in center)."""
    seg = np.zeros((30, 30), dtype=np.uint8)
    seg[10:20, 10:20] = 1
    return seg


@pytest.fixture
def simple_3d_image():
    """Create a simple 3D test image."""
    return np.random.rand(15, 15, 15).astype(np.float32)


@pytest.fixture
def simple_3d_segmentation():
    """Create a simple 3D test segmentation (cube in center)."""
    seg = np.zeros((15, 15, 15), dtype=np.uint8)
    seg[5:10, 5:10, 5:10] = 1
    return seg


@pytest.fixture
def default_kernel():
    """Create a default RBF kernel for testing."""
    from gpssi.kernel import RbfKernel

    return RbfKernel(w0=1.0, w1=5.0, eps=1e-8)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")
