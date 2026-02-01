"""Tests for the kernel module."""

import numpy as np
import pytest

from gpssi.kernel import Kernel, RbfKernel, rbf


class TestRbfKernel:
    """Test cases for RBF kernel."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        kernel = RbfKernel(w0=1.0, w1=2.0)
        assert kernel.w0 == 1.0
        assert kernel.w1 == 2.0
        assert kernel.eps is None

    def test_init_with_eps(self):
        """Test initialization with epsilon."""
        kernel = RbfKernel(w0=1.0, w1=2.0, eps=1e-6)
        assert kernel.eps == 1e-6

    def test_init_invalid_w0(self):
        """Test that negative w0 raises error."""
        with pytest.raises(ValueError, match="w0 must be positive"):
            RbfKernel(w0=-1.0, w1=2.0)

    def test_init_invalid_w1(self):
        """Test that non-positive w1 raises error."""
        with pytest.raises(ValueError, match="w1 must be positive"):
            RbfKernel(w0=1.0, w1=0.0)

    def test_init_invalid_eps(self):
        """Test that negative eps raises error."""
        with pytest.raises(ValueError, match="eps must be non-negative"):
            RbfKernel(w0=1.0, w1=2.0, eps=-1e-6)

    def test_kernel_self_similarity(self):
        """Test that kernel returns w0 for identical points."""
        kernel = RbfKernel(w0=5.0, w1=2.0)
        x = np.array([[0.0, 0.0]])
        K = kernel(x, x)
        assert K.shape == (1, 1)
        assert np.isclose(K[0, 0], 5.0)

    def test_kernel_symmetry(self):
        """Test that kernel matrix is symmetric."""
        kernel = RbfKernel(w0=1.0, w1=2.0, eps=1e-8)
        rng = np.random.default_rng(42)
        x = rng.random((10, 3))
        K = kernel(x, x)
        assert np.allclose(K, K.T)

    def test_kernel_positive_definite(self):
        """Test that kernel matrix is positive definite."""
        kernel = RbfKernel(w0=1.0, w1=2.0, eps=1e-8)
        rng = np.random.default_rng(42)
        x = rng.random((10, 2))
        K = kernel(x, x)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > 0)

    def test_kernel_distance_decay(self):
        """Test that kernel value decreases with distance."""
        kernel = RbfKernel(w0=1.0, w1=2.0)
        x1 = np.array([[0.0, 0.0]])
        x2_close = np.array([[0.1, 0.1]])
        x2_far = np.array([[10.0, 10.0]])

        k_close = kernel(x1, x2_close)[0, 0]
        k_far = kernel(x1, x2_far)[0, 0]

        assert k_close > k_far

    def test_kernel_length_scale(self):
        """Test that larger length scale gives slower decay."""
        x1 = np.array([[0.0]])
        x2 = np.array([[5.0]])

        kernel_short = RbfKernel(w0=1.0, w1=1.0)
        kernel_long = RbfKernel(w0=1.0, w1=10.0)

        k_short = kernel_short(x1, x2)[0, 0]
        k_long = kernel_long(x1, x2)[0, 0]

        assert k_long > k_short

    def test_kernel_different_shapes(self):
        """Test kernel with different sized inputs."""
        kernel = RbfKernel(w0=1.0, w1=2.0)
        x1 = np.array([[0, 0], [1, 1], [2, 2]])
        x2 = np.array([[0, 0], [3, 3]])

        K = kernel(x1, x2)
        assert K.shape == (3, 2)

    def test_repr(self):
        """Test string representation."""
        kernel = RbfKernel(w0=1.0, w1=2.0, eps=1e-8)
        repr_str = repr(kernel)
        assert "RbfKernel" in repr_str
        assert "w0=1.0" in repr_str
        assert "w1=2.0" in repr_str


class TestRbfFunction:
    """Test cases for standalone rbf function."""

    def test_rbf_output_shape(self):
        """Test output shape of rbf function."""
        x1 = np.array([[0, 0], [1, 1]])
        x2 = np.array([[2, 2], [3, 3], [4, 4]])
        result = rbf(x1, x2, w0=1.0, w1=1.0)
        assert result.shape == (2, 3)

    def test_rbf_1d_input(self):
        """Test rbf with 1D input (automatically converts to 2D)."""
        x1 = np.array([[0], [1], [2]])
        x2 = np.array([[0], [1]])
        result = rbf(x1, x2, w0=1.0, w1=1.0)
        assert result.shape == (3, 2)

    def test_rbf_diagonal_eps(self):
        """Test that diagonal eps is added correctly."""
        x = np.array([[0, 0], [1, 1]])
        eps = 0.1
        result = rbf(x, x, w0=1.0, w1=10.0, diag_eps=eps)

        # Diagonal elements should include eps
        assert result[0, 0] > 1.0  # w0 + eps
        assert result[1, 1] > 1.0

    def test_rbf_non_square_no_eps(self):
        """Test that eps is not added for non-square matrices."""
        x1 = np.array([[0, 0], [1, 1]])
        x2 = np.array([[2, 2]])
        result = rbf(x1, x2, w0=1.0, w1=1.0, diag_eps=0.1)
        # Should not raise error, eps should be ignored
        assert result.shape == (2, 1)


class TestKernelInheritance:
    """Test that RbfKernel properly inherits from Kernel."""

    def test_isinstance(self):
        """Test that RbfKernel is an instance of Kernel."""
        kernel = RbfKernel(w0=1.0, w1=2.0)
        assert isinstance(kernel, Kernel)
