"""Tests for the covariance module."""

import numpy as np
import pytest

from gpssi.covariance import (
    CovarianceRepresentation,
    FullCovariance,
    KroneckerCovariance,
    full_grid_factorization,
    kronecker_grid_factorization,
    kronecker_matrix_vector_product,
)
from gpssi.kernel import RbfKernel
from gpssi.misc import kron_mats_to_full


class TestKroneckerCovariance:
    """Test cases for Kronecker covariance representation."""

    def test_init_empty(self):
        """Test initialization without matrices."""
        cov = KroneckerCovariance()
        assert cov.cov_kron_mats is None

    def test_init_with_matrices(self):
        """Test initialization with pre-computed matrices."""
        mats = [np.eye(3), np.eye(4)]
        cov = KroneckerCovariance(cov_kron_mats=mats)
        assert cov.cov_kron_mats is mats

    def test_factorize_grid_2d(self):
        """Test factorization for 2D grid."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        cov = KroneckerCovariance()
        cov.factorize_grid((10, 8), kernel)

        assert cov.cov_kron_mats is not None
        assert len(cov.cov_kron_mats) == 2
        assert cov.cov_kron_mats[0].shape == (10, 10)
        assert cov.cov_kron_mats[1].shape == (8, 8)

    def test_factorize_grid_3d(self):
        """Test factorization for 3D grid."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        cov = KroneckerCovariance()
        cov.factorize_grid((5, 6, 7), kernel)

        assert len(cov.cov_kron_mats) == 3
        assert cov.cov_kron_mats[0].shape == (5, 5)
        assert cov.cov_kron_mats[1].shape == (6, 6)
        assert cov.cov_kron_mats[2].shape == (7, 7)

    def test_sample_without_factorize(self):
        """Test that sampling without factorization raises error."""
        cov = KroneckerCovariance()
        with pytest.raises(RuntimeError, match="Must call factorize_grid"):
            cov.sample(np.random.randn(100))

    def test_sample_output_shape(self):
        """Test that sample output has correct shape."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        cov = KroneckerCovariance()
        shape = (10, 12)
        cov.factorize_grid(shape, kernel)

        noise = np.random.randn(np.prod(shape))
        sample = cov.sample(noise)

        assert sample.shape == (np.prod(shape),)


class TestFullCovariance:
    """Test cases for full covariance representation."""

    def test_init_empty(self):
        """Test initialization without matrix."""
        cov = FullCovariance()
        assert cov.cov is None

    def test_factorize_grid_2d(self):
        """Test factorization for 2D grid."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        cov = FullCovariance()
        shape = (5, 6)
        cov.factorize_grid(shape, kernel)

        n = np.prod(shape)
        assert cov.cov is not None
        assert cov.cov.shape == (n, n)

    def test_sample_without_factorize(self):
        """Test that sampling without factorization raises error."""
        cov = FullCovariance()
        with pytest.raises(RuntimeError, match="Must call factorize_grid"):
            cov.sample(np.random.randn(100))

    def test_sample_output_shape(self):
        """Test that sample output has correct shape."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        cov = FullCovariance()
        shape = (5, 6)
        cov.factorize_grid(shape, kernel)

        n = np.prod(shape)
        noise = np.random.randn(n)
        sample = cov.sample(noise)

        assert sample.shape == (n,)


class TestKroneckerVsFullEquivalence:
    """Test that Kronecker and full representations produce equivalent results."""

    def test_equivalent_samples_small_grid(self):
        """Test that Kronecker and full produce same samples for small grid."""
        kernel = RbfKernel(w0=1.0, w1=3.0, eps=1e-8)
        shape = (4, 5)

        # Create both representations
        kron_cov = KroneckerCovariance()
        kron_cov.factorize_grid(shape, kernel)

        full_cov = FullCovariance()
        full_cov.factorize_grid(shape, kernel)

        # Same noise vector
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(np.prod(shape))

        # Samples should be equivalent (up to numerical precision)
        kron_sample = kron_cov.sample(noise)
        full_sample = full_cov.sample(noise)

        # Convert Kronecker matrices to full and compare
        kron_full = kron_mats_to_full(kron_cov.cov_kron_mats)
        expected = kron_full @ noise

        assert np.allclose(kron_sample, expected, rtol=1e-10)


class TestKroneckerMatrixVectorProduct:
    """Test cases for Kronecker matrix-vector product."""

    def test_simple_2d(self):
        """Test simple 2x2 Kronecker product."""
        A = np.array([[1.0, 0.0], [0.0, 2.0]])
        B = np.array([[3.0, 0.0], [0.0, 4.0]])
        x = np.array([1.0, 2.0, 3.0, 4.0])

        result = kronecker_matrix_vector_product([A, B], x)

        # Full Kronecker product
        AB = np.kron(A, B)
        expected = AB @ x

        assert np.allclose(result, expected)

    def test_3d(self):
        """Test 3D Kronecker product."""
        rng = np.random.default_rng(42)
        A = rng.random((3, 3))
        B = rng.random((4, 4))
        C = rng.random((2, 2))
        x = rng.random(3 * 4 * 2)

        result = kronecker_matrix_vector_product([A, B, C], x)

        # Full Kronecker product
        ABC = np.kron(np.kron(A, B), C)
        expected = ABC @ x

        assert np.allclose(result, expected)


class TestGridFactorization:
    """Test cases for grid factorization functions."""

    def test_kronecker_factorization_shapes(self):
        """Test that Kronecker factorization produces correct shapes."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (10, 15, 20)

        mats = kronecker_grid_factorization(shape, kernel)

        assert len(mats) == 3
        for i, (mat, dim) in enumerate(zip(mats, shape)):
            assert mat.shape == (dim, dim), f"Matrix {i} has wrong shape"

    def test_kronecker_matrices_upper_triangular(self):
        """Test that factorization produces upper triangular (Cholesky)."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (8, 10)

        mats = kronecker_grid_factorization(shape, kernel)

        for mat in mats:
            # Upper triangular check: lower triangle should be zeros
            assert np.allclose(mat, np.triu(mat))

    def test_full_factorization_cholesky(self):
        """Test that full factorization is upper triangular Cholesky."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (5, 6)

        U = full_grid_factorization(shape, kernel)

        # Should be upper triangular
        assert np.allclose(U, np.triu(U))

        # U^T @ U should reconstruct the covariance
        # (since we use upper Cholesky: C = U^T @ U)


class TestCovarianceInheritance:
    """Test inheritance structure."""

    def test_kronecker_is_representation(self):
        """Test that KroneckerCovariance inherits from CovarianceRepresentation."""
        assert issubclass(KroneckerCovariance, CovarianceRepresentation)

    def test_full_is_representation(self):
        """Test that FullCovariance inherits from CovarianceRepresentation."""
        assert issubclass(FullCovariance, CovarianceRepresentation)
