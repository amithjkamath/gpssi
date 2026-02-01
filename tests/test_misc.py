"""Tests for miscellaneous utility functions."""

import numpy as np

from gpssi.misc import kron_mats_to_full


class TestKronMatsToFull:
    """Test cases for kron_mats_to_full function."""

    def test_single_matrix(self):
        """Test with single matrix (should return same matrix)."""
        A = np.array([[1, 2], [3, 4]])
        result = kron_mats_to_full([A])
        assert np.allclose(result, A)

    def test_two_matrices(self):
        """Test Kronecker product of two matrices."""
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[1, 2], [3, 4]])

        result = kron_mats_to_full([A, B])
        expected = np.kron(A, B)

        assert np.allclose(result, expected)

    def test_three_matrices(self):
        """Test Kronecker product of three matrices."""
        A = np.eye(2)
        B = np.array([[1, 2], [3, 4]])
        C = np.array([[5, 6], [7, 8]])

        result = kron_mats_to_full([A, B, C])
        expected = np.kron(np.kron(A, B), C)

        assert np.allclose(result, expected)

    def test_identity_matrices(self):
        """Test with identity matrices."""
        I2 = np.eye(2)
        I3 = np.eye(3)

        result = kron_mats_to_full([I2, I3])

        # Kronecker product of identities is identity
        assert result.shape == (6, 6)
        assert np.allclose(result, np.eye(6))

    def test_output_shape(self):
        """Test that output shape is product of input shapes."""
        A = np.random.randn(3, 3)
        B = np.random.randn(4, 4)
        C = np.random.randn(2, 2)

        result = kron_mats_to_full([A, B, C])

        expected_shape = (3 * 4 * 2, 3 * 4 * 2)
        assert result.shape == expected_shape
