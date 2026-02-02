"""Covariance matrix representations and operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
from scipy import linalg

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gpssi.kernel import Kernel


class CovarianceRepresentation(ABC):
    """Abstract base class for covariance matrix representations.

    Different representations trade off between memory usage and computational
    efficiency for different operations.
    """

    @abstractmethod
    def factorize_grid(self, shape: tuple[int, ...], kernel: Kernel) -> None:
        """Factorize covariance matrix for a regular grid.

        Args:
            shape: Shape of the grid (e.g., image dimensions).
            kernel: Kernel function to compute covariances.
        """

    @abstractmethod
    def sample(self, noise_vec: NDArray[np.floating]) -> NDArray[np.floating]:
        """Generate a sample by multiplying with noise vector.

        Args:
            noise_vec: Standard normal noise vector.

        Returns:
            Sample from the covariance distribution.
        """


class KroneckerCovariance(CovarianceRepresentation):
    """Kronecker product representation of covariance matrix.

    Uses the fact that covariance on a grid can be expressed as a Kronecker
    product of 1D covariance matrices. This dramatically reduces memory and
    computation for large grids.

    The covariance matrix is represented as:
        C = C_1 ⊗ C_2 ⊗ ... ⊗ C_d

    where C_i is the covariance matrix for dimension i.

    Args:
        cov_kron_mats: List of Cholesky factors for each dimension. Defaults to None.
    """

    def __init__(self, cov_kron_mats: Optional[list[NDArray[np.floating]]] = None) -> None:
        """Initialize Kronecker covariance representation."""
        self.cov_kron_mats = cov_kron_mats

    def factorize_grid(self, shape: tuple[int, ...], kernel: Kernel) -> None:
        """Compute Kronecker factorization for a grid.

        Args:
            shape: Shape of the grid.
            kernel: Kernel function.
        """
        self.cov_kron_mats = kronecker_grid_factorization(shape, kernel)

    def sample(self, noise_vec: NDArray[np.floating]) -> NDArray[np.floating]:
        """Generate sample using efficient Kronecker matrix-vector product.

        Args:
            noise_vec: Standard normal noise vector.

        Returns:
            Sample from the distribution.

        Raises:
            RuntimeError: If factorize_grid has not been called.
        """
        if self.cov_kron_mats is None:
            msg = "Must call factorize_grid before sampling"
            raise RuntimeError(msg)
        return kronecker_matrix_vector_product(self.cov_kron_mats, noise_vec)


class FullCovariance(CovarianceRepresentation):
    """Full covariance matrix representation.

    Stores the complete covariance matrix. Only suitable for small grids
    due to O(n^2) memory requirement.

    Args:
        cov: Cholesky factor of the covariance matrix. Defaults to None.
    """

    def __init__(self, cov: Optional[NDArray[np.floating]] = None) -> None:
        """Initialize full covariance representation."""
        self.cov = cov

    def factorize_grid(self, shape: tuple[int, ...], kernel: Kernel) -> None:
        """Compute full covariance matrix and its Cholesky factor.

        Args:
            shape: Shape of the grid.
            kernel: Kernel function.
        """
        self.cov = full_grid_factorization(shape, kernel)

    def sample(self, noise_vec: NDArray[np.floating]) -> NDArray[np.floating]:
        """Generate sample using matrix-vector multiplication.

        Args:
            noise_vec: Standard normal noise vector.

        Returns:
            Sample from the distribution.

        Raises:
            RuntimeError: If factorize_grid has not been called.
        """
        if self.cov is None:
            msg = "Must call factorize_grid before sampling"
            raise RuntimeError(msg)
        return self.cov @ noise_vec


def kronecker_matrix_vector_product(
    kron_matrices: list[NDArray[np.floating]], x: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Efficient Kronecker matrix-vector product.

    Computes (A_1 ⊗ A_2 ⊗ ... ⊗ A_d) @ x without forming the full matrix.

    Based on:
        Saatçi, Yunus. "Scalable inference for structured Gaussian process models."
        Diss. University of Cambridge, 2012.

    Args:
        kron_matrices: List of matrices in the Kronecker product.
        x: Vector to multiply.

    Returns:
        Result of the matrix-vector product.
    """
    x_res = x.copy()
    for kron_mat in reversed(kron_matrices):
        n = kron_mat.shape[0]
        x_m = x_res.reshape(x_res.size // n, n).T
        z = kron_mat @ x_m
        x_res = z.ravel()
    return x_res


def kronecker_grid_factorization(
    shape: tuple[int, ...], kernel: Kernel
) -> list[NDArray[np.floating]]:
    """Compute Kronecker factorization for a regular grid.

    Args:
        shape: Grid dimensions.
        kernel: Kernel function.

    Returns:
        List of Cholesky factors for each dimension.
    """
    kron_matrices = []
    for d in range(len(shape)):
        pos_d = np.arange(shape[d])[:, np.newaxis].astype(np.float64)
        cov_d = kernel(pos_d, pos_d)
        u_d = cast("NDArray[np.floating]", linalg.cholesky(cov_d, lower=False))
        kron_matrices.append(u_d)
    return kron_matrices


def full_grid_factorization(shape: tuple[int, ...], kernel: Kernel) -> NDArray[np.floating]:
    """Compute full covariance matrix for a grid.

    Args:
        shape: Grid dimensions.
        kernel: Kernel function.

    Returns:
        Cholesky factor of the covariance matrix.
    """
    ndim = len(shape)
    pos = np.indices(shape).transpose((*tuple(range(1, ndim + 1)), 0))
    pos_vec = pos.reshape(-1, ndim).astype(np.float64)
    cov = kernel(pos_vec, pos_vec)
    return cast("NDArray[np.floating]", linalg.cholesky(cov, lower=False))
