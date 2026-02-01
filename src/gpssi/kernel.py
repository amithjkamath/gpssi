"""Kernel functions for Gaussian processes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Kernel(ABC):
    """Abstract base class for kernel functions.

    A kernel function computes the covariance between pairs of points.
    """

    @abstractmethod
    def __call__(
        self, x1: NDArray[np.floating], x2: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute kernel values between points.

        Args:
            x1: First set of points with shape (n, d).
            x2: Second set of points with shape (m, d).

        Returns:
            Kernel matrix with shape (n, m).
        """


class RbfKernel(Kernel):
    """Radial Basis Function (RBF) kernel, also known as squared exponential.

    The RBF kernel is defined as:
        k(x1, x2) = w0 * exp(-||x1 - x2||^2 / w1^2)

    Args:
        w0: Amplitude parameter (variance).
        w1: Length scale parameter.
        eps: Small value added to diagonal for numerical stability. Defaults to None.

    Example:
        >>> kernel = RbfKernel(w0=1.0, w1=2.0, eps=1e-8)
        >>> x = np.array([[0, 0], [1, 1]])
        >>> K = kernel(x, x)
    """

    def __init__(self, w0: float, w1: float, eps: Optional[float] = None) -> None:
        """Initialize RBF kernel with parameters."""
        if w0 <= 0:
            msg = f"w0 must be positive, got {w0}"
            raise ValueError(msg)
        if w1 <= 0:
            msg = f"w1 must be positive, got {w1}"
            raise ValueError(msg)
        if eps is not None and eps < 0:
            msg = f"eps must be non-negative, got {eps}"
            raise ValueError(msg)

        self.w0 = w0
        self.w1 = w1
        self.eps = eps

    def __call__(
        self, x1: NDArray[np.floating], x2: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute RBF kernel matrix.

        Args:
            x1: First set of points with shape (n, d).
            x2: Second set of points with shape (m, d).

        Returns:
            Kernel matrix with shape (n, m).
        """
        return rbf(x1, x2, self.w0, self.w1, self.eps)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RbfKernel(w0={self.w0}, w1={self.w1}, eps={self.eps})"


def rbf(
    x1: NDArray[np.floating],
    x2: NDArray[np.floating],
    w0: float,
    w1: float,
    diag_eps: Optional[float] = None,
) -> NDArray[np.floating]:
    """Compute RBF kernel matrix.

    Args:
        x1: First set of points with shape (n, d).
        x2: Second set of points with shape (m, d).
        w0: Amplitude parameter.
        w1: Length scale parameter.
        diag_eps: Small value added to diagonal for numerical stability.

    Returns:
        Kernel matrix with shape (n, m).
    """
    # Ensure 2D arrays
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)

    # Compute squared Euclidean distances
    x1_sq = np.sum(x1**2, axis=1, keepdims=True)
    x2_sq = np.sum(x2**2, axis=1, keepdims=True)
    sqdist = x1_sq + x2_sq.T - 2 * np.dot(x1, x2.T)

    # Ensure non-negative distances (numerical stability)
    sqdist = np.maximum(sqdist, 0)

    # Compute kernel
    result = w0 * np.exp(-sqdist / (w1**2))

    # Add diagonal regularization if specified
    if diag_eps is not None and result.shape[0] == result.shape[1]:
        result += np.eye(result.shape[0]) * diag_eps

    return result
