"""Miscellaneous utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def kron_mats_to_full(
    kron_mats: list[NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Convert Kronecker factors to full matrix.

    Computes the full Kronecker product: A_1 ⊗ A_2 ⊗ ... ⊗ A_d

    Warning: This can produce very large matrices. Only use for testing
    or small matrices.

    Args:
        kron_mats: List of matrices to combine via Kronecker product.

    Returns:
        Full Kronecker product matrix.

    Example:
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[5, 6], [7, 8]])
        >>> full = kron_mats_to_full([A, B])  # 4x4 matrix
    """
    full: NDArray[np.floating] = np.array([[1.0]])
    for kron_mat in kron_mats:
        full = np.kron(full, kron_mat)
    return full
