"""Main GPSSI sampling functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

from gpssi import covariance as c
from gpssi import kernel as k

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_covariance(
    img_shape: Tuple[int, ...],
    kernel: k.Kernel,
    cov_repr: str = "kron",
) -> c.CovarianceRepresentation:
    """Create and factorize a covariance representation for sampling.

    Args:
        img_shape: Shape of the image/grid.
        kernel: Kernel function for computing covariances.
        cov_repr: Type of covariance representation. Options:
            - "kron": Kronecker representation (memory efficient, recommended).
            - "full": Full matrix representation (only for small images).

    Returns:
        Factorized covariance representation ready for sampling.

    Raises:
        ValueError: If unknown covariance representation is requested.

    Example:
        >>> kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        >>> cov = get_covariance((100, 100), kernel, cov_repr="kron")
    """
    if cov_repr == "kron":
        cov = c.KroneckerCovariance()
    elif cov_repr == "full":
        cov = c.FullCovariance()
    else:
        msg = f'Unknown covariance representation "{cov_repr}". Use "kron" or "full".'
        raise ValueError(msg)

    cov.factorize_grid(img_shape, kernel)
    return cov


def get_sample(
    geo_map: NDArray[np.floating],
    cov: c.CovarianceRepresentation,
    noise_vec: Optional[NDArray[np.floating]] = None,
    *,
    return_geo_sample: bool = False,
) -> Union[NDArray[np.bool_], Tuple[NDArray[np.bool_], NDArray[np.floating]]]:
    """Generate a sample segmentation from geodesic map and covariance.

    Takes a geodesic distance map and adds spatially correlated noise
    according to the covariance structure, then thresholds at zero to
    produce a binary segmentation.

    Args:
        geo_map: Geodesic distance map (positive outside, negative inside).
        cov: Factorized covariance representation.
        noise_vec: Standard normal noise vector. If None, generated randomly.
        return_geo_sample: If True, also return the noisy geodesic map.

    Returns:
        If return_geo_sample is False:
            Binary segmentation sample.
        If return_geo_sample is True:
            Tuple of (segmentation, noisy geodesic map).

    Example:
        >>> geo_map = get_geodesic_map(image, segmentation, lmbda=0.9)
        >>> cov = get_covariance(image.shape, kernel)
        >>> sample = get_sample(geo_map, cov)
    """
    if noise_vec is None:
        rng = np.random.default_rng()
        noise_vec = rng.standard_normal(geo_map.size)

    var = cov.sample(noise_vec)

    geo_sample = geo_map.ravel() + var
    geo_sample = geo_sample.reshape(geo_map.shape)

    seg_sample = geo_sample <= 0

    if return_geo_sample:
        return seg_sample, geo_sample
    return seg_sample
