"""Geodesic distance computation for images.

This module provides geodesic distance computation without external dependencies,
replacing the GeodisTK library with scipy-based implementations.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GeodesicMethod(Enum):
    """Methods for computing geodesic distances."""

    FAST_MARCHING = "fast_marching"
    RASTER_SCAN = "raster_scan"
    EUCLIDEAN = "euclidean"


def get_geodesic_map(
    np_img: NDArray[np.floating],
    np_seg: Union[NDArray[np.integer], NDArray[np.bool_]],
    lmbda: float = 0.9,
    iterations: Optional[int] = None,
    spacing: Optional[Tuple[float, ...]] = None,
    *,
    method: GeodesicMethod = GeodesicMethod.RASTER_SCAN,
) -> NDArray[np.floating]:
    """Compute geodesic distance map from a segmentation.

    Computes signed geodesic distances where:
    - Positive values: outside the segmentation
    - Negative values: inside the segmentation

    Args:
        np_img: Input image (2D or 3D).
        np_seg: Binary segmentation mask.
        lmbda: Weight for image gradient (0=pure Euclidean, 1=pure geodesic).
            Defaults to 0.9.
        iterations: Number of iterations for raster scan method. Defaults to 2 for
            2D and 4 for 3D.
        spacing: Voxel spacing for 3D images. Required for 3D, ignored for 2D.
        method: Method for distance computation. Defaults to RASTER_SCAN.

    Returns:
        Signed geodesic distance map with same shape as input.

    Raises:
        ValueError: If image dimensions are invalid or spacing not provided for 3D.
    """
    if np_img.ndim not in (2, 3):
        msg = f"Image must be 2D or 3D, got {np_img.ndim}D"
        raise ValueError(msg)
    if np_img.shape != np_seg.shape:
        msg = f"Image and segmentation shapes must match: {np_img.shape} vs {np_seg.shape}"
        raise ValueError(msg)

    mask = np_seg.astype(bool)

    if np_img.ndim == 3:
        if spacing is None:
            msg = "Spacing is required for 3D images"
            raise ValueError(msg)
        if iterations is None:
            iterations = 4
        spacing_arr = np.array(spacing)
    else:
        if iterations is None:
            iterations = 2
        spacing_arr = np.array([1.0, 1.0])

    # Find the boundary of the mask for computing signed distance
    # The boundary is where the mask meets non-mask regions
    if np_img.ndim == 2:
        struct = np.ones((3, 3), dtype=bool)
    else:
        struct = np.ones((3, 3, 3), dtype=bool)

    # Erode the mask - boundary pixels are those in mask but not in eroded mask
    eroded = ndimage.binary_erosion(mask, structure=struct)
    boundary = mask & ~eroded

    if method == GeodesicMethod.EUCLIDEAN:
        # Pure Euclidean distance from boundary
        dist_outside = _euclidean_distance(boundary, spacing_arr)
        dist_inside = _euclidean_distance(boundary, spacing_arr)
        # Create signed distance: negative inside, positive outside
        np_geo = dist_outside.copy()
        np_geo[mask] = -dist_inside[mask]
    elif method == GeodesicMethod.RASTER_SCAN:
        # Geodesic distance using raster scan from boundary
        dist_outside = _geodesic_raster_scan(
            np_img, boundary, lmbda, iterations, spacing_arr
        )
        dist_inside = _geodesic_raster_scan(
            np_img, boundary, lmbda, iterations, spacing_arr
        )
        np_geo = dist_outside.copy()
        np_geo[mask] = -dist_inside[mask]
    elif method == GeodesicMethod.FAST_MARCHING:
        # Fast marching method from boundary
        dist_outside = _fast_marching_geodesic(np_img, boundary, lmbda, spacing_arr)
        dist_inside = _fast_marching_geodesic(np_img, boundary, lmbda, spacing_arr)
        np_geo = dist_outside.copy()
        np_geo[mask] = -dist_inside[mask]
    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)

    return np_geo


def _euclidean_distance(
    mask: NDArray[np.bool_], spacing: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Compute Euclidean distance transform.

    Args:
        mask: Binary mask (distance from True voxels).
        spacing: Voxel spacing.

    Returns:
        Euclidean distance transform.
    """
    return ndimage.distance_transform_edt(~mask, sampling=spacing)


def _geodesic_raster_scan(
    img: NDArray[np.floating],
    seed_mask: NDArray[np.bool_],
    lmbda: float,
    iterations: int,
    spacing: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute geodesic distance using raster scan method.

    This is a fast approximation of geodesic distance that considers
    image gradients as edge weights.

    Args:
        img: Input image.
        seed_mask: Seeds (True values are seeds with distance 0).
        lmbda: Weight for gradient term (0-1).
        iterations: Number of forward-backward scan iterations.
        spacing: Voxel spacing.

    Returns:
        Geodesic distance from seeds.
    """
    # Normalize image to [0, 1]
    img_normalized = img.astype(np.float64)
    img_min, img_max = img_normalized.min(), img_normalized.max()
    if img_max > img_min:
        img_normalized = (img_normalized - img_min) / (img_max - img_min)

    # Initialize distance map
    dist = np.full(img.shape, np.inf, dtype=np.float64)
    dist[seed_mask] = 0.0

    ndim = img.ndim

    # Generate neighbor offsets based on dimensionality
    if ndim == 2:
        # 8-connected neighborhood for 2D
        offsets: List[Tuple[int, ...]] = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        # 26-connected neighborhood for 3D
        offsets = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di != 0 or dj != 0 or dk != 0:
                        offsets.append((di, dj, dk))

    # Precompute spatial distances for each offset
    spatial_dists: List[float] = []
    for offset in offsets:
        offset_arr = np.array(offset) * spacing[: len(offset)]
        spatial_dists.append(float(np.sqrt(np.sum(offset_arr**2))))

    # Raster scan iterations
    for _ in range(iterations):
        # Forward scan
        dist = _raster_pass(
            dist, img_normalized, offsets, spatial_dists, lmbda, forward=True
        )
        # Backward scan
        dist = _raster_pass(
            dist, img_normalized, offsets, spatial_dists, lmbda, forward=False
        )

    return dist


def _raster_pass(
    dist: NDArray[np.floating],
    img: NDArray[np.floating],
    offsets: List[Tuple[int, ...]],
    spatial_dists: List[float],
    lmbda: float,
    *,
    forward: bool,
) -> NDArray[np.floating]:
    """Perform one raster scan pass.

    Args:
        dist: Current distance map.
        img: Normalized image.
        offsets: Neighbor offsets.
        spatial_dists: Spatial distances for each offset.
        lmbda: Gradient weight.
        forward: If True, scan forward; if False, scan backward.

    Returns:
        Updated distance map.
    """
    dist = dist.copy()
    shape = dist.shape
    ndim = len(shape)

    # Determine iteration order
    if forward:
        ranges = [range(s) for s in shape]
    else:
        ranges = [range(s - 1, -1, -1) for s in shape]

    # For 2D
    if ndim == 2:
        for i in ranges[0]:
            for j in ranges[1]:
                if dist[i, j] == 0:
                    continue
                current_val = img[i, j]

                for idx, offset in enumerate(offsets):
                    spatial_dist = spatial_dists[idx]
                    ni, nj = i + offset[0], j + offset[1]

                    if 0 <= ni < shape[0] and 0 <= nj < shape[1]:
                        neighbor_val = img[ni, nj]
                        grad_dist = abs(current_val - neighbor_val)

                        # Combined distance: spatial + gradient-weighted
                        edge_weight = (
                            1 - lmbda
                        ) * spatial_dist + lmbda * grad_dist * spatial_dist
                        new_dist = dist[ni, nj] + edge_weight

                        if new_dist < dist[i, j]:
                            dist[i, j] = new_dist

    # For 3D
    elif ndim == 3:
        for i in ranges[0]:
            for j in ranges[1]:
                for k in ranges[2]:
                    if dist[i, j, k] == 0:
                        continue
                    current_val = img[i, j, k]

                    for idx, offset in enumerate(offsets):
                        spatial_dist = spatial_dists[idx]
                        ni, nj, nk = i + offset[0], j + offset[1], k + offset[2]

                        if (
                            0 <= ni < shape[0]
                            and 0 <= nj < shape[1]
                            and 0 <= nk < shape[2]
                        ):
                            neighbor_val = img[ni, nj, nk]
                            grad_dist = abs(current_val - neighbor_val)

                            edge_weight = (
                                1 - lmbda
                            ) * spatial_dist + lmbda * grad_dist * spatial_dist
                            new_dist = dist[ni, nj, nk] + edge_weight

                            if new_dist < dist[i, j, k]:
                                dist[i, j, k] = new_dist

    return dist


def _fast_marching_geodesic(
    img: NDArray[np.floating],
    seed_mask: NDArray[np.bool_],
    lmbda: float,
    spacing: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute geodesic distance using fast marching-like method.

    Uses a hybrid approach: Euclidean distance weighted by image gradients.

    Args:
        img: Input image.
        seed_mask: Seeds (True values are seeds with distance 0).
        lmbda: Weight for gradient term.
        spacing: Voxel spacing.

    Returns:
        Geodesic distance from seeds.
    """
    # Compute Euclidean distance
    euc_dist = ndimage.distance_transform_edt(~seed_mask, sampling=spacing)

    # Compute gradient magnitude
    img_normalized = img.astype(np.float64)
    img_min, img_max = img_normalized.min(), img_normalized.max()
    if img_max > img_min:
        img_normalized = (img_normalized - img_min) / (img_max - img_min)

    # Compute gradient magnitude using Sobel filters
    if img.ndim == 2:
        grad_x = ndimage.sobel(img_normalized, axis=0)
        grad_y = ndimage.sobel(img_normalized, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    else:
        grad_x = ndimage.sobel(img_normalized, axis=0)
        grad_y = ndimage.sobel(img_normalized, axis=1)
        grad_z = ndimage.sobel(img_normalized, axis=2)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    # Normalize gradient magnitude
    grad_max = grad_mag.max()
    if grad_max > 0:
        grad_mag = grad_mag / grad_max

    # Weight distance by gradient (higher gradient = slower propagation)
    speed = 1.0 / (1.0 + lmbda * grad_mag)

    # Approximate geodesic distance
    geodesic_dist = euc_dist / speed
    geodesic_dist[seed_mask] = 0.0

    return geodesic_dist
