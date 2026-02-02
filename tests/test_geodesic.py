"""Tests for the geodesic distance module."""

import numpy as np
import pytest

from gpssi.geodesic import (
    GeodesicMethod,
    _euclidean_distance,
    _fast_marching_geodesic,
    _geodesic_raster_scan,
    get_geodesic_map,
)


class TestGeodesicMap:
    """Test cases for get_geodesic_map function."""

    def test_2d_basic(self):
        """Test basic 2D geodesic map computation."""
        # Create simple test image and segmentation
        img = np.zeros((20, 20), dtype=np.float32)
        seg = np.zeros((20, 20), dtype=np.uint8)
        seg[5:15, 5:15] = 1  # Larger square in center

        geo_map = get_geodesic_map(img, seg, lmbda=0.5)

        assert geo_map.shape == img.shape
        # Deep inside should be negative (not on boundary)
        assert geo_map[10, 10] < 0  # Center of segmentation
        # Outside far from boundary should be positive
        assert np.all(geo_map[0, :] > 0)

    def test_2d_signed_distance(self):
        """Test that geodesic map produces signed distances."""
        img = np.ones((30, 30), dtype=np.float32) * 100
        seg = np.zeros((30, 30), dtype=np.uint8)
        seg[5:25, 5:25] = 1  # Larger square

        geo_map = get_geodesic_map(img, seg, lmbda=0.0)  # Pure euclidean

        # Check signs
        assert geo_map[15, 15] < 0  # Deep inside
        assert geo_map[0, 0] > 0  # Outside

    def test_3d_requires_spacing(self):
        """Test that 3D requires spacing parameter."""
        img = np.zeros((10, 10, 10), dtype=np.float32)
        seg = np.zeros((10, 10, 10), dtype=np.uint8)
        seg[4:6, 4:6, 4:6] = 1

        with pytest.raises(ValueError, match="Spacing is required"):
            get_geodesic_map(img, seg, lmbda=0.5)

    def test_3d_with_spacing(self):
        """Test 3D geodesic map with spacing."""
        img = np.zeros((15, 15, 15), dtype=np.float32)
        seg = np.zeros((15, 15, 15), dtype=np.uint8)
        seg[3:12, 3:12, 3:12] = 1  # Larger cube

        geo_map = get_geodesic_map(img, seg, lmbda=0.5, spacing=(1.0, 1.0, 1.0))

        assert geo_map.shape == img.shape
        assert geo_map[7, 7, 7] < 0  # Center should be inside (negative)
        assert geo_map[0, 0, 0] > 0  # Outside

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise error."""
        img = np.zeros((10, 10, 10, 10), dtype=np.float32)
        seg = np.zeros((10, 10, 10, 10), dtype=np.uint8)

        with pytest.raises(ValueError, match="2D or 3D"):
            get_geodesic_map(img, seg, lmbda=0.5)

    def test_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        img = np.zeros((20, 20), dtype=np.float32)
        seg = np.zeros((10, 10), dtype=np.uint8)

        with pytest.raises(ValueError, match="shapes must match"):
            get_geodesic_map(img, seg, lmbda=0.5)

    def test_euclidean_method(self):
        """Test Euclidean distance method."""
        img = np.zeros((30, 30), dtype=np.float32)
        seg = np.zeros((30, 30), dtype=np.uint8)
        seg[5:25, 5:25] = 1  # Larger square

        geo_map = get_geodesic_map(img, seg, method=GeodesicMethod.EUCLIDEAN)

        assert geo_map.shape == img.shape
        # Should still be signed
        assert geo_map[15, 15] < 0  # Inside
        assert geo_map[0, 0] > 0  # Outside

    def test_fast_marching_method(self):
        """Test fast marching method."""
        img = np.random.rand(30, 30).astype(np.float32)
        seg = np.zeros((30, 30), dtype=np.uint8)
        seg[5:25, 5:25] = 1  # Larger square

        geo_map = get_geodesic_map(img, seg, method=GeodesicMethod.FAST_MARCHING)

        assert geo_map.shape == img.shape
        # Should be signed
        assert geo_map[15, 15] < 0  # Inside
        assert geo_map[0, 0] > 0  # Outside

    def test_lambda_effect(self):
        """Test that lambda parameter affects result on non-uniform image."""
        # Create image with strong gradient
        img = np.zeros((30, 30), dtype=np.float32)
        img[:, 15:] = 255  # Strong edge in the middle

        seg = np.zeros((30, 30), dtype=np.uint8)
        seg[10:20, 10:20] = 1  # Square crossing the edge

        geo_low = get_geodesic_map(
            img, seg, lmbda=0.1, method=GeodesicMethod.FAST_MARCHING
        )
        geo_high = get_geodesic_map(
            img, seg, lmbda=0.9, method=GeodesicMethod.FAST_MARCHING
        )

        # With different lambda on a gradient image, results should differ
        # The high lambda gives more weight to image gradient
        assert not np.allclose(geo_low, geo_high)

    def test_custom_iterations(self):
        """Test custom iterations parameter."""
        img = np.zeros((20, 20), dtype=np.float32)
        seg = np.zeros((20, 20), dtype=np.uint8)
        seg[8:12, 8:12] = 1

        geo_2iter = get_geodesic_map(img, seg, lmbda=0.5, iterations=2)
        geo_10iter = get_geodesic_map(img, seg, lmbda=0.5, iterations=10)

        # More iterations should give more accurate (generally lower) distances
        # for raster scan method
        assert geo_2iter.shape == geo_10iter.shape


class TestEuclideanDistance:
    """Test cases for Euclidean distance function."""

    def test_simple_2d(self):
        """Test simple 2D Euclidean distance."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True  # Single seed point

        dist = _euclidean_distance(mask, np.array([1.0, 1.0]))

        assert dist.shape == (10, 10)
        assert dist[5, 5] == 0.0
        # Distance to corners
        assert dist[0, 0] > 0
        assert np.isclose(dist[5, 6], 1.0)  # One pixel away
        assert np.isclose(dist[6, 6], np.sqrt(2))  # Diagonal

    def test_anisotropic_spacing(self):
        """Test with anisotropic spacing."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True

        dist = _euclidean_distance(mask, np.array([1.0, 2.0]))  # Different spacing

        # Distance should reflect spacing
        assert np.isclose(dist[5, 6], 2.0)  # 1 pixel in y direction
        assert np.isclose(dist[6, 5], 1.0)  # 1 pixel in x direction


class TestRasterScan:
    """Test cases for raster scan geodesic distance."""

    def test_uniform_image(self):
        """Test on uniform image (should approximate Euclidean)."""
        img = np.ones((15, 15), dtype=np.float64)
        seed = np.zeros((15, 15), dtype=bool)
        seed[7, 7] = True

        dist = _geodesic_raster_scan(
            img, seed, lmbda=0.0, iterations=4, spacing=np.array([1.0, 1.0])
        )

        assert dist.shape == (15, 15)
        assert dist[7, 7] == 0.0
        assert dist[0, 0] > 0

    def test_convergence_with_iterations(self):
        """Test that more iterations improve convergence."""
        img = np.ones((20, 20), dtype=np.float64)
        seed = np.zeros((20, 20), dtype=bool)
        seed[10, 10] = True
        spacing = np.array([1.0, 1.0])

        dist_1 = _geodesic_raster_scan(
            img, seed, lmbda=0.0, iterations=1, spacing=spacing
        )
        dist_4 = _geodesic_raster_scan(
            img, seed, lmbda=0.0, iterations=4, spacing=spacing
        )

        # With more iterations, distances should be more accurate (generally smaller)
        # for points far from seed
        far_point_1 = dist_1[0, 0]
        far_point_4 = dist_4[0, 0]

        # More iterations should give same or better (smaller) distance
        assert far_point_4 <= far_point_1 + 1e-10


class TestFastMarching:
    """Test cases for fast marching geodesic."""

    def test_basic(self):
        """Test basic fast marching computation."""
        img = np.random.rand(15, 15).astype(np.float64)
        seed = np.zeros((15, 15), dtype=bool)
        seed[7, 7] = True

        dist = _fast_marching_geodesic(
            img, seed, lmbda=0.5, spacing=np.array([1.0, 1.0])
        )

        assert dist.shape == (15, 15)
        assert dist[7, 7] == 0.0
        assert np.all(dist >= 0)


class TestGeodesicMethod:
    """Test cases for GeodesicMethod enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert GeodesicMethod.FAST_MARCHING.value == "fast_marching"
        assert GeodesicMethod.RASTER_SCAN.value == "raster_scan"
        assert GeodesicMethod.EUCLIDEAN.value == "euclidean"
