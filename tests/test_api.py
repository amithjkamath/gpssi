"""Tests for the public API and package imports."""

import numpy as np

import gpssi
from gpssi import (
    CovarianceRepresentation,
    FullCovariance,
    GeodesicMethod,
    Kernel,
    KroneckerCovariance,
    RbfKernel,
    get_covariance,
    get_geodesic_map,
    get_sample,
)


class TestPackageImports:
    """Test that package exports are correct."""

    def test_import_package(self):
        """Test basic package import."""
        assert hasattr(gpssi, "__version__")
        assert isinstance(gpssi.__version__, str)

    def test_import_main_functions(self):
        """Test import of main functions."""

        assert callable(get_covariance)
        assert callable(get_geodesic_map)
        assert callable(get_sample)

    def test_import_covariance_classes(self):
        """Test import of covariance classes."""

        assert CovarianceRepresentation is not None
        assert FullCovariance is not None
        assert KroneckerCovariance is not None

    def test_import_kernel_classes(self):
        """Test import of kernel classes."""

        assert Kernel is not None
        assert RbfKernel is not None

    def test_import_geodesic_method(self):
        """Test import of GeodesicMethod enum."""

        assert hasattr(GeodesicMethod, "FAST_MARCHING")
        assert hasattr(GeodesicMethod, "RASTER_SCAN")
        assert hasattr(GeodesicMethod, "EUCLIDEAN")

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""

        expected_exports = [
            "get_sample",
            "get_covariance",
            "get_geodesic_map",
            "CovarianceRepresentation",
            "KroneckerCovariance",
            "FullCovariance",
            "Kernel",
            "RbfKernel",
            "GeodesicMethod",
        ]

        for name in expected_exports:
            assert name in gpssi.__all__, f"{name} not in __all__"
            assert hasattr(gpssi, name), f"{name} not accessible from package"


class TestAPIUsage:
    """Test typical API usage patterns."""

    def test_basic_usage_example(self):
        """Test basic usage as shown in documentation."""

        # Create test data
        shape = (20, 20)
        img = np.random.rand(*shape).astype(np.float32)
        seg = np.zeros(shape, dtype=np.uint8)
        seg[8:12, 8:12] = 1

        # Get geodesic map
        geo_map = get_geodesic_map(img, seg, lmbda=0.9, method=GeodesicMethod.EUCLIDEAN)

        # Create kernel and covariance
        kernel = RbfKernel(w0=5, w1=5, eps=1e-8)
        cov = get_covariance(shape, kernel, cov_repr="kron")

        # Generate samples
        sample = get_sample(geo_map, cov)

        assert sample.shape == shape
        assert sample.dtype == np.bool_

    def test_multiple_samples(self):
        """Test generating multiple samples."""

        shape = (15, 15)
        img = np.random.rand(*shape).astype(np.float32)
        seg = np.zeros(shape, dtype=np.uint8)
        seg[5:10, 5:10] = 1

        geo_map = get_geodesic_map(img, seg, lmbda=0.5, method=GeodesicMethod.EUCLIDEAN)
        kernel = RbfKernel(w0=1, w1=3, eps=1e-8)
        cov = get_covariance(shape, kernel)

        # Generate 5 samples
        samples = [get_sample(geo_map, cov) for _ in range(5)]

        assert len(samples) == 5
        assert all(s.shape == shape for s in samples)

    def test_reproducible_sampling(self):
        """Test reproducible sampling with fixed noise."""

        shape = (10, 10)
        geo_map = np.random.randn(*shape)

        kernel = RbfKernel(w0=1, w1=3, eps=1e-8)
        cov = get_covariance(shape, kernel)

        # Fixed noise
        rng = np.random.default_rng(123)
        noise = rng.standard_normal(np.prod(shape))

        sample1 = get_sample(geo_map, cov, noise_vec=noise.copy())
        sample2 = get_sample(geo_map, cov, noise_vec=noise.copy())

        assert np.array_equal(sample1, sample2)
