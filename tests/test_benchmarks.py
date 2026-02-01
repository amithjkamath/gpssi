"""Performance benchmark tests for gpssi package.

Run with: pytest tests/test_benchmarks.py --benchmark-only
"""

import numpy as np
import pytest

from gpssi import get_covariance, get_geodesic_map, get_sample
from gpssi.covariance import (
    KroneckerCovariance,
    FullCovariance,
    kronecker_matrix_vector_product,
)
from gpssi.geodesic import GeodesicMethod
from gpssi.kernel import RbfKernel


# Skip all benchmarks if pytest-benchmark is not available
pytest.importorskip("pytest_benchmark")


class TestKernelBenchmarks:
    """Benchmarks for kernel operations."""

    @pytest.mark.benchmark(group="kernel")
    def test_rbf_kernel_small(self, benchmark):
        """Benchmark RBF kernel for small inputs."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        x = np.random.randn(100, 2)

        result = benchmark(kernel, x, x)

        assert result.shape == (100, 100)

    @pytest.mark.benchmark(group="kernel")
    def test_rbf_kernel_medium(self, benchmark):
        """Benchmark RBF kernel for medium inputs."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        x = np.random.randn(500, 2)

        result = benchmark(kernel, x, x)

        assert result.shape == (500, 500)

    @pytest.mark.benchmark(group="kernel")
    def test_rbf_kernel_large(self, benchmark):
        """Benchmark RBF kernel for large inputs."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        x = np.random.randn(1000, 2)

        result = benchmark(kernel, x, x)

        assert result.shape == (1000, 1000)


class TestCovarianceBenchmarks:
    """Benchmarks for covariance operations."""

    @pytest.mark.benchmark(group="covariance-factorize")
    def test_kronecker_factorize_small(self, benchmark):
        """Benchmark Kronecker factorization for small grid."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (20, 20)

        def factorize():
            cov = KroneckerCovariance()
            cov.factorize_grid(shape, kernel)
            return cov

        result = benchmark(factorize)
        assert result.cov_kron_mats is not None

    @pytest.mark.benchmark(group="covariance-factorize")
    def test_kronecker_factorize_medium(self, benchmark):
        """Benchmark Kronecker factorization for medium grid."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (100, 100)

        def factorize():
            cov = KroneckerCovariance()
            cov.factorize_grid(shape, kernel)
            return cov

        result = benchmark(factorize)
        assert result.cov_kron_mats is not None

    @pytest.mark.benchmark(group="covariance-factorize")
    def test_kronecker_factorize_large(self, benchmark):
        """Benchmark Kronecker factorization for large grid."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (500, 500)

        def factorize():
            cov = KroneckerCovariance()
            cov.factorize_grid(shape, kernel)
            return cov

        result = benchmark(factorize)
        assert result.cov_kron_mats is not None

    @pytest.mark.benchmark(group="covariance-factorize")
    def test_full_factorize_small(self, benchmark):
        """Benchmark full factorization for small grid."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (20, 20)

        def factorize():
            cov = FullCovariance()
            cov.factorize_grid(shape, kernel)
            return cov

        result = benchmark(factorize)
        assert result.cov is not None

    @pytest.mark.benchmark(group="covariance-sample")
    def test_kronecker_sample_small(self, benchmark):
        """Benchmark Kronecker sampling for small grid."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (50, 50)
        cov = KroneckerCovariance()
        cov.factorize_grid(shape, kernel)
        noise = np.random.randn(np.prod(shape))

        result = benchmark(cov.sample, noise)

        assert result.shape == (np.prod(shape),)

    @pytest.mark.benchmark(group="covariance-sample")
    def test_kronecker_sample_medium(self, benchmark):
        """Benchmark Kronecker sampling for medium grid."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (100, 100)
        cov = KroneckerCovariance()
        cov.factorize_grid(shape, kernel)
        noise = np.random.randn(np.prod(shape))

        result = benchmark(cov.sample, noise)

        assert result.shape == (np.prod(shape),)

    @pytest.mark.benchmark(group="covariance-sample")
    def test_kronecker_sample_large(self, benchmark):
        """Benchmark Kronecker sampling for large grid."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (500, 500)
        cov = KroneckerCovariance()
        cov.factorize_grid(shape, kernel)
        noise = np.random.randn(np.prod(shape))

        result = benchmark(cov.sample, noise)

        assert result.shape == (np.prod(shape),)


class TestKroneckerProductBenchmarks:
    """Benchmarks for Kronecker matrix-vector product."""

    @pytest.mark.benchmark(group="kronecker-mvp")
    def test_kronecker_mvp_2d_small(self, benchmark):
        """Benchmark 2D Kronecker MVP small."""
        mats = [np.random.randn(50, 50), np.random.randn(50, 50)]
        x = np.random.randn(50 * 50)

        result = benchmark(kronecker_matrix_vector_product, mats, x)

        assert result.shape == (50 * 50,)

    @pytest.mark.benchmark(group="kronecker-mvp")
    def test_kronecker_mvp_2d_large(self, benchmark):
        """Benchmark 2D Kronecker MVP large."""
        mats = [np.random.randn(200, 200), np.random.randn(200, 200)]
        x = np.random.randn(200 * 200)

        result = benchmark(kronecker_matrix_vector_product, mats, x)

        assert result.shape == (200 * 200,)

    @pytest.mark.benchmark(group="kronecker-mvp")
    def test_kronecker_mvp_3d(self, benchmark):
        """Benchmark 3D Kronecker MVP."""
        mats = [
            np.random.randn(30, 30),
            np.random.randn(30, 30),
            np.random.randn(30, 30),
        ]
        x = np.random.randn(30 * 30 * 30)

        result = benchmark(kronecker_matrix_vector_product, mats, x)

        assert result.shape == (30 * 30 * 30,)


class TestGeodesicBenchmarks:
    """Benchmarks for geodesic distance computation."""

    @pytest.mark.benchmark(group="geodesic")
    def test_geodesic_euclidean_small(self, benchmark):
        """Benchmark Euclidean geodesic for small image."""
        img = np.random.rand(50, 50).astype(np.float32)
        seg = np.zeros((50, 50), dtype=np.uint8)
        seg[20:30, 20:30] = 1

        result = benchmark(
            get_geodesic_map, img, seg, lmbda=0.5, method=GeodesicMethod.EUCLIDEAN
        )

        assert result.shape == img.shape

    @pytest.mark.benchmark(group="geodesic")
    def test_geodesic_euclidean_medium(self, benchmark):
        """Benchmark Euclidean geodesic for medium image."""
        img = np.random.rand(100, 100).astype(np.float32)
        seg = np.zeros((100, 100), dtype=np.uint8)
        seg[40:60, 40:60] = 1

        result = benchmark(
            get_geodesic_map, img, seg, lmbda=0.5, method=GeodesicMethod.EUCLIDEAN
        )

        assert result.shape == img.shape

    @pytest.mark.benchmark(group="geodesic")
    @pytest.mark.slow
    def test_geodesic_raster_small(self, benchmark):
        """Benchmark raster scan geodesic for small image."""
        img = np.random.rand(30, 30).astype(np.float32)
        seg = np.zeros((30, 30), dtype=np.uint8)
        seg[12:18, 12:18] = 1

        result = benchmark(
            get_geodesic_map,
            img,
            seg,
            lmbda=0.5,
            iterations=2,
            method=GeodesicMethod.RASTER_SCAN,
        )

        assert result.shape == img.shape

    @pytest.mark.benchmark(group="geodesic")
    def test_geodesic_fast_marching_medium(self, benchmark):
        """Benchmark fast marching geodesic for medium image."""
        img = np.random.rand(100, 100).astype(np.float32)
        seg = np.zeros((100, 100), dtype=np.uint8)
        seg[40:60, 40:60] = 1

        result = benchmark(
            get_geodesic_map, img, seg, lmbda=0.5, method=GeodesicMethod.FAST_MARCHING
        )

        assert result.shape == img.shape


class TestEndToEndBenchmarks:
    """End-to-end workflow benchmarks."""

    @pytest.mark.benchmark(group="e2e")
    def test_full_workflow_small(self, benchmark):
        """Benchmark full sampling workflow for small image."""
        shape = (30, 30)

        def workflow():
            # Create test data
            img = np.random.rand(*shape).astype(np.float32)
            seg = np.zeros(shape, dtype=np.uint8)
            seg[10:20, 10:20] = 1

            # Compute geodesic map
            geo_map = get_geodesic_map(img, seg, lmbda=0.5, method=GeodesicMethod.EUCLIDEAN)

            # Setup covariance
            kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
            cov = get_covariance(shape, kernel, cov_repr="kron")

            # Generate sample
            sample = get_sample(geo_map, cov)
            return sample

        result = benchmark(workflow)
        assert result.shape == shape

    @pytest.mark.benchmark(group="e2e")
    def test_full_workflow_medium(self, benchmark):
        """Benchmark full sampling workflow for medium image."""
        shape = (100, 100)

        def workflow():
            # Create test data
            img = np.random.rand(*shape).astype(np.float32)
            seg = np.zeros(shape, dtype=np.uint8)
            seg[30:70, 30:70] = 1

            # Compute geodesic map (using Euclidean for speed)
            geo_map = get_geodesic_map(img, seg, lmbda=0.5, method=GeodesicMethod.EUCLIDEAN)

            # Setup covariance
            kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
            cov = get_covariance(shape, kernel, cov_repr="kron")

            # Generate sample
            sample = get_sample(geo_map, cov)
            return sample

        result = benchmark(workflow)
        assert result.shape == shape

    @pytest.mark.benchmark(group="e2e")
    def test_sampling_only_medium(self, benchmark):
        """Benchmark just the sampling step (pre-computed covariance)."""
        shape = (100, 100)

        # Pre-compute everything
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        cov = get_covariance(shape, kernel, cov_repr="kron")
        geo_map = np.random.randn(*shape).astype(np.float64)

        result = benchmark(get_sample, geo_map, cov)
        assert result.shape == shape

    @pytest.mark.benchmark(group="e2e")
    def test_multiple_samples_medium(self, benchmark):
        """Benchmark generating multiple samples."""
        shape = (80, 80)
        n_samples = 10

        # Pre-compute
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        cov = get_covariance(shape, kernel, cov_repr="kron")
        geo_map = np.random.randn(*shape).astype(np.float64)

        def generate_samples():
            samples = []
            for _ in range(n_samples):
                samples.append(get_sample(geo_map, cov))
            return samples

        result = benchmark(generate_samples)
        assert len(result) == n_samples


class TestScalingBenchmarks:
    """Benchmarks to test scaling behavior."""

    @pytest.mark.benchmark(group="scaling")
    @pytest.mark.parametrize("size", [25, 50, 100, 200])
    def test_kronecker_scaling(self, benchmark, size):
        """Test how Kronecker sampling scales with image size."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (size, size)
        cov = KroneckerCovariance()
        cov.factorize_grid(shape, kernel)
        noise = np.random.randn(size * size)

        result = benchmark(cov.sample, noise)

        assert result.shape == (size * size,)


class TestMemoryBenchmarks:
    """Tests to verify memory-efficient implementations."""

    def test_kronecker_vs_full_memory(self):
        """Compare memory usage of Kronecker vs Full representations."""
        import sys

        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (50, 50)

        # Kronecker
        kron_cov = KroneckerCovariance()
        kron_cov.factorize_grid(shape, kernel)

        # Full (only for comparison with small sizes)
        full_cov = FullCovariance()
        full_cov.factorize_grid(shape, kernel)

        # Estimate memory
        kron_mem = sum(m.nbytes for m in kron_cov.cov_kron_mats)
        full_mem = full_cov.cov.nbytes

        # Kronecker should use much less memory
        assert kron_mem < full_mem
        assert full_mem / kron_mem > 100  # Should be ~100x less for 50x50

    def test_kronecker_large_grid_feasibility(self):
        """Test that large grids are feasible with Kronecker representation."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (1000, 1000)  # 1 million pixels

        # This should complete quickly and not run out of memory
        cov = KroneckerCovariance()
        cov.factorize_grid(shape, kernel)

        # Memory should be reasonable (2 matrices of 1000x1000 doubles)
        total_mem = sum(m.nbytes for m in cov.cov_kron_mats)
        assert total_mem < 20_000_000  # Less than 20 MB

        # Sampling should also work
        noise = np.random.randn(np.prod(shape))
        sample = cov.sample(noise)
        assert sample.shape == (np.prod(shape),)
