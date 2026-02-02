"""Tests for the main gpssi module."""

import numpy as np
import pytest

from gpssi import get_covariance, get_sample
from gpssi.covariance import FullCovariance, KroneckerCovariance
from gpssi.kernel import RbfKernel


class TestGetCovariance:
    """Test cases for get_covariance function."""

    def test_kronecker_representation(self):
        """Test creating Kronecker covariance representation."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        cov = get_covariance((10, 10), kernel, cov_repr="kron")

        assert isinstance(cov, KroneckerCovariance)
        assert cov.cov_kron_mats is not None

    def test_full_representation(self):
        """Test creating full covariance representation."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        cov = get_covariance((5, 5), kernel, cov_repr="full")

        assert isinstance(cov, FullCovariance)
        assert cov.cov is not None

    def test_invalid_representation(self):
        """Test that invalid representation raises error."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)

        with pytest.raises(ValueError, match="Unknown covariance representation"):
            get_covariance((10, 10), kernel, cov_repr="invalid")

    def test_3d_covariance(self):
        """Test covariance for 3D grid."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        cov = get_covariance((5, 6, 7), kernel, cov_repr="kron")

        assert len(cov.cov_kron_mats) == 3


class TestGetSample:
    """Test cases for get_sample function."""

    def test_sample_shape(self):
        """Test that sample has correct shape."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (20, 25)
        cov = get_covariance(shape, kernel, cov_repr="kron")

        # Create dummy geodesic map
        geo_map = np.random.randn(*shape).astype(np.float64)

        sample = get_sample(geo_map, cov)

        assert sample.shape == shape
        assert sample.dtype == np.bool_

    def test_sample_is_binary(self):
        """Test that sample is binary."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (15, 15)
        cov = get_covariance(shape, kernel, cov_repr="kron")

        geo_map = np.random.randn(*shape).astype(np.float64)
        sample = get_sample(geo_map, cov)

        # Should only contain True/False
        unique_vals = np.unique(sample)
        assert len(unique_vals) <= 2
        assert all(v in [True, False] for v in unique_vals)

    def test_sample_with_custom_noise(self):
        """Test sample with custom noise vector."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (10, 10)
        cov = get_covariance(shape, kernel, cov_repr="kron")

        geo_map = np.zeros(shape)
        noise_vec = np.zeros(np.prod(shape))  # Zero noise

        sample = get_sample(geo_map, cov, noise_vec=noise_vec)

        # With zero geodesic map and zero noise, threshold at 0 should give all True
        assert np.all(sample)

    def test_sample_reproducibility(self):
        """Test that same noise gives same sample."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (12, 12)
        cov = get_covariance(shape, kernel, cov_repr="kron")

        geo_map = np.random.randn(*shape).astype(np.float64)

        rng = np.random.default_rng(42)
        noise1 = rng.standard_normal(np.prod(shape))

        rng = np.random.default_rng(42)
        noise2 = rng.standard_normal(np.prod(shape))

        sample1 = get_sample(geo_map, cov, noise_vec=noise1)
        sample2 = get_sample(geo_map, cov, noise_vec=noise2)

        assert np.array_equal(sample1, sample2)

    def test_return_geo_sample(self):
        """Test returning geodesic sample."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (10, 10)
        cov = get_covariance(shape, kernel, cov_repr="kron")

        geo_map = np.random.randn(*shape).astype(np.float64)

        result = get_sample(geo_map, cov, return_geo_sample=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

        seg_sample, geo_sample = result
        assert seg_sample.shape == shape
        assert geo_sample.shape == shape
        assert seg_sample.dtype == np.bool_
        assert np.issubdtype(geo_sample.dtype, np.floating)

    def test_geo_sample_threshold_consistency(self):
        """Test that segmentation is consistent with geo_sample threshold."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (15, 15)
        cov = get_covariance(shape, kernel, cov_repr="kron")

        geo_map = np.random.randn(*shape).astype(np.float64)

        seg_sample, geo_sample = get_sample(geo_map, cov, return_geo_sample=True)

        # Segmentation should match threshold
        expected_seg = geo_sample <= 0
        assert np.array_equal(seg_sample, expected_seg)

    def test_sample_3d(self):
        """Test sampling on 3D grid."""
        kernel = RbfKernel(w0=1.0, w1=5.0, eps=1e-8)
        shape = (5, 6, 7)
        cov = get_covariance(shape, kernel, cov_repr="kron")

        geo_map = np.random.randn(*shape).astype(np.float64)
        sample = get_sample(geo_map, cov)

        assert sample.shape == shape


class TestEndToEndSampling:
    """End-to-end tests for the sampling workflow."""

    def test_sampling_preserves_structure(self):
        """Test that samples roughly preserve segmentation structure."""
        kernel = RbfKernel(w0=1.0, w1=3.0, eps=1e-8)
        shape = (30, 30)
        cov = get_covariance(shape, kernel, cov_repr="kron")

        # Create a geodesic map: negative inside circle, positive outside
        y, x = np.ogrid[: shape[0], : shape[1]]
        center = (shape[0] // 2, shape[1] // 2)
        radius = 8
        dist_from_center = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
        geo_map = dist_from_center - radius  # Negative inside circle

        # Generate multiple samples and check they're roughly circular
        n_samples = 10
        samples = []
        for _ in range(n_samples):
            sample = get_sample(geo_map, cov)
            samples.append(sample)

        # All samples should have some foreground
        for sample in samples:
            assert sample.sum() > 0

        # Samples should vary due to randomness
        first_sample = samples[0]
        any_different = any(not np.array_equal(s, first_sample) for s in samples[1:])
        assert any_different

    def test_larger_variance_more_variation(self):
        """Test that larger kernel w0 produces more variation."""
        shape = (20, 20)

        # Create simple geodesic map
        geo_map = np.zeros(shape)
        geo_map[:10, :] = -1  # Top half inside
        geo_map[10:, :] = 1  # Bottom half outside

        # Small variance kernel
        kernel_small = RbfKernel(w0=0.1, w1=5.0, eps=1e-8)
        cov_small = get_covariance(shape, kernel_small, cov_repr="kron")

        # Large variance kernel
        kernel_large = RbfKernel(w0=10.0, w1=5.0, eps=1e-8)
        cov_large = get_covariance(shape, kernel_large, cov_repr="kron")

        # Generate samples and measure variation
        n_samples = 20
        rng = np.random.default_rng(42)

        samples_small = []
        samples_large = []

        for _ in range(n_samples):
            noise = rng.standard_normal(np.prod(shape))
            samples_small.append(get_sample(geo_map, cov_small, noise_vec=noise.copy()))
            samples_large.append(get_sample(geo_map, cov_large, noise_vec=noise.copy()))

        # Compute variance in samples (how much they differ from mean)
        var_small = np.var(samples_small, axis=0).mean()
        var_large = np.var(samples_large, axis=0).mean()

        # Larger kernel variance should produce more sample variation
        assert var_large > var_small
