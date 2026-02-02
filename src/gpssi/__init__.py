"""GPSSI: Gaussian Process Sampling of Segmentation Images.

This package provides tools for sampling image segmentations using
Gaussian process methods for uncertainty quantification.

Based on the paper:
    Lê, Matthieu, et al. "Sampling image segmentations for uncertainty
    quantification." Medical image analysis 34 (2016): 42-51.
"""

from gpssi.covariance import (
    CovarianceRepresentation,
    FullCovariance,
    KroneckerCovariance,
)
from gpssi.geodesic import GeodesicMethod, get_geodesic_map
from gpssi.gpssi import get_covariance, get_sample
from gpssi.kernel import Kernel, RbfKernel

__all__ = [
    "CovarianceRepresentation",
    "FullCovariance",
    "GeodesicMethod",
    "Kernel",
    "KroneckerCovariance",
    "RbfKernel",
    "get_covariance",
    "get_geodesic_map",
    "get_sample",
]

__version__ = "0.2.0"
__author__ = "Alain Jungo"
__license__ = "Apache-2.0"
